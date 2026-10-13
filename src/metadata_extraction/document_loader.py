import hashlib
import io
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings
from .errors import ConfigurationError, DocumentError
from .formats import IMAGE_EXTENSIONS, SUPPORTED_EXTENSIONS

LOADER_VERSION = "docx-visible-content-v4"


@dataclass(frozen=True)
class TextBlock:
    source_id: str
    text: str


@dataclass(frozen=True)
class ImagePart:
    source_id: str
    data: bytes
    mime_type: str = "image/png"


@dataclass(frozen=True)
class DocumentContent:
    filename: str
    sha256: str
    text_blocks: tuple[TextBlock, ...]
    images: tuple[ImagePart, ...]

    @property
    def source_ids(self) -> set[str]:
        return {b.source_id for b in self.text_blocks} | {p.source_id for p in self.images}


def _image_parts(data: bytes, source: str, settings: Settings) -> list[ImagePart]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                image = ImageOps.exif_transpose(image).convert("RGB")
                width, height = image.size
                parts = []
                top = 0
                while top < height:
                    bottom = min(top + settings.tile_height, height)
                    buffer = io.BytesIO()
                    image.crop((0, top, width, bottom)).save(buffer, format="PNG")
                    parts.append(ImagePart(f"{source}:y{top}-{bottom}", buffer.getvalue()))
                    if len(parts) > settings.max_image_parts:
                        raise DocumentError("Image exceeds MAX_IMAGE_PARTS; use smaller input.")
                    if bottom == height:
                        break
                    top = bottom - settings.tile_overlap
                return parts
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise DocumentError("Image is unreadable or too large to process safely.") from exc


def _visible_nodes(element, skip=()):
    """Walk Word's final view: keep insertions and omit deleted/moved-away content."""
    if element.tag in {qn("w:del"), qn("w:moveFrom"), *skip}:
        return
    yield element
    for child in element:
        yield from _visible_nodes(child, skip)


def _paragraph_text(element):
    # Paragraph.text omits runs wrapped in inline content controls or tracked insertions.
    parts = []
    for node in _visible_nodes(element, {qn("w:txbxContent")}):
        if node.tag == qn("w:t"):
            parts.append(node.text or "")
        elif node.tag == qn("w:tab"):
            parts.append("\t")
        elif node.tag in {qn("w:br"), qn("w:cr")}:
            parts.append("\n")
        elif node.tag == qn("w:noBreakHyphen"):
            parts.append("\u2011")
    return "".join(parts)


def _read_region(container, parent, prefix, settings, blocks, images, seen_images):
    """Read body/header/footer blocks, including textboxes and nested tables."""
    for index, element in enumerate(container, 1):
        source = f"{prefix}-{index}"
        if element.tag in {qn("w:del"), qn("w:moveFrom")}:
            continue
        seen_paragraphs = set()
        if element.tag.endswith("}altChunk"):
            raise DocumentError("DOCX contains an unsupported embedded document (altChunk).")
        if element.tag.endswith("}p"):
            text = _paragraph_text(element)
            seen_paragraphs.add(element)
        elif element.tag.endswith("}tbl"):
            table = Table(element, parent)
            rows = []
            for row in table.rows:
                cells = []
                for cell in row.cells:
                    paragraphs = [
                        node for node in _visible_nodes(
                            cell._tc, {qn("w:tbl"), qn("w:txbxContent")}
                        ) if node.tag == qn("w:p")
                    ]
                    seen_paragraphs.update(paragraphs)
                    cells.append("\n".join(_paragraph_text(p) for p in paragraphs))
                rows.append(" | ".join(cells))
            text = "\n".join(rows)
        else:
            # Content controls and other block wrappers may contain ordinary paragraphs.
            _read_region(element, parent, source, settings, blocks, images, seen_images)
            continue
        if text.strip():
            blocks.append(TextBlock(source, text))
        for box_index, box in enumerate(_visible_nodes(element), 1):
            if box.tag.endswith("}txbxContent"):
                for para_index, para in enumerate(_visible_nodes(box), 1):
                    if para.tag == qn("w:p") and para not in seen_paragraphs:
                        seen_paragraphs.add(para)
                        text = _paragraph_text(para)
                        if text.strip():
                            blocks.append(
                                TextBlock(f"{source}:textbox-{box_index}-{para_index}", text)
                            )
            elif box is not element and box.tag.endswith("}tbl"):
                # Outer cell.text does not include paragraphs from nested tables.
                for para_index, para in enumerate(_visible_nodes(box), 1):
                    if para.tag == qn("w:p") and para not in seen_paragraphs:
                        seen_paragraphs.add(para)
                        text = _paragraph_text(para)
                        if text.strip():
                            blocks.append(
                                TextBlock(f"{source}:nested-{box_index}-{para_index}", text)
                            )
        # Relationships are scoped to each Word part, not globally to the document.
        for node in _visible_nodes(element):
            relationship_ns = (
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
            )
            if node.tag.endswith("}blip"):
                rid = node.get(relationship_ns + "embed")
                if not rid and node.get(relationship_ns + "link"):
                    raise DocumentError("DOCX contains an externally linked image; embed it first.")
            elif node.tag.endswith("}imagedata"):
                rid = node.get(relationship_ns + "id")
            else:
                continue
            identity = (str(parent.part.partname), rid)
            if rid and identity not in seen_images:
                seen_images.add(identity)
                relationship = parent.part.rels[rid]
                if relationship.is_external:
                    raise DocumentError("DOCX contains an externally linked image; embed it first.")
                images.extend(
                    _image_parts(
                        relationship.target_part.blob, f"{source}:embedded-{rid}", settings
                    )
                )


def _read_pdf(data, settings, blocks, images):
    import pymupdf

    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise DocumentError("PDF is password protected; upload an unlocked copy.")
            if not pdf.page_count:
                raise DocumentError("PDF contains no pages.")
            # Bound parsing even for PDFs consisting of thousands of blank pages.
            if pdf.page_count > settings.max_image_parts:
                raise DocumentError("PDF page count exceeds MAX_IMAGE_PARTS.")
            for number, page in enumerate(pdf, 1):
                source = f"pdf-page-{number}"
                text = page.get_text("text", sort=True)
                for widget in page.widgets() or ():
                    if widget.field_value:
                        text += f"\n{widget.field_name}: {widget.field_value}"
                if text.strip():
                    blocks.append(TextBlock(f"{source}:text", text))
                if sum(len(b.text) for b in blocks) > settings.max_document_chars:
                    raise DocumentError("PDF exceeds MAX_DOCUMENT_CHARS; no text was truncated.")
                # Vision sees the entire page. Text models OCR scanned/mixed pages;
                # native text-only pages need no OCR or image rendering.
                if (
                    settings.effective_image_input == "vision"
                    or not text.strip()
                    or "\ufffd" in text
                    or page.get_image_info()
                ):
                    if page.rect.width * page.rect.height * 4 > 50_000_000:
                        raise DocumentError("PDF page is too large to render safely.")
                    pixmap = page.get_pixmap(dpi=144, colorspace=pymupdf.csRGB, alpha=False)
                    images.extend(_image_parts(pixmap.tobytes("png"), source, settings))
                    if len(images) > settings.max_image_parts:
                        raise DocumentError("PDF exceeds MAX_IMAGE_PARTS; no pages were dropped.")
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("PDF is corrupt, unsupported, or cannot be rendered.") from exc


def load_document(path: Path, settings: Settings) -> DocumentContent:
    try:
        return load_bytes(path.read_bytes(), path.name, settings)
    except OSError as exc:
        raise DocumentError(f"Cannot read input file: {path.name}") from exc


def load_bytes(data: bytes, filename: str, settings: Settings) -> DocumentContent:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentError("Supported input formats are DOCX, PDF, PNG, JPG and JPEG.")
    if not data:
        raise DocumentError("Input document is empty.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise DocumentError(f"Input exceeds {settings.max_upload_mb} MB.")
    text_blocks: list[TextBlock] = []
    images: list[ImagePart] = []
    if suffix in IMAGE_EXTENSIONS:
        signature = b"\x89PNG\r\n\x1a\n" if suffix == ".png" else b"\xff\xd8\xff"
        if not data.startswith(signature):
            raise DocumentError("Image content does not match its PNG/JPEG extension.")
        images.extend(_image_parts(data, "image-1", settings))
    elif suffix == ".pdf":
        if not data.lstrip().startswith(b"%PDF-"):
            raise DocumentError("File content is not a PDF document.")
        _read_pdf(data, settings, text_blocks, images)
    else:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if sum(info.file_size for info in archive.infolist()) > 100 * 1024 * 1024:
                    raise DocumentError("DOCX expanded contents exceed 100 MB.")
            doc = Document(io.BytesIO(data))
            seen_images = set()
            _read_region(doc.element.body, doc, "body", settings, text_blocks, images, seen_images)
            seen_parts = set()
            for kind in ("header", "footer"):
                for section in doc.sections:
                    for variant in (kind, f"first_page_{kind}", f"even_page_{kind}"):
                        region = getattr(section, variant)
                        if region.part.partname in seen_parts:
                            continue
                        seen_parts.add(region.part.partname)
                        _read_region(
                            region._element,
                            region,
                            str(region.part.partname),
                            settings,
                            text_blocks,
                            images,
                            seen_images,
                        )
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError(
                "DOCX is corrupt, unsupported, or contains unreadable images."
            ) from exc
    if not text_blocks and not images:
        raise DocumentError("Document contains no readable text or images.")
    if len(images) > settings.max_image_parts:
        raise DocumentError("Document exceeds MAX_IMAGE_PARTS; no content was silently dropped.")
    if sum(len(b.text) for b in text_blocks) > settings.max_document_chars:
        raise DocumentError("Document exceeds MAX_DOCUMENT_CHARS; no text was truncated.")
    image_input, ocr_mode = settings.effective_image_input, settings.effective_ocr_mode
    if images and image_input == "text" and ocr_mode == "off":
        raise ConfigurationError(
            "IMAGE_INPUT=text requires OCR_MODE=paddle or rapid for image documents."
        )
    if ocr_mode in {"paddle", "rapid"} and images:
        from .ocr import transcribe, transcribe_rapid

        ocr_blocks = transcribe(images) if ocr_mode == "paddle" else transcribe_rapid(images)
        if image_input == "text" and not ocr_blocks:
            raise DocumentError(
                "OCR found no readable image text; use vision input or another image."
            )
        text_blocks.extend(ocr_blocks)
    if sum(len(b.text) for b in text_blocks) > settings.max_document_chars:
        raise DocumentError("Document plus OCR exceeds MAX_DOCUMENT_CHARS; no text was truncated.")
    return DocumentContent(
        filename, hashlib.sha256(data).hexdigest(), tuple(text_blocks), tuple(images)
    )
