"""Local image transcription with RapidOCR or PaddleOCR."""

import io
import os
from functools import lru_cache

from .config import PROJECT_ROOT
from .errors import ConfigurationError, DocumentError


@lru_cache(maxsize=1)
def _engine():
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PROJECT_ROOT / ".cache" / "ocr-models"))
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise ConfigurationError(
            'OCR_MODE=paddle requires pip install -e ".[ocr]" and a supported Python version.'
        ) from exc
    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        device="cpu",
        cpu_threads=4,
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def transcribe(images):
    import numpy as np
    from PIL import Image

    from .document_loader import TextBlock

    blocks = []
    try:
        engine = _engine()
        for part in images:
            array = np.asarray(Image.open(io.BytesIO(part.data)).convert("RGB"))[:, :, ::-1]
            for result in engine.predict(array):
                text = "\n".join(result.get("rec_texts", []))
                if text.strip():
                    blocks.append(TextBlock(f"{part.source_id}:ocr", text))
    except ConfigurationError:
        raise
    except Exception as exc:
        raise DocumentError(
            "PaddleOCR transcription failed; check its installation/models."
        ) from exc
    return blocks


@lru_cache(maxsize=1)
def _rapid_engine():
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise ConfigurationError('OCR_MODE=rapid requires pip install -e ".[ocr-fast]".') from exc
    return RapidOCR(
        params={
            "Global.use_cls": False,
            "Global.log_level": "warning",
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )


def transcribe_rapid(images):
    import numpy as np
    from PIL import Image

    from .document_loader import TextBlock

    blocks = []
    try:
        engine = _rapid_engine()
        for part in images:
            array = np.asarray(Image.open(io.BytesIO(part.data)).convert("RGB"))[:, :, ::-1]
            result = engine(array, use_cls=False)
            lines = list(result.txts or [])
            if lines:
                blocks.append(TextBlock(f"{part.source_id}:ocr", "\n".join(lines)))
    except ConfigurationError:
        raise
    except Exception as exc:
        raise DocumentError("RapidOCR transcription failed; check its runtime/models.") from exc
    return blocks
