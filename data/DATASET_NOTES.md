# Dataset observations and reference corrections

## Files and coverage

The original agreement files, `train.csv` and `test.csv` are preserved unchanged.
`results_corrected.xlsx` is a separate revised reference file used only for evaluation.
The extraction model receives document content, never reference answers.

| Folder | Agreements | Formats | Original matching reference rows |
| --- | --- | --- | --- |
| `train/` | 10 | 7 DOCX, 3 PNG | 9 |
| `test/` | 4 | 2 DOCX, 2 PNG | 4 |

The original train CSV has ten rows, but `24158401-Rental-Agreement` belongs to the test
folder and is also present in the test CSV. The duplicate is consolidated in the corrected
workbook. Its `Consolidated` sheet contains thirteen unique document rows and seven columns.
`46239065-Standard-Rental-Agreement-Rental-With-Performance-Fee.docx` remains without a
reference answer in either version. It can be extracted but cannot contribute to recall.
The notes use numeric filename prefixes below; each maps to the corresponding agreement file.

## Observations

- The original rows for `54770958` and `54945838` are swapped relative to the scans.
- Original end dates include `31.11.2009`, `31.04.2011` and `31.02.2011`, which are invalid
  calendar dates. Duration-based corrections use calendar months and an inclusive end date.
- Several original names and one commencement date contain surrounding whitespace.
- The two Geraldine agreements identify the tenant with an O initial, while the labels use Q.
- `6683127` contains only the shortened contract; its longer counterpart `6683129` includes
  the 15-day termination clause. Related versions should remain together in future validation splits.
- `44737744` contains degraded text and embedded images. Empty corrected values represent
  a conservative reference decision for uncertain content; they are not proof that no value
  existed in a better source copy.
- `47854715` contains inconsistent number/word spellings in some clauses. The repeated rent
  wording supports 9000, but source inconsistencies remain a limitation.
- `95980236` has a signature-date heading from 2005 and a commencement clause from 2010.
  The reference follows the explicit commencement clause rather than the heading.
- Test filenames ending in `.pdf.docx` are DOCX files; matching removes both suffixes.
- Tall PNGs join several pages. Image-section identifiers describe pixel ranges rather than
  original page numbers.

## Correction log

This table records every changed field relative to the original CSV rows, including
formatting-only changes. Quoted values expose leading/trailing spaces. Blank reference cells
remain blank; the evaluator does not repair labels, swap rows or infer missing ground truth.
The workbook has not been rewritten as part of the documentation consolidation.

| Document | Field | Original | Corrected | Basis |
| --- | --- | --- | --- | --- |
| 6683127 | Termination Notice (Days) | `"15"` | (empty) | The shortened file stops before the ordinary termination clause; do not copy the 15-day clause from the longer version. |
| 6683127 | Party Two | `"GERALDINE Q. GALINATO"` | `"GERALDINE O. GALINATO"` | The party declaration spells GERALDINE O. GALINATO, not Q. |
| 6683129 | Party Two | `"GERALDINE Q. GALINATO"` | `"GERALDINE O. GALINATO"` | The party declaration spells GERALDINE O. GALINATO, not Q. |
| 18325926 | Agreement End Date | `"31.11.2009"` | `"04.11.2009"` | Eleven months from 05.12.2008, using the inclusive end-date convention, ends 04.11.2009; 31 November is invalid. |
| 18325926 | Party One | `"MR.K.Kuttan "` | `"MR.K.Kuttan"` | Remove leading/trailing whitespace; retain the underlying name or date. |
| 18325926 | Party Two | `"P.M. Narayana Namboodri "` | `"P.M. Narayana Namboodri"` | Remove leading/trailing whitespace; retain the underlying name or date. |
| 36199312 | Agreement End Date | `"31.04.2011"` | `"31.03.2011"` | The scan gives 01.05.2010 and an eleven-month term; the inclusive end date is 31.03.2011. 31 April is invalid. |
| 36199312 | Party One | `"Balaji.R "` | `"Balaji.R"` | Remove leading/trailing whitespace; retain the underlying name or date. |
| 44737744 | Agreement Value | `"3000"` | (empty) | The corrected workbook leaves this unreadable/uncertain amount empty rather than treating the original label as document evidence. |
| 44737744 | Agreement Start Date | `"20.09.2010"` | (empty) | The corrected workbook leaves this incomplete/uncertain commencement date empty. |
| 44737744 | Agreement End Date | `"19.07.2011"` | (empty) | A reliable end date cannot be derived from an uncertain commencement date; the corrected reference is empty. |
| 44737744 | Party One | `"M.V.V. VIJAYA SHANKAR"` | `"M. V. V. VIJAY A SHANKAR"` | The corrected workbook changes spacing in the landlord name. This is a transcription/formatting choice, not a claim of a different person. |
| 47854715 | Agreement End Date | `"31.02.2011"` | `"28.02.2011"` | The agreement commences 01.04.2010 for eleven months; the inclusive end date is 28.02.2011. 31 February is invalid. |
| 47854715 | Party One | `" P C MATHEW"` | `"P C MATHEW"` | Remove leading/trailing whitespace; retain the underlying name or date. |
| 47854715 | Party Two | `" L GOPINATH "` | `"L GOPINATH"` | Remove leading/trailing whitespace; retain the underlying name or date. |
| 50070534 | Agreement Start Date | `"01.04.2010 "` | `"01.04.2010"` | Remove leading/trailing whitespace; retain the underlying name or date. |
| 50070534 | Party One | `" P. JohnsonRavikumar"` | `"P. JohnsonRavikumar"` | Remove leading/trailing whitespace; retain the underlying name or date. |
| 50070534 | Party Two | `"Saravanan BV "` | `"Saravanan BV"` | Remove leading/trailing whitespace; retain the underlying name or date. |
| 54945838 | Agreement Value | `"5500"` | `"8000"` | Remap the swapped reference to the actual scan: 54770958 is Asha/Ramesh (5500); 54945838 is Parthasarathy/Bathini (8000). |
| 54945838 | Agreement Start Date | `"21.04.2011"` | `"01.04.2011"` | Remap the swapped reference to the actual scan: 54770958 is Asha/Ramesh (5500); 54945838 is Parthasarathy/Bathini (8000). |
| 54945838 | Agreement End Date | `"19.02.2012"` | `"31.03.2012"` | Remap the swapped reference to the actual scan: 54770958 is Asha/Ramesh (5500); 54945838 is Parthasarathy/Bathini (8000). |
| 54945838 | Termination Notice (Days) | `"60"` | `"90"` | Remap the swapped reference to the actual scan: 54770958 is Asha/Ramesh (5500); 54945838 is Parthasarathy/Bathini (8000). |
| 54945838 | Party One | `"Asha Ramesh & Ramesh K.N"` | `"K. Parthasarathy"` | Remap the swapped reference to the actual scan: 54770958 is Asha/Ramesh (5500); 54945838 is Parthasarathy/Bathini (8000). |
| 54945838 | Party Two | `"Sadasivuni Deepthi & Sadasivuni Kiran"` | `"Veerabrahmam Bathini"` | Remap the swapped reference to the actual scan: 54770958 is Asha/Ramesh (5500); 54945838 is Parthasarathy/Bathini (8000). |
| 54770958 | Agreement Value | `"8000"` | `"5500"` | Remap the swapped reference to the actual scan: 54770958 is Asha/Ramesh (5500); 54945838 is Parthasarathy/Bathini (8000). |
| 54770958 | Agreement Start Date | `"01.04.2011"` | `"20.04.2011"` | Remap the swapped reference to the actual scan: 54770958 is Asha/Ramesh (5500); 54945838 is Parthasarathy/Bathini (8000). The former starts 20.04.2011 and runs eleven months, ending 19.03.2012 inclusively. |
| 54770958 | Agreement End Date | `"31.03.2012"` | `"19.03.2012"` | Remap the swapped reference to the actual scan: 54770958 is Asha/Ramesh (5500); 54945838 is Parthasarathy/Bathini (8000). The former starts 20.04.2011 and runs eleven months, ending 19.03.2012 inclusively. |
| 54770958 | Termination Notice (Days) | `"90"` | `"60"` | Remap the swapped reference to the actual scan: 54770958 is Asha/Ramesh (5500); 54945838 is Parthasarathy/Bathini (8000). |
| 54770958 | Party One | `"K. Parthasarathy"` | `"Asha Ramesh & Ramesh K.N"` | Remap the swapped reference to the actual scan: 54770958 is Asha/Ramesh (5500); 54945838 is Parthasarathy/Bathini (8000). |
| 54770958 | Party Two | `"Veerabrahmam Bathini"` | `"Sadasivuni Deepthi & Sadasivuni Kiran"` | Remap the swapped reference to the actual scan: 54770958 is Asha/Ramesh (5500); 54945838 is Parthasarathy/Bathini (8000). |
| 24158401 | Party One | `"Hanumaiah "` | `"Hanumaiah"` | Remove leading/trailing whitespace; retain the underlying name or date. |
| 24158401 | Party Two | `" Vishal Bhardwaj "` | `"Vishal Bhardwaj"` | Remove leading/trailing whitespace; retain the underlying name or date. |
| 95980236 | Agreement End Date | `"31.03.2011"` | `"28.02.2011"` | The duration clause specifies eleven months from 01.04.2010; the inclusive end date is 28.02.2011. |
| 95980236 | Party One | `" S.Sakunthala"` | `"S.Sakunthala"` | Remove leading/trailing whitespace; retain the underlying name or date. |
| 156155545 | Party One | `"V.K.NATARAJ "` | `"V.K.NATARAJ"` | Remove leading/trailing whitespace; retain the underlying name or date. |
| 156155545 | Party Two | `" VYSHNAVI DAIRY SPECIALITIES Private Ltd"` | `"VYSHNAVI DAIRY SPECIALITIES Private Ltd"` | Remove leading/trailing whitespace; retain the underlying name or date. |
| 228094620 | Termination Notice (Days) | `"30"` | (empty) | The one-month clause concerns intentional damage, not ordinary termination; it is excluded under the chosen notice interpretation. |
| 228094620 | Party One | `" KAPIL MEHROTRA "` | `"KAPIL MEHROTRA"` | Remove leading/trailing whitespace; retain the underlying name or date. |
| 228094620 | Party Two | `".B.Kishore "` | `".B.Kishore"` | Remove leading/trailing whitespace; retain the underlying name or date. |

In addition to the field changes, the corrected workbook renames the notice header from
`Renewal Notice (Days)` to `Termination Notice (Days)`, consolidates the shared `24158401`
reference into one row, and stores all labelled documents in a single sheet. Other field
values remain as recorded in the original labels. The `Aggrement` header spelling is kept
for compatibility with the supplied CSVs.

## Notice interpretation

The switch from **Renewal Notice (Days)** to **Termination Notice (Days)** was a conscious
scope decision after reviewing the original CSV answers alongside the actual documents.
Explicit renewal-notice periods were uncommon, while most supplied notice answers reflected
ordinary termination/vacating notice. The agreements also contain other notice types,
including breach, property damage and eviction. The original column used overlapping
notice meanings rather than a consistent renewal-only definition.

Two examples show why these meanings should not be combined:

- `47854715` requires two months for ordinary termination and one month for negotiating
  renewal. Its original answer is 60 days, corresponding to termination rather than renewal.
- `228094620` includes one month's notice for intentional property damage. Its original
  answer is 30 days, but that is an exceptional breach-related clause, not ordinary
  termination or renewal notice. The revised termination reference is therefore empty.

This mix was interpreted as a reference-data inconsistency and corrected deliberately by
standardizing the application and revised workbook to **ordinary termination/vacating
notice**. Renewal deadlines and exceptional breach/eviction clauses are excluded. If the
available document does not support this field, its revised reference is left empty rather
than selecting an unrelated notice period. The shortened `6683127` document, for example,
does not contain the ordinary 15-day termination clause found in its longer counterpart.

This is a documented interpretation of the supplied data, not a claim that renewal and
termination are interchangeable or that the assignment formally changed its requirement.
The revised workbook uses `Termination Notice (Days)`; the original CSVs retain
`Renewal Notice (Days)` unchanged. The evaluator accepts both headers for compatibility,
but a header alias does not remove semantic disagreements in the original answers.

## Scoring and remaining limitations

The README reports strict recall under the assignment's literal exact-match definition,
normalized recall after formatting cleanup, and tolerant recall with guarded fuzzy name
matches. All three use the same denominator: labelled documents in the selected run,
including failed/skipped documents. Unrelated reference rows are excluded. Run documents
without references are listed as unscored. The two supplementary scores are distinct from
the assignment's strict metric.

A missing field in a returned row is incorrect unless its reference is empty. Two empty
values match when an extracted row exists. Blank corrected references therefore affect
scores and should be interpreted alongside the source-quality notes, not as independent
proof that the system found all information. Normalization and fuzzy comparison rules are
specified in the README; they do not modify the workbook.

The corrected references include judgement calls about unreadable content, notice semantics
and inclusive end dates. They are a reviewed reference variant, not an independently validated
benchmark. There is no weight training, but the dataset and labels were inspected during
development, and output settings were adjusted after a test response exceeded its budget.
Test results are therefore not a blind held-out estimate of generalization. A larger,
separately reviewed and held-out dataset is needed for that assessment.
