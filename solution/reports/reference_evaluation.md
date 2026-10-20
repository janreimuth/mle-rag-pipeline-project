# Reference Evaluation Report

## Configuration

- Question set: `questions.json`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Vector store: `Qdrant` local mode with cosine similarity
- Retrieval: `top_k=3`, `min_score=0.30`
- Ranking metrics use safety-filtered chunks before the evidence threshold.
- Generation: `deterministic extractive dry mode`

## Metric Rubric

- `recall@k`: mean fraction of all labeled relevant documents retrieved per supported question.
- `HitRate@k`: fraction of supported questions with at least one relevant retrieved document.
- `MRR@k`: reciprocal rank of the first relevant chunk, averaged over supported questions; repeated sources keep their chunk positions.
- `answer relevance`: literal required-term coverage on supported questions, not a semantic relevance judge; a false abstention receives 0. Unsupported questions are scored by decision accuracy instead.
- `citation correctness`: supported questions are answered with citations to expected sources in accepted evidence; a false abstention is incorrect. Unsupported questions are scored by decision accuracy instead.
- `faithfulness`: literal support in cited evidence for answered cases only, not semantic entailment. Live paraphrases need manual review and remain unrated; abstentions are excluded.
- `decision accuracy`: answers supported questions and abstains on unsupported questions, averaged over all questions. Only `answered` and `abstained` count as decisions; `blocked_input` and `invalid_citations` are scored as incorrect for every question.
- Ranking metrics, answer relevance, and citation correctness are not rated when no supported cases exist. Faithfulness is not aggregated when there are no answers or any answer needs manual review.

## Results

| Metric | Score |
| --- | ---: |
| Retrieval recall@3 | 1.000 |
| Retrieval HitRate@3 | 1.000 |
| Mean reciprocal rank@3 | 0.962 |
| Answer relevance | 1.000 |
| Citation correctness | 1.000 |
| Faithfulness (literal support only) | 1.000 |
| Decision accuracy | 1.000 |

## Per-Question Evidence

| ID | Expected source | Retrieved chunk source order | Status | Relevance | Citations correct | Literal support | Safety event |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| q01 | billing-refunds | invoice-corrections, billing-refunds, billing-refunds | answered | 1.00 | yes | yes | - |
| q02 | billing-refunds | billing-refunds, billing-refunds, billing-refunds | answered | 1.00 | yes | yes | - |
| q03 | invoice-corrections | invoice-corrections, invoice-corrections, billing-refunds | answered | 1.00 | yes | yes | - |
| q04 | account-mfa | account-mfa, account-mfa, account-password | answered | 1.00 | yes | yes | - |
| q05 | account-password | account-password, account-password, account-mfa | answered | 1.00 | yes | yes | - |
| q06 | programme-extensions | programme-extensions, programme-extensions, programme-attendance | answered | 1.00 | yes | yes | - |
| q07 | technical-api-limits | technical-api-limits, technical-incidents, retrieved-content-safety | answered | 1.00 | yes | yes | - |
| q08 | technical-incidents | technical-incidents, technical-incidents, technical-api-limits | answered | 1.00 | yes | yes | - |
| q09 | privacy-data, technical-incidents | technical-incidents, privacy-data, account-mfa | answered | 1.00 | yes | yes | - |
| q10 | retrieved-content-safety | retrieved-content-safety, retrieved-content-safety, privacy-data | answered | 1.00 | yes | yes | filtered_retrieved_instruction:retrieved-content-safety-1 |
| q11 | - | programme-attendance, programme-attendance, technical-incidents | abstained (no_accepted_evidence) | n/a | n/a | n/a | - |
| q12 | - | technical-api-limits, invoice-corrections, billing-refunds | abstained (no_accepted_evidence) | n/a | n/a | n/a | - |
| q13 | account-mfa, privacy-data | account-mfa, account-mfa, privacy-data | answered | 1.00 | yes | yes | - |
| q14 | programme-extensions | programme-extensions, programme-extensions, technical-api-limits | answered | 1.00 | yes | yes | - |
| q15 | programme-attendance | programme-attendance, programme-attendance, privacy-data | answered | 1.00 | yes | yes | - |
| q16 | - | programme-attendance, privacy-data, programme-attendance | abstained (no_accepted_evidence) | n/a | n/a | n/a | - |

## Failure And Fragility Analysis

- No labeled case fails the deterministic rubric. This is a small, curated benchmark and does not prove production reliability.
- Fragile success candidate `q09` has top score `0.347`; paraphrase expansion should test whether the selected threshold remains defensible.
- Fragile success candidate `q06` has top score `0.518`; paraphrase expansion should test whether the selected threshold remains defensible.

## Threshold Decision

The reference pipeline uses `min_score=0.30`. The weakest supported top hit is `q09` at `0.347`, while the strongest unsupported top hit is `q12` at `0.225`. The current threshold separates those observed scores. These are diagnostics for this question set, not evidence used to select settings. Calibrate the threshold together with the generator's abstention rule on `calibration.json`, then freeze both before the final `questions.json` evaluation. See `solution/README.md` for the default reference calibration rationale.

## Prioritized Recommendations

1. Expand the benchmark with paraphrases and multi-source questions before changing the threshold.
2. Compare the embedding model and chunking configuration using the same labeled cases.
3. Run a separately reviewed live-generation evaluation and inspect citation failures before operational use.
