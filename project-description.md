# Project Description

Northstar runs professional learning programmes through a web platform. Support
colleagues answer questions about billing, account access, programme policies,
technical incidents, and privacy. Information is distributed across internal
articles, and the team needs a retrieval-augmented assistant that answers from
approved material rather than improvising.

You receive a small knowledge base under
[data/knowledge_base/](data/knowledge_base/) and question sets under
[data/evaluation/](data/evaluation/). Build a
local-first RAG system and an evaluation report that gives reviewers enough
evidence to decide whether this assistant should move beyond a prototype.

## Main Task

Build and evaluate a local RAG application that answers from approved sources,
preserves citations, abstains when evidence is insufficient or inapplicable,
and treats retrieved instructions as untrusted data.

The separate [reference solution](solution/README.md) demonstrates one
complete implementation. It is not the only valid design.

## Input Data

The knowledge base is fictional and intentionally compact so that failures are
easy to inspect. Each Markdown document is an approved support source. The
document about retrieved-content safety contains an example of a malicious
instruction; your application must treat that text as data about a threat, not
as an instruction to follow.

The JSON question sets provide:

- a question identifier and user query,
- expected source document IDs (empty for unsupported questions),
- key facts expected in a supported answer,
- whether the system should abstain because the answer is absent.

Use [calibration.json](data/evaluation/calibration.json) to choose retrieval and
abstention settings. It includes nearby but unsupported requests, not only
obviously unrelated topics. Freeze your choices before running the final
[questions.json](data/evaluation/questions.json) evaluation. Both sets are small
teaching fixtures, not evidence of general production reliability.

Do not edit the labels merely to improve metrics. When you disagree with a
label, document the disagreement in your report.

## Required Implementation

### 1. Ingestion and Chunking

Load every Markdown source and split long documents into retrieval chunks.
Each stored chunk must retain at least its `document_id`, title, and chunk
position so a generated answer can cite evidence.

Explain:

- your chunk size and overlap choice,
- how you handle headings or paragraph boundaries,
- what information is kept as retrieval metadata.

### 2. Embeddings and Vector Storage

Create an embedding for each chunk and index it in a retrieval backend. A local
[Qdrant](https://qdrant.tech/documentation/quickstart/) vector-database
collection or [FAISS](https://faiss.ai/) in-process vector index is sufficient.
If you choose another backend, identify whether it is an index or database and
justify its persistence and operational trade-offs.

Record:

- embedding model name and version,
- vector-store/index configuration,
- how the index can be rebuilt from the source documents.

### 3. Retrieval and Thresholding

Implement top-k retrieval. Your system must not answer every query merely
because some vector is closest: define an evidence threshold or an equivalent
abstention rule, and test it using the unsupported evaluation questions.
Use the supplied calibration set when selecting that rule; repeated tuning on
the final evaluation labels makes the final scores optimistic. A high
similarity score alone does not establish that a policy applies to the user's
request.

### 4. Grounded Generation and Guardrails

Generate answers from retrieved context only. Answers must cite source IDs in
square brackets, such as `[billing-refunds]`, and the application must reject
citations that were not in the retrieved evidence.

Your pipeline must also avoid following instructions embedded in retrieved
documents. See the
[OWASP Prompt Injection Prevention](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html) guidance
for the threat model.

Provider-backed generation is optional for local validation. If you add it,
keep secrets in `.env`, document the provider and model, and retain a
deterministic no-key execution path.

### 5. Evaluation Report

Evaluate all cases in `questions.json` and submit
`reports/evaluation_report.md`. Include:

- retrieval configuration and generation mode,
- `recall@k`, mean reciprocal rank (MRR), answer relevance, faithfulness, and
  decision accuracy (answering supported questions and abstaining on unsupported ones),
- a per-question results table,
- analysis of at least two failures or fragile successes,
- the chosen retrieval threshold and the evidence used to select it,
- three prioritized improvement recommendations.

Answer relevance and faithfulness may be implemented through a clearly
documented deterministic rubric or an evaluated model-based judge. A metric
without its rubric is not interpretable.
Citation correctness alone does not establish
[faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/). The reference
checks literal support for extractive answers and leaves live paraphrases
unrated until independently reviewed.

## Suggested Repository Layout

```text
your-rag-project/
├── data/
│   ├── knowledge_base/
│   └── evaluation/
├── reports/
│   └── evaluation_report.md
├── src/
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── generation.py
│   └── evaluation.py
├── tests/
├── .env.example
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

## Milestones and Acceptance Checklist

### Milestone 1 - Corpus and Index

- [ ] Load the unchanged source documents and create a real vector index.
- [ ] Show a chunk-count summary and a sample chunk with its source metadata.
- [ ] Provide a working command that rebuilds the index from those sources.

### Milestone 2 - Cited Question Answering

- [ ] Show a supported cited answer and an unsupported question that abstains.
- [ ] Handle instructions embedded in retrieved content explicitly.
- [ ] Test chunk metadata, retrieval, and rejection of fabricated citations.

### Milestone 3 - Evaluation and Recommendation

- [ ] Document settings selected with the calibration set.
- [ ] Run all final evaluation cases through the same end-to-end pipeline.
- [ ] Submit the report specified above, connecting failures and recommendations
  to retrieved evidence.
- [ ] Pass tests locally and document setup and commands reproducible from a
  fresh environment.

## Submission Questions

Answer the following questions in your README or evaluation report:

1. Which retrieval failures matter most to the support scenario, and why?
2. How did you select your no-answer threshold?
3. What evidence suggests an answer is faithful to retrieved context?
4. What would you change before allowing live support colleagues to use the
   system?
