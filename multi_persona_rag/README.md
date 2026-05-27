# Multi-Persona RAG

A Retrieval-Augmented Generation (RAG) system tested across two audience personas — **engineering/research** and **marketing** — to study how the same retrieval pipeline serves users with different information needs.

> **Course project** — UC Berkeley MIDS DATASCI 267 (Generative AI), Assignment 5. The "company" in the scenario is fictional. The underlying corpus is publicly sourced material on LLMs and GenAI.

## Scenario

A four-week proof-of-concept evaluating whether RAG can simultaneously serve:

- **Engineering**: technical Q&A about GenAI concepts and implementation details.
- **Marketing**: accurate, approved messaging for content production.

The same retriever + LLM stack is asked the same questions but prompted to answer for each audience, allowing direct comparison of how prompting and retrieval interact with audience-specific information needs.

## Result

Testing across **78 questions** spanning both audiences:

| Audience | Best LLM | MDS Mean | Answer Relevancy |
|---|---|---|---|
| Research | Mistral-7B-Instruct-v0.3 | **0.65** | **0.76** |
| Marketing | Mistral-7B-Instruct-v0.3 | 0.56 | 0.75 |

Mistral edged Cohere overall, while Cohere produced slightly more grounded (faithful) responses. Full numbers, ablations, and discussion are in [docs/Final_Project_Report_267.pdf](docs/Final_Project_Report_267.pdf) and [output/Cohere_vs_Mistral_Report.md](output/Cohere_vs_Mistral_Report.md).

## Repository layout

```
multi_persona_rag/
├── data/        # gold answers + test questions
├── docs/        # final project report (PDF)
├── notebooks/   # 10 experiment notebooks (run in numeric order)
└── output/      # CSVs, plots, and summary reports
```

## Notebook order

The notebooks were developed and should be read in numeric order. Each one builds on findings from the previous experiment.

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_scoring.ipynb` | Scoring methodology and metric definitions (MDS, faithfulness, answer relevancy). |
| 02 | `02_retriever_experiments.ipynb` | Initial retriever sweep — chunk sizes, overlaps, `k`. |
| 03 | `03_embedding_experiments.ipynb` | Embedding model comparison (e.g. `multi-qa-mpnet-base-dot-v1`, MiniLM, distilroberta). |
| 04 | `04_llm_experiments.ipynb` | LLM parameter sweep (temperature, top-p, repetition penalty). |
| 05 | `05_prompt_experiments.ipynb` | Prompt design and few-shot variations. |
| 06 | `06_visualizations.ipynb` | MDS plots, correlation matrices, embedding-space comparisons. |
| 07 | `07_retriever_revisited.ipynb` | Second-pass retriever tuning with findings from earlier rounds. |
| 08 | `08_full_run_mistral.ipynb` | End-to-end run on the full question set with Mistral. |
| 09 | `09_full_run_cohere.ipynb` | End-to-end run with Cohere for direct comparison. |
| 10 | `10_final.ipynb` | Final consolidated analysis and write-up. |

## Stack

- **Retrievers / orchestration**: LangChain (`langchain`, `langchain-community`, `langchain-cohere`)
- **Embeddings**: `sentence-transformers` (multi-qa-mpnet, MiniLM, distilroberta)
- **LLMs**: Mistral-7B-Instruct-v0.3, Cohere Command
- **Evaluation**: custom MDS metric + faithfulness / answer relevancy

## Running the notebooks

Notebooks were authored in Google Colab and read API keys from Colab's `userdata` secrets manager. To run locally, set the following environment variables:

```bash
export COHERE_API_KEY=...
export HF_TOKEN=...           # for Mistral via Hugging Face
```

The Colab `userdata.get(...)` calls also fall back to `os.environ.get(...)` where applicable.

## Disclaimer

This is coursework. The fictional company scenario is part of the assignment prompt — there is no real customer data. The corpus consists of public material about LLMs and Generative AI.
