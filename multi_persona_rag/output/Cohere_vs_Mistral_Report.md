# 📊 Cohere vs Mistral RAG Performance Comparison

## Overview

| Metric | Value |
|--------|-------|
| LLMs Tested | Cohere, Mistral-7B-Instruct-v0.3 |
| Questions | 78 |

---

##  Overall Results

| Config | LLM | MDS Mean | Faithfulness | Answer Rel. | Zero MDS |
|--------|-----|----------|--------------|-------------|----------|
| Research | **Mistral** | **0.6471** | 0.8127 | **0.7637** | **14** |
| Research | Cohere | 0.5260 | **0.8344** | 0.6388 | 26 |
| Marketing | Mistral | 0.5576 | **0.6333** | 0.7529 | 19 |
| Marketing | Cohere | 0.5554 | 0.6135 | **0.7532** | **18** |

###  Winner: Mistral

---

##  Research Audience: Detailed Comparison

| Metric | Mistral | Cohere | Change | Winner |
|--------|---------|--------|--------|--------|
| MDS Mean | **0.6471** | 0.5260 | -0.1211 | Mistral |
| Faithfulness | 0.8127 | **0.8344** | +0.0218 | Cohere |
| Answer Relevancy | **0.7637** | 0.6388 | -0.1250 | Mistral |
| Zero MDS Count | **14** | 26 | +12 | Mistral |

---

##  Key Findings

### 1. Mistral Wins Overall
- Higher MDS for both audiences
- Better Answer Relevancy scores
- Fewer zero MDS failures

### 2. Cohere's Strength: Faithfulness
- Slightly higher faithfulness
- Responses are more grounded in context

### 3. Cohere's Weakness: Answer Relevancy
- Significantly lower Answer Relevancy
- Causes more zero MDS rows

---

##  Recommendations

| Use Case | Recommended LLM |
|----------|-----------------|
| **General RAG** | **Mistral** |
| **High-Faithfulness needed** | Cohere |

---

*Report generated automatically*
