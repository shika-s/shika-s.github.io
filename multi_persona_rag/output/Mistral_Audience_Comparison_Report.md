#  Mistral RAG Performance - Research vs Marketing Audience

## Overview

| Metric | Value |
|--------|-------|
| LLM | mistralai/Mistral-7B-Instruct-v0.3 |
| Total Questions | 78 |
| Chunk Size | 500 |
| Embedding | multi-qa-mpnet-base-dot-v1 |
| Research Config | research_prompt_48 |
| Marketing Config | marketing_prompt_q33 |

---

##  Overall Performance Summary

| Metric | Research | Marketing | Winner |
|--------|----------|-----------|--------|
| **MDS Mean** | **0.5606** | 0.5569 | Research |
| Faithfulness | **0.7849** | 0.6489 | Research |
| Answer Relevancy | 0.6742 | **0.7410** | Marketing |
| Context Precision | **0.8710** | 0.8112 | Research |
| Context Recall | 0.7741 | **0.8433** | Marketing |
| Zero MDS Rows | 22 | **19** | Marketing |

---

##  Win Count Per Question

| Audience | Questions Won |
|----------|---------------|
| **Research** | **40** |
| Marketing | 34 |
| Tie | 4 |

---

##  Performance Distribution

### Research Audience

| Category | Count | Percentage |
|----------|-------|------------|
| Excellent (≥0.85) | 16 | 20.5% |
|  Good (0.70-0.85) | 29 | 37.2% |
|  Fair (0.50-0.70) | 7 | 9.0% |
|  Poor (0.001-0.50) | 1 | 1.3% |
|  Zero (<0.001) | 22 | 28.2% |

### Marketing Audience

| Category | Count | Percentage |
|----------|-------|------------|
|  Excellent (≥0.85) | 13 | 16.7% |
|  Good (0.70-0.85) | 26 | 33.3% |
|  Fair (0.50-0.70) | 16 | 20.5% |
|  Poor (0.001-0.50) | 2 | 2.6% |
|  Zero (<0.001) | 19 | 24.4% |

---

##  Zero MDS Analysis

### Summary

| Audience | Total Zero | Zero Faithfulness | Zero Answer Rel. |
|----------|------------|-------------------|------------------|
| Research | 22 | 1 | 21 |
| Marketing | 19 | 3 | 16 |

### Questions Failing for BOTH Audiences

[2, 8, 9, 12, 13, 19, 24, 51, 73]

**Count: 9 questions** - these need investigation.

---

##  Key Insights

### 1. Research Prompt Strengths
- Higher Faithfulness (0.78 vs 0.65)
- Better Context Precision (0.87 vs 0.81)

### 2. Marketing Prompt Strengths
- Higher Answer Relevancy (0.74 vs 0.67)
- Better Context Recall (0.84 vs 0.77)
- Fewer zero MDS failures (19 vs 22)

### 3. Common Weakness
- Zero Answer Relevancy is the primary cause of failures for both audiences

---

##  Recommendations

1. **Use audience-appropriate prompts** based on priorities
2. **Investigate 9 problem questions** that fail for both audiences
3. **Address Answer Relevancy issues** in response format

---

*Report generated automatically*
