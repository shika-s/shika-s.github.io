# Shikha Sharma — Machine Learning & AI Portfolio

shikhasharma@berkeley.edu

Master's in Information and Data Science, University of California, Berkeley (completed May 2026)

## About

Machine learning and AI professional building scalable data pipelines, predictive models, and LLM applications, with 10+ years of prior software engineering leadership across healthcare, e-commerce, and semiconductor systems.

## Projects

### ML Systems Engineering

- **[End-to-End ML API](end_to_end_machine_learning_api/)** — DistilBERT sentiment API with FastAPI, Docker, Kubernetes (HPA, probes, caching), k6 load testing.
- **[Caching & Kubernetes Lab](caching-and-kubernetes/)** — FastAPI + Redis service deployed on Kubernetes with caching and tests.

### Hackathon

- **[Four Sigma Quantitative Finance Hackathon](four_sigma_hackathon/)** — Team submission predicting 30-minute stock returns. I built the feature engineering pipeline (rolling 30-min stats, cross-sectional z-scores, 5-min bar features, sector-momentum signals from spectral-clustering pseudo-sectors) and the final LightGBM / RF / XGBoost ensemble, which scored +0.0713 mean Spearman IC.

### NLP and GenAI
- **[Multi-Persona RAG](multi_persona_rag/)** — RAG system serving engineering vs marketing audiences; compared retrievers, embeddings, prompts, and LLMs (Mistral, Cohere) across 78 questions.
- **[Cybersecurity Policy Generation](cybersecurity_policy_generation/)** — Ontology-grounded policy generation using LLMs, Neo4j, and DPO/LoRA fine-tuning. Maps regulatory frameworks (NIST CSF, SOC 2, ISO 27001) into a knowledge graph for retrieval-augmented policy drafting.

### Machine Learning
- **[Flight Delay Prediction](flight_delay_prediction/)** — Time-series modeling on 31M flights (2015–2019) using DOT and NOAA datasets, with explicit data-leakage controls.
- **[WFP Food Security Forecasting](capstone/)** — Capstone case study. Multi-source forecasting pipeline (FEWS NET, World Bank, HDX, ACLED, IPC) for food security across four Sub-Saharan countries, deployed monthly on AWS (EventBridge → ECS Fargate → S3 → Lambda → App Runner). Architecture and IaC published; pipeline source under client license.
- **[Bird Sound Classification](bird_sound_classification/)** — Deep learning for Kaggle bird-sound recognition.
- **[Facial Emotion Detection](facial_emotion_detection/)** — CNN for facial emotion recognition, framed around depression screening. Custom architecture reached 84.4% accuracy / 0.964 AUC, outperforming VGG16, ResNet, and EfficientNet transfer-learning baselines. *(Pre-MIDS — MIT Professional Education / Great Learning, 2023.)*

### Data Engineering and Analytics

- **[Neo4j Financial Portfolio](neo4j_financial_portfolio/)** — Graph-based portfolio analysis using community detection on a Neo4j time-series graph database.
- **[Housing Price Analysis](housing_price_analysis/)** — Linear regression study following the cookiecutter-data-science layout.


## Coursework

- Capstone
- Machine Learning Systems Engineering
- Natural Language Processing with Deep Learning
- Fundamentals of Generative Artificial Intelligence (AI)
- Machine Learning at Scale
- Applied Machine Learning
- Fundamentals of Data Engineering
- Statistics for Data Science
- Research Design and Applications for Data and Analysis
- Introduction to Data Science Programming
