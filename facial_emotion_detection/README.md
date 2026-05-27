# Facial Emotion Detection

A CNN-based facial emotion recognition (FER) system, framed around depression-screening use cases.

> **Pre-MIDS coursework** — capstone for MIT Professional Education's *Applied Data Science Program* delivered by Great Learning, December 2023.

## Problem

Detect human emotion from facial images. Motivated by mental-health screening: FER can help surface candidates for interventional treatment, with the goal of reducing the downstream economic and human cost of untreated depression.

## Approach

Trained and compared six models on the course-provided facial emotion image dataset:

- **Transfer-learning baselines**: VGG16, ResNet, EfficientNet
- **Custom CNN variants**: model3, model4, model5 (progressively tuned architecture and regularization)

## Result

Custom CNN architecture (`model5`) outperformed the transfer-learning baselines across all metrics:

| Metric | VGG16 | ResNet | EfficientNet | model3 | model4 | **model5** |
|---|---|---|---|---|---|---|
| Loss | 80.59 | 85.23 | 81.23 | 56.07 | 52.32 | **48.52** |
| Accuracy | 69.53 | 63.28 | 65.62 | 78.13 | 81.25 | **84.38** |
| Precision | 77.06 | 73.68 | 70.87 | 84.69 | 83.47 | **85.47** |
| Recall | 65.62 | 54.69 | 57.03 | 73.44 | 78.91 | **78.13** |
| AUC-ROC | 91.53 | 87.13 | 88.73 | 94.81 | 95.58 | **96.40** |

Same data is in [model_accuracy_comparison.csv](model_accuracy_comparison.csv).

## Files

- [facial_emotion_detection.ipynb](facial_emotion_detection.ipynb) — full training and evaluation notebook
- [SSharma-CAPSTONE-Presentation3.0.pdf](SSharma-CAPSTONE-Presentation3.0.pdf) — final capstone presentation
- [PerformanceMetricsTable.png](PerformanceMetricsTable.png) — rendered version of the results table
- [model_accuracy_comparison.csv](model_accuracy_comparison.csv) — same metrics in CSV form

## Dataset

The facial emotion image dataset was provided by the course (downloaded via Great Learning's Olympus LMS) and is not redistributed here.

## Acknowledgment

The notebook scaffolding (problem statement, dataset-loading instructions, and section headings) is provided by the course. The modeling, training pipeline, hyperparameter choices, comparative analysis, and final write-up are mine.
