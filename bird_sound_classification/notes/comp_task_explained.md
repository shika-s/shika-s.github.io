Here’s a concise walkthrough of BirdCLEF+ 2025 and what it entails:

## What’s the task?

We’ll build machine-learning models that listen to five-second audio clips recorded at El Silencio Natural Reserve in Colombia’s Magdalena Valley and predict which species are present. Although named “BirdCLEF,” this edition also covers amphibians, mammals, insects and reptiles.

## Why it matters

* **Biodiversity monitoring:** In regions like the lowland Magdalena Valley—where over 70 % of the original rainforest has been converted to pasture—tracking changes in animal communities is crucial for guiding restoration and conservation.
* **Scalability:** Traditional field surveys are labor-intensive and costly. Passive acoustic monitoring (PAM) allows continuous, wide-area sampling.
* **Conservation impact:** Better automated detection helps Fundación Biodiversa Colombia and partners evaluate restoration success and respond more rapidly to emerging threats.

## Core goals

1. **Species identification:** Detect under-studied or rare taxa from audio across multiple groups.
2. **Data-efficient training:** Achieve strong results even for species with very few labeled examples.
3. **Semi-supervision:** Incorporate unlabeled recordings to boost performance on scarce classes.

## Timeline (all deadlines 11:59 PM UTC)

* **March 10, 2025:** Competition opens
* **May 29, 2025:**

  * Last day to accept rules and join/merge teams
* **June 5, 2025:** Final model submission

## Evaluation metric

A **macro-averaged ROC-AUC** over all species, omitting any classes with no positive labels in the test set. This treats each species equally and rewards models that do well on rare classes.

## Submission format

* Your output must be a CSV named `submission.csv`.
* **Rows:** each five-second window (`row_id`)
* **Columns:** one per species, containing the predicted probability of presence.

## Rewards & recognition

* **Prizes:** \$15 000 for 1st place, \$10 000 for 2nd, \$8 000 for 3rd, \$7 000 for 4th, \$5 000 for 5th.
* **Best Working Note Award:** Submit a short technical report to the CLEF 2025 conference; top two receive \$2 500 each. Reports are judged on originality, quality, contribution and clarity.

## Code competition rules

* **Notebook–based submissions only.**
* **Runtime limits:**

  * CPU: ≤ 90 min
  * GPU: disabled (you can submit but will only get a 1 min runtime—best to stick with CPU)
* **No internet access** during execution.
* **External data & pre-trained models** are allowed if publicly and freely available.
* After committing to your Kaggle Notebook, click **Submit** to produce `submission.csv`.

---

By combining careful feature engineering (e.g., spectrogram transforms), data-augmentation strategies, and techniques for handling class imbalance or few-shot learning, we’ll help conservationists monitor species trends more effectively across this critical tropical ecosystem. Good luck!
