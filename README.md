# Adaptive Stage-Aware Spatio-Temporal Reliability Estimation for Long-Tailed Noisy Label Learning

## Overview

This repository contains the implementation and experimental development for the undergraduate research project:

**“An Adaptive Stage-Aware Spatio-Temporal Reliability Estimation Algorithm for Long-Tailed Noisy Label Learning.”**

The project investigates sample reliability when **class imbalance (long-tailed data)** and **label noise** occur together. A key challenge is that difficult but correctly labelled tail-class samples can resemble mislabeled samples when reliability is judged using only loss or current prediction confidence.

The proposed approach studies three complementary sample-level reliability signals:

1. **Prediction Confidence (C)** — current output-space evidence.
2. **Temporal Prediction Consistency (T)** — stability of a sample's predictions across training.
3. **Neighbourhood Homogeneity (N)** — local support from nearby samples in learned feature space.

The long-term goal is to combine these signals into a reliability estimator and investigate whether their relative importance should change according to the training stage.

---

## Research Motivation

Under Long-Tailed Noisy Label Learning (LTNLL), a low-confidence or high-loss sample is not necessarily noisy. A correctly labelled sample from a minority class may also be difficult because the model has seen much less evidence for that class.

This research therefore asks whether confidence, temporal behaviour, and local feature-space neighbourhood information can provide complementary reliability evidence, particularly for preserving **hard clean tail samples**.

---

## Current Experimental Setup

The current development notebook uses a controlled CIFAR-10 benchmark:

- **Dataset:** CIFAR-10
- **Backbone:** ResNet-18
- **Number of classes:** 10
- **Long-tail imbalance ratio:** 10:1
- **Long-tailed training samples:** 20,431
- **Synthetic label noise:** 20% symmetric noise
- **Optimizer:** SGD
- **Loss:** Cross-Entropy
- **Environment:** Google Colab with GPU

The clean CIFAR-10 labels are retained only for evaluation so that reliability scores can be compared against the true clean/noisy status of each training sample.

---

## Current Progress

### Completed

- [x] Constructed a controlled long-tailed CIFAR-10 training set.
- [x] Injected 20% symmetric label noise.
- [x] Defined head, medium, and tail class groups.
- [x] Built and trained a ResNet-18 baseline.
- [x] Implemented **Prediction Confidence (C)** as the probability assigned to the observed training label.
- [x] Evaluated confidence for clean/noisy samples and across head/medium/tail groups.
- [x] Recorded per-sample prediction histories across a 20-epoch development run.
- [x] Implemented an initial **Temporal Consistency (T)** score using agreement between consecutive predicted classes.
- [x] Compared confidence and temporal-consistency signal quality across training epochs.

### Current Finding

The initial confidence analysis indicates that prediction confidence is useful overall but becomes substantially less reliable toward tail classes.

The first simple temporal-consistency formulation, based only on whether the predicted class remains unchanged between consecutive epochs, showed weak clean/noisy discrimination. This is treated as a development finding rather than a final conclusion, and motivates testing a richer probability-distribution-based temporal measure.

### Next Steps

- [ ] Develop a stronger probability-based temporal-consistency formulation.
- [ ] Implement **Neighbourhood Homogeneity (N)** using learned feature representations and local nearest neighbours.
- [ ] Evaluate C, T, and N individually.
- [ ] Test pairwise signal combinations.
- [ ] Implement fixed-weight C + T + N fusion.
- [ ] Develop and evaluate stage-aware/adaptive weighting.
- [ ] Measure hard-clean-tail retention and clean/noisy discrimination.
- [ ] Run final experiments using multiple random seeds and stronger benchmark settings.

---

## Planned Reliability Formulation

The intended sample-level reliability score is conceptually:

\[
R_i(t) = w_C(t) C_i(t) + w_T(t) T_i(t) + w_N(t) N_i(t)
\]

where:

- \(C_i(t)\) = prediction confidence,
- \(T_i(t)\) = temporal prediction consistency,
- \(N_i(t)\) = neighbourhood homogeneity,
- \(w_C(t), w_T(t), w_N(t)\) = signal weights that may vary with training stage.

The study will compare this stage-aware formulation against individual signals and fixed-weight fusion.

---

## Repository Structure

```text
.
├── README.md
├── notebooks/
│   └── LTNLL_Baseline.ipynb
├── results/
│   └── README.md            # later: small tables/figures
└── .gitignore
```

Large datasets, model checkpoints, and generated experiment artifacts should not be committed directly to the repository.

---

## Reproducibility Notes

The notebook currently uses a fixed random seed for development experiments. Final reported experiments should be repeated across multiple random seeds and summarized using mean and standard deviation.

The repository is under active development, so methods and experimental settings may change as the research progresses.

---

## References

The research is informed by prior work on noisy-label learning, long-tailed noisy-label learning, temporal learning dynamics, hard-clean sample identification, neighbourhood consistency, and dynamic weighting. Full academic references will be added as the implementation and thesis are finalized.

---

## Status

**Research in progress.**  
This repository currently contains development-stage experiments and should not be interpreted as a finalized implementation or final research result.
