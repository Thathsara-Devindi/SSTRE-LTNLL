# SSTRE-LTNLL

**An Adaptive Stage-Aware Spatio-Temporal Reliability Estimation Algorithm for Long-Tailed Noisy Label Learning**

SSTRE-LTNLL asks whether changing the mix of three reliability signals during training helps a classifier learn from long-tailed data with noisy labels, particularly when difficult but correctly labelled tail examples resemble mislabelled examples. The study compares a fixed mixture with a prespecified stage schedule under matched training settings. The contribution is the implemented integration and controlled comparison of these signals, including a frozen subgroup audit of the proposed benefit.

## Final experimental protocol

| Dataset | Observed training labels | Frozen training images | Clean validation images | Imbalance factor |
|---|---|---:|---:|---:|
| CIFAR-10N | `random_label1` | 18,388 | 5,000 | 10 |
| CIFAR-100N | `noisy_label` (fine classes) | 17,614 | 5,000 | 10 |

The study uses human noisy annotations in the original CIFAR training order. Source labels support controlled long-tail construction, clean validation and offline audits. Training receives frozen sample IDs and observed noisy labels. Dataset split seed 42 stays fixed when the training seed changes. The CIFAR-10N trainer consumes the original frozen split, and the CIFAR-100N constructor uses PCG64 with seed 42. See [data layout and reproduction](docs/REPRODUCTION.md).

The model is ResNet-18 initialized without pretrained weights. It uses a 3×3 stride-1 stem, an identity max pool and a dataset-sized classifier. Training uses a padded random crop, horizontal flip and normalization. Evaluation uses tensor conversion and the same fixed channel normalization. The locked configuration records the exact values, including the shared normalization used for CIFAR-100N.

### Reliability and training loss

At loss epoch *e*, the score uses completed evidence from that run:

- **C:** probability of the observed training label at epoch *e−1*.
- **T:** fraction of the five predictions from epochs *e−6* through *e−2* whose argmax agrees with that label.
- **N:** mean model probability assigned to the query sample's observed label across its ten nearest learned-feature neighbors. Cosine similarity uses normalized penultimate features and excludes the query itself. The neighborhood bank refreshes after epochs 30, 35, …, 95.

| Loss epochs | Stage-Aware coefficients (C, T, N) | Fixed Fusion |
|---|---|---|
| 1–30 | Unit-weight CE warm-up | Unit-weight CE warm-up |
| 31–50 | (2/5, 1/5, 2/5) | (23/70, 24/70, 23/70) |
| 51–75 | (1/3, 1/3, 1/3) | (23/70, 24/70, 23/70) |
| 76–100 | (4/15, 7/15, 4/15) | (23/70, 24/70, 23/70) |

The schedule is specified in advance. After warm-up, the weighted sum of C, T and N gives reliability `R`. The implementation computes `q = 0.25 + 0.75R` and divides each `q` by its mean over the full frozen training set. The loss is the batch mean of per-example cross-entropy multiplied by these detached positive weights. Every training example keeps a positive weight.

Training runs for 100 epochs with SGD (initial LR 0.1, momentum 0.9, weight decay 0.0005), batch size 128 and LR drops by 0.1 after epochs 60 and 85. Checkpoint selection uses the first strict maximum of clean validation macro-F1. See the [locked training configuration](experiments/main100/reviewed_protocol/protocol.json).

## Verified results

The seven original main runs use training seed 42. They include CE, Fixed and Stage-Aware on both datasets plus C-only on CIFAR-10N. The post-hoc extension adds Fixed/Stage-Aware pairs on both datasets at training seeds 43 and 44, and five CIFAR-10N seed-42 signal ablations (T-only, N-only, C+T, C+N and T+N). The original seven held-out test evaluations used validation-selected checkpoints frozen before test inference. The extensions report validation classification and training-set reliability.

| Dataset | Fixed Fusion official-test accuracy | Stage-Aware official-test accuracy |
|---|---:|---:|
| CIFAR-10N | 83.83% | 84.80% |
| CIFAR-100N | 45.79% | 45.25% |

Across training seeds 42, 43 and 44, the mean paired Stage-Aware minus Fixed validation accuracy difference was −0.21 percentage points on CIFAR-10N and −0.33 on CIFAR-100N. The paired results did not demonstrate consistent overall superiority. [Official test results](results/ORIGINAL_SEVEN_OFFICIAL_TEST_RESULTS.csv) · [Paired validation results](results/THREE_SEED_VALIDATION_RAW.csv) · [Validation statistics](results/THREE_SEED_VALIDATION_STATISTICS.csv)

The frozen reliability audit ranks each run's loss-epoch-100 scores, retains the top 70% hypothetically and breaks ties by ascending frozen local sample ID. The difficult clean tail groups are frozen low-confidence proxies, defined with audit-only reference labels. This evaluation counts samples below a cutoff. The training procedure discards zero samples.

| Dataset | Training seed | Fixed below cutoff | Stage-Aware below cutoff |
|---|---:|---:|---:|
| CIFAR-10N | 42 | 39/392 | 44/392 |
| CIFAR-10N | 43 | 33/392 | 42/392 |
| CIFAR-10N | 44 | 49/392 | 50/392 |
| CIFAR-100N | 42 | 140/315 | 145/315 |
| CIFAR-100N | 43 | 121/315 | 122/315 |
| CIFAR-100N | 44 | 126/315 | 119/315 |

Stage-Aware rejected more hard clean tail samples in every CIFAR-10N seed. CIFAR-100N was mixed. The expected consistent improvement in hard-clean-tail retention was unsupported. Overall clean/noisy ranking, subgroup rejection and classification accuracy measure different outcomes. [Paired reliability results](results/THREE_SEED_FROZEN_RELIABILITY_RAW.csv) · [CIFAR-10N signal ablations](results/CIFAR10N_NINE_METHOD_VALIDATION_AND_LOSS100.csv)

## Source and reproduction

- [Main100 implementation](experiments/main100/) contains the original seed-42 model, data, training and checkpoint modules and the seven reviewed configurations.
- [Post-hoc implementation](experiments/posthoc100/) contains the verified additional seeds and signal ablations.
- [Dataset preparation](dataset_preparation/) contains the CIFAR-100N freeze and validation driver and verified source hashes.
- [Evaluation](evaluation/) contains the reliability audit and the original official-test evaluator modules.
- [Provenance](docs/PROVENANCE.md) records the exact source manifests and every publication packaging change.

From the repository root, run `python tools/check_release.py` with Python 3.11 or later to verify the packaged file hashes and Python syntax without installing dependencies. [Reproduction guide](docs/REPRODUCTION.md) lists the verified runtime dependencies, data layout, inputs and storage requirements. The code and aggregate results are available here. Exact replay requires separately obtained frozen inputs and saved evidence.

## Limitations and historical notebooks

The evaluation uses one frozen split per dataset, three matched training seeds for Fixed versus Stage-Aware, and single-seed signal ablations on CIFAR-10N. The hard-clean-tail masks are exploratory proxies. The post-hoc analyses use validation and training-split evidence. There is no matched published-method reproduction or claim of superiority over published LTNLL methods. The results do not identify a unique cause for the observed ranking or classification differences.

[LTNLL_Baseline.ipynb](LTNLL_Baseline.ipynb) and [LTNLL_Research.ipynb](LTNLL_Research.ipynb) are historical Colab development notebooks. Their early synthetic-noise and short-budget experiments are separate from the final protocol above.

The research code uses the existing [MIT license](LICENSE). Obtain CIFAR images and CIFAR-N annotations from their upstream releases and follow their terms.
