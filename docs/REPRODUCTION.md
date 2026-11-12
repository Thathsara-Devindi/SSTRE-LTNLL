# Reproducing the experiments

## Runtime and checks

The recorded training environment used Python 3.11, PyTorch 2.9.1, TorchVision 0.24.1 and the CUDA 12.6 wheel index. The other recorded package pins and the Modal client version are in [requirements.txt](../requirements.txt). Select the PyTorch build appropriate to your environment. The packaged training entrypoints describe the original T4 allocation and create their recorded image in Modal.

Run the dependency-free integrity and syntax check from the repository root:

```text
python tools/check_release.py
```

The two experiment bundles expose the same Python package name, `sstre100`. Use one bundle's `src` directory per interpreter session. Each source bundle retains the relative layout used by its configuration loader.

## Frozen data

Obtain the CIFAR images and CIFAR-N annotations from their upstream providers. The [CIFAR-100N input digest record](../dataset_preparation/cifar100n_source_hashes.json) lists the exact official source files used to prepare that split. [CIFAR-10N configuration](../experiments/main100/reviewed_protocol/datasets/cifar10n.json) records the original frozen split and archive digests.

The trainers expect this structure under the input Volume root:

```text
LTNLL_Research/
  datasets/
    cifar10_official/cifar-10-python.tar.gz
    cifar100_official/cifar-100-python.tar.gz
  Phase1_V2/dataset_split_v2.npz
  Phase1_CIFAR100N_V1/
    train.npz
    validation.npz
    FREEZE_RECEIPT.json
```

The training loader checks these files and their identities. The archived CIFAR-10N split uses its exact original ordered IDs and annotations. The historical notebooks show early split development. Re-creating a similar split from those notebooks is insufficient unless it matches both the locked fingerprint and the expected split archive hash.

For CIFAR-100N, the [freeze driver](../dataset_preparation/freeze_and_validate.py) uses the original constructor with PCG64(42), per-class validation selection, long-tail counts and final deterministic permutations. Supply the two hash-matching upstream files in the directory named by `SSTRE_CIFAR100_SOURCE_DIR`. Set `SSTRE_CIFAR100_FREEZE_DIR` to a new destination and `SSTRE_CIFAR100_REPORT_DIR` to a separate, existing report directory. Run `python dataset_preparation/freeze_and_validate.py` only with that dataset and the recorded dependencies available. The preparation checks the official archive, annotation keys, image-row alignment and ordered split fingerprint. Keep the resulting audit-only labels and masks separate from training views.

The original CIFAR-100N planning contract contains null values for the as-yet-unprepared annotation and fingerprint fields. The final data identity is verified through the freeze receipt and the post-hoc fingerprint and file-hash checks. Historical preparation flags remain part of the reviewed protocol, while the retention and writer amendments specify the implemented checkpoint policy.

## Training

The main and extension launchers are adapted copies of the original Modal entrypoints. They require an authenticated account and explicitly selected resources:

| Environment variable | Meaning |
|---|---|
| `SSTRE_INPUT_VOLUME` | Your existing Volume with the frozen input layout above |
| `SSTRE_OUTPUT_VOLUME` | Your existing results Volume |
| `SSTRE_MODAL_ENVIRONMENT` | The environment containing both Volumes |
| `SSTRE_CLAIMS_DICT` | The claim dictionary name for the post-hoc launcher |

Provide these settings on the launch host and use the corresponding Modal CLI environment. Inputs mount read-only. The launcher passes the names to the remote image so its own imports resolve the same resources. These values are account resource names, and Modal credentials are managed through the normal local Modal authentication flow.

Use [main100_modal.py](../experiments/main100/main100_modal.py) with a dataset and method from the seven [run configurations](../experiments/main100/reviewed_protocol/runs/). Its arguments are defined by `main`, including explicit cloud execution, training and result-write confirmations. Use [posthoc100_modal.py](../experiments/posthoc100/posthoc100_modal.py) with a `run_key` from [POSTHOC_REGISTER.json](../experiments/posthoc100/POSTHOC_REGISTER.json), which also requires a claim-write confirmation. Use one condition at a time. An actual cloud submission requires sufficient storage, a T4 allocation and permission to incur compute charges.

No cloud job was launched to test this public packaging. The original trainers were verified through their saved execution records. This release includes their original model, weighting and training code. See [provenance](PROVENANCE.md) for the changed launch settings and hashes. Original interruption/recovery plans and lineage metadata are private execution records.

## Evaluation and audit evidence

The [official-test evaluator modules](../evaluation/official_test/sstre_test/) implement the frozen seven-checkpoint test. Running the original evaluator requires the original locked model-selection manifest, evaluation rules, protocol bundle, checkpoints and official test archives. The released training configuration contains packaging-only changes and therefore has a different protocol file hash from the original evaluator's lock. This release provides the verified aggregate test table. The exact locked evaluator inputs and private launcher are outside the public bundle.

The [endpoint audit driver](../evaluation/reliability/analyze_missing_local.py) is the original local analysis with configurable evidence and output paths. It expects the verified split, masks, saved trajectory scores, earlier analysis tables and receipts in their original relative evidence layout. Supply that root as `SSTRE_AUDIT_EVIDENCE`, the accepted CIFAR-10N Stage-Aware seed-43 audit directory as `SSTRE_STAGE43_EVIDENCE`, and a new output directory as `SSTRE_AUDIT_OUTPUT`. The original receipt's local readback references must resolve to your authorized local copies. With all of those inputs, the entrypoint is `python evaluation/reliability/analyze_missing_local.py`. Its proportional per-class policy is an exploratory counterfactual separate from the frozen global top-70% result. The public result tables contain aggregate verified values. The per-sample archives, masks and private receipts required to rerun this driver are held separately.

