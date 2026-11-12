# Source provenance and public packaging

The main training bundle comes from `experiments/phase6_main100_v2_20261026_01_retention_v1_writer_v2` in the original research workspace. The paired seeds and ablations come from `experiments/phase6_posthoc100_v1_20261028_01`. Their original SHA-256 manifests are preserved in [provenance](../provenance/). [FILE_PROVENANCE.json](../provenance/FILE_PROVENANCE.json) maps each published research file to its original relative path, original digest, release digest and any packaging change. These are source hashes from saved experiment bundles. The repository's historical Git commit identifies the early Colab development snapshot.

All model, reliability, training and checkpoint modules under each bundle's `src/sstre100` are byte copies. The reviewed protocol mathematics and validation code are byte copies. The original source manifests, saved run preparation evidence and independent completion reports link the main Fixed/Stage-Aware source to the completed 100-epoch runs. The post-hoc manifest pins the thirteen-condition extension. The evaluator modules match the saved final official-test bundle manifest. The root-cause analysis matches its original package manifest before the input-path adaptation.

The public copy makes these packaging changes:

- Put the two independent bundles under `experiments/main100` and `experiments/posthoc100`, retaining their required relative layout.
- Configure the launchers' account resources using explicit environment variables. The main bundle omits the separate storage smoke script. Historical launch approvals, operational scripts and recovery receipts stay outside the release.
- Remove personal paths and account storage hints from reviewed metadata. Recompute the public configuration locks, dependent run digests, post-hoc base digest and public source manifests. The original manifests remain available for comparison. The release config hashes therefore differ from those attached to the historical checkpoints.
- Parameterize the CIFAR-100N freeze and audit drivers' input and output locations. The freeze driver reads the same two verified source digests from a minimal public record. The dataset construction, signal, loss, ranking and metric calculations are unchanged.
- Copy verified aggregate tables. Remove the local source-path column from paired tables that contain it. Metric cells are retained.
- Preserve checked-in bytes of hashed files with `.gitattributes`. Extend `.gitignore` for credentials, caches and generated research assets.

The official test runtime and reliability audit evidence have separate input requirements in the [reproduction guide](REPRODUCTION.md). The original two notebooks remain historical material. No third-party repository or package source is bundled. PyTorch, TorchVision, NumPy, Pillow and Modal are external dependencies distributed under their own terms.

