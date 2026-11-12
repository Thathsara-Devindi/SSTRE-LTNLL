"""Freeze approved CIFAR100N v1 using the locked constructor, then audit locally.

No network, Modal, training, official-test decoding, source modification, or overwrite.
"""
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ADAPTER = HERE.parent / "experiments/main100"
sys.path.insert(0, str(ADAPTER / "src"))
from sstre100.freeze100 import prepare_from_verified_sources
from sstre100.data import cached_images, logical_fingerprint
from sstre100.protocol import load, sha256


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name].copy() for name in values.files}


def main():
    CACHE = Path(os.environ["SSTRE_CIFAR100_SOURCE_DIR"])
    DESTINATION = Path(os.environ["SSTRE_CIFAR100_FREEZE_DIR"])
    REPORT_DIR = Path(os.environ["SSTRE_CIFAR100_REPORT_DIR"])
    require(not DESTINATION.exists(), "Freeze directory exists; never overwrite")
    report_path = REPORT_DIR / "FREEZE_VALIDATION_REPORT.json"
    require(not report_path.exists(), "Validation report exists; never overwrite")
    manifest = json_read(ADAPTER / "SOURCE_MANIFEST.json")
    for name, expected in manifest.items():
        require(sha256(ADAPTER / name) == expected, "Validated adapter changed: " + name)
    protocol, contract, run, tools = load("cifar100n", "ce")
    expected_counts = np.array([math.floor(450 * 0.1 ** (c / 99)) for c in range(100)])
    require(np.array_equal(expected_counts, contract["expected_source_class_counts"]), "Count rule differs")
    require(int(expected_counts.sum()) == 17614 and expected_counts[0] / expected_counts[-1] == 10,
            "Expected long-tail sample count or IF differs")
    acquired = json_read(HERE / "cifar100n_source_hashes.json")
    archive = CACHE / "cifar-100-python.tar.gz"
    annotation = CACHE / "CIFAR-100_human.pt"
    require(sha256(archive) == acquired["images"]["sha256"], "Archive source changed")
    require(sha256(annotation) == acquired["annotations"]["sha256"], "Annotation source changed")
    unsafe_globals = set(torch.serialization.get_unsafe_globals_in_checkpoint(annotation))
    require(unsafe_globals == {"numpy.ndarray", "numpy.dtype", "numpy.core.multiarray._reconstruct"},
            "Unexpected serialized annotation globals")
    # The inspected official release contains int64 NumPy arrays. The scoped
    # allowlist permits only those data constructors; weights_only remains True.
    allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype("int64"))]
    with torch.serialization.safe_globals(allowed):
        annotations = torch.load(annotation, map_location="cpu", weights_only=True)
    require(set(annotations) == {"clean_label", "noisy_label", "clean_coarse_label", "noisy_coarse_label"},
            "Official annotation keys differ")
    clean = np.asarray(annotations["clean_label"])
    noisy = np.asarray(annotations["noisy_label"])
    for name, values in [("clean_label", clean), ("noisy_label", noisy)]:
        require(values.shape == (50000,) and values.dtype.kind in "iu", "Invalid " + name)
        require(np.array_equal(np.unique(values), np.arange(100)), "Not 100 fine classes: " + name)
    pixels, source_clean = cached_images(archive, acquired["images"]["sha256"], 100)
    require(np.array_equal(clean, source_clean), "Full 50,000-row image/annotation alignment failed")
    require(pixels.shape == (50000, 32, 32, 3) and pixels.dtype == np.uint8, "Image format differs")
    del pixels
    provenance = {
        "acquisition": acquired,
        "locked_contract_sha256": sha256(ADAPTER / "reviewed_protocol/datasets/cifar100n.json"),
        "locked_protocol_sha256": sha256(ADAPTER / "reviewed_protocol/protocol.json"),
        "constructor_sha256": sha256(ADAPTER / "reviewed_protocol/protocol_tools.py"),
        "freeze_adapter_sha256": sha256(ADAPTER / "src/sstre100/freeze100.py"),
        "source_train_member": "cifar-100-python/train",
        "source_train_member_official_md5": "16019d7e3df5f24257cddd939b257f8d",
        "annotation_format": "official Python-order release; fine labels only",
        "annotation_loading": "weights_only=True; scoped inspected NumPy array/int64 constructors",
        "full_clean_label_alignment_count": 50000,
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "official_test_decoded": False,
    }
    with torch.serialization.safe_globals(allowed):
        receipt = prepare_from_verified_sources(
            archive, annotation, DESTINATION, contract, tools,
            expected_archive_sha256=acquired["images"]["sha256"],
            expected_annotation_sha256=acquired["annotations"]["sha256"],
            provenance=provenance, approved=True)
    train = arrays(DESTINATION / "train.npz")
    validation = arrays(DESTINATION / "validation.npz")
    audit = arrays(DESTINATION / "AUDIT_ONLY.npz")
    require(set(train) == {"original_ids", "observed_labels"}, "Training contains extra/clean fields")
    require(set(validation) == {"original_ids", "targets"}, "Validation fields differ")
    require(set(audit) == {"clean_labels", "is_noisy"}, "Audit fields differ")
    ids, val_ids, observed = train["original_ids"], validation["original_ids"], train["observed_labels"]
    require(ids.shape == (17614,) and val_ids.shape == (5000,), "Frozen counts differ")
    require(len(np.unique(ids)) == len(ids) and len(np.unique(val_ids)) == len(val_ids), "Duplicate IDs")
    require(np.all((ids >= 0) & (ids < 50000)) and np.all((val_ids >= 0) & (val_ids < 50000)), "ID bounds")
    require(np.intersect1d(ids, val_ids).size == 0, "Training-validation leakage")
    require(np.array_equal(observed, noisy[ids]), "Training labels are not the official noisy labels")
    require(np.array_equal(validation["targets"], clean[val_ids]), "Validation clean-label alignment")
    require(np.array_equal(audit["clean_labels"], clean[ids]), "Audit clean-label alignment")
    require(np.array_equal(audit["is_noisy"], clean[ids] != observed), "Audit noise flag alignment")
    require(np.array_equal(np.bincount(clean[ids], minlength=100), expected_counts), "Long-tail counts")
    require(np.array_equal(np.bincount(clean[val_ids], minlength=100), np.full(100, 50)), "Validation counts")
    # Independent reconstruction checks the saved ordered arrays, without writing
    # another split or changing the already frozen one.
    rng = np.random.Generator(np.random.PCG64(42))
    expected_train, expected_val = [], []
    for c in range(100):
        permutation = rng.permutation(np.flatnonzero(clean == c))
        expected_val.extend(permutation[:50])
        expected_train.extend(permutation[50:50 + int(expected_counts[c])])
    expected_train = rng.permutation(np.asarray(expected_train, dtype="<i8"))
    expected_val = rng.permutation(np.asarray(expected_val, dtype="<i8"))
    require(np.array_equal(ids, expected_train) and np.array_equal(val_ids, expected_val),
            "Independent seed42 PCG64 reconstruction differs")
    fingerprint = logical_fingerprint(ids, val_ids, observed, b"cifar100n_fine_IR10_seed42_v1\0")
    require(fingerprint == receipt["fingerprint"], "Logical split fingerprint differs")
    for name, expected in receipt["files_sha256"].items():
        require(sha256(DESTINATION / name) == expected, "Frozen export hash mismatch: " + name)
    for name, expected in manifest.items():
        require(sha256(ADAPTER / name) == expected, "Preserved adapter modified: " + name)
    with (REPORT_DIR / "CLASS_COUNTS_AUDIT_ONLY.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fine_class", "selected_source_class_count", "validation_count", "observed_training_count"])
        observed_counts = np.bincount(observed, minlength=100)
        for c in range(100):
            writer.writerow([c, int(expected_counts[c]), 50, int(observed_counts[c])])
    files = {name: {"sha256": sha256(DESTINATION / name), "bytes": (DESTINATION / name).stat().st_size}
             for name in ["train.npz", "validation.npz", "AUDIT_ONLY.npz", "FREEZE_RECEIPT.json"]}
    report = {
        "status": "PASS", "utc": datetime.now(timezone.utc).isoformat(), "dataset": "cifar100n",
        "frozen_directory": str(DESTINATION), "cache_directory": str(CACHE),
        "seed": 42, "rng": "numpy.Generator(numpy.random.PCG64(42))", "numpy_version": np.__version__,
        "imbalance_factor": 10, "training_samples": len(ids), "validation_samples": len(val_ids),
        "validation_per_class": 50, "classes": 100, "observed_annotation_key": "noisy_label",
        "clean_annotation_key": "clean_label", "full_image_label_order_alignment_count": 50000,
        "train_val_disjoint": True, "independent_order_reconstruction": "PASS",
        "no_clean_labels_in_training_export": True, "test_decoded_or_evaluated": False,
        "fingerprint": fingerprint, "source_class_counts": expected_counts.tolist(),
        "observed_training_class_counts": observed_counts.tolist(),
        "source_noise_rate_AUDIT_ONLY": float(np.mean(clean != noisy)),
        "selected_noise_count_AUDIT_ONLY": int(audit["is_noisy"].sum()),
        "selected_noise_rate_AUDIT_ONLY": float(audit["is_noisy"].mean()),
        "archive_sha256": acquired["images"]["sha256"], "archive_md5": acquired["images"]["md5"],
        "annotation_sha256": acquired["annotations"]["sha256"],
        "annotation_git_commit": acquired["annotation_commit"], "files": files,
        "validated_adapter_source_files_unchanged": len(manifest),
        "local_freeze_only": True, "modal_upload_performed": False, "training_performed": False,
        "next_experiment": "CIFAR100N CE seed42 100epochs after separately approved input transfer and GPU execution",
    }
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    lines = ["# CIFAR-100N dataset freeze: PASS", "", "26 October 2026. Local preparation only.", "",
             f"Frozen directory: `{DESTINATION}`", f"Fingerprint: `{fingerprint}`", "",
             "Official Python CIFAR-100 archive MD5 and SHA-256 verified; official CIFAR-100N Git blob verified.",
             "All 50,000 source clean labels match the image training-row order. Fine noisy labels feed training.",
             "Seed42 PCG64 split independently reproduced: 17,614 train; 5,000 validation; 50 validation per class; IF10.",
             "Training and validation IDs are unique and disjoint; separate training/validation/audit exports verified.",
             "weights_only=True was retained with an inspected, scoped NumPy int64 array allowlist.",
             "Official test data were not decoded/evaluated. Existing source/configuration hashes remain unchanged.",
             "", "Modal input transfer and CIFAR-100N CE training require separate approval."]
    with (REPORT_DIR / "FREEZE_VALIDATION_REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n\n".join(lines) + "\n")
    with (REPORT_DIR / "RESEARCH_PROGRESS.md").open("x", encoding="utf-8") as stream:
        stream.write("# Research progress — 26 October 2026\n\nCIFAR10N CE main100 verification closed PASS; no checks repeated.\n\n"
                     "CIFAR100N official sources acquired and separate frozen dataset validated PASS. "
                     f"See FREEZE_VALIDATION_REPORT.json and {DESTINATION}.\n\n"
                     "No Modal connection/upload, cloud jobs, GPU training, official-test evaluation, package install, or GitHub mutation.\n\n"
                     "Next experiment: CIFAR100N CE seed42,100epochs under the unchanged locked protocol; "
                     "requires approved input transfer and separate GPU execution approval.\n")
    print(json.dumps({k: v for k, v in report.items() if "class_counts" not in k}, indent=2), flush=True)


if __name__ == "__main__":
    main()
