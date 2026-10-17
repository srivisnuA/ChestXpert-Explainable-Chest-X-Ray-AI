"""Validate image and patient isolation before training or threshold tuning."""
import hashlib
import json


def manifest_hash(names):
    return hashlib.sha256(json.dumps(sorted(set(names))).encode("utf-8")).hexdigest()


def verify_validation_manifest(checkpoint, names):
    expected = checkpoint.get("split_hashes", {}).get("validation")
    if not expected or expected != manifest_hash(names):
        raise ValueError("Checkpoint does not confirm this validation manifest was reserved during training; retrain with --validation-list")


def split_metadata(df, train_names, test_names, validation_names=None):
    train_names, test_names = set(train_names), set(test_names)
    validation_names = set(validation_names or [])
    if df["Image Index"].isna().any() or df["Image Index"].duplicated().any():
        raise ValueError("Metadata image identifiers must be present and unique")
    if not train_names or not test_names:
        raise ValueError("Training and test manifests must be non-empty")
    if train_names & test_names:
        raise ValueError("Training/test image overlap detected")
    if not validation_names <= train_names:
        raise ValueError("Validation images must come from the training pool")
    known = set(df["Image Index"])
    if (train_names | test_names) - known:
        raise ValueError("Split manifest contains images missing from metadata")
    groups = [df[df["Image Index"].isin(names)].copy() for names in
              (train_names - validation_names, validation_names, test_names)]
    if groups[0].empty:
        raise ValueError("Validation selection leaves no training images")
    if "Patient ID" in df.columns:
        if df["Patient ID"].isna().any():
            raise ValueError("Patient identifiers must be present for isolation checks")
        patients = [set(group["Patient ID"].astype(str)) for group in groups]
        if any(patients[i] & patients[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Patient overlap detected between training, validation or test sets")
    return tuple(groups)
