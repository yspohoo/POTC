"""Task-specific split paths and provenance checks for frozen graph encoders."""
import hashlib
from pathlib import Path


def task_directory(dataset, source, target):
    from variable import absolute_path
    if dataset not in {"amazon", "douban"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    root = Path(absolute_path) / "data" / dataset
    if dataset == "douban":
        root = root / "processed"
    return root / f"{source}-{target}"


def read_rankings(path):
    records = {}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            parts = line.split()
            if not parts:
                continue
            user, *items = map(int, parts)
            if user < 0 or any(item < 0 for item in items) or user in records:
                raise ValueError(f"Invalid or repeated user record at {path}:{number}")
            if len(items) != len(set(items)):
                raise ValueError(f"Repeated item at {path}:{number}")
            if items:
                records[user] = items
    return records


def graph_fingerprint(folder):
    """Identify graph edges and ID spaces without using held-out interactions."""
    folder = Path(folder)
    digest = hashlib.sha256()
    for name in ["train.txt", "user-map.txt", "item-map.txt"]:
        digest.update(name.encode())
        with (folder / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def check_disjoint(folder):
    folder = Path(folder)
    splits = {s: read_rankings(folder / f"{s}.txt") for s in ["train", "valid", "test"]}
    for user in set().union(*(set(v) for v in splits.values())):
        train, valid, test = [set(splits[s].get(user, [])) for s in ["train", "valid", "test"]]
        if train & valid or train & test or valid & test:
            raise ValueError(f"Overlapping interaction splits for user {user} in {folder}")
    return splits


def validate_task(task):
    """Require each mapping split to match the corresponding target graph split."""
    task = Path(task)
    check_disjoint(task / "source")
    target = check_disjoint(task / "target")
    mapping = check_disjoint(task)
    for split in ["train", "valid", "test"]:
        for user, items in mapping[split].items():
            if set(items) != set(target[split].get(user, [])):
                raise ValueError(f"Mapping and target {split} differ for user {user}")


def load_encoder_state(model, checkpoint, graph_dir, device):
    import torch
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("graph_fingerprint") != graph_fingerprint(graph_dir):
        raise ValueError(f"Checkpoint does not match the training graph: {checkpoint}. Pretrain this task again.")
    model.load_state_dict(payload["state_dict"])
