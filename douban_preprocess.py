"""Prepare Douban movie/music implicit-feedback data from quoted TSV reviews.


"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_interactions(path: Path, item_column: str, encoding: str = "utf-8-sig"):
    """Read by header name; deduplicate user-item pairs and ignore rating magnitude."""
    pairs = set()
    csv.field_size_limit(64 * 1024 * 1024)
    with Path(path).open(encoding=encoding, newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", quotechar='"', strict=True)
        required = {"user_id", item_column, "rating"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected columns {sorted(required)}; got {reader.fieldnames}")
        for row in reader:
            if None in row or any(row.get(k) is None for k in required):
                raise ValueError(f"{path}:{reader.line_num}: malformed TSV record")
            user, item = row["user_id"].strip(), row[item_column].strip()
            if not user or not item or any(c.isspace() for c in user + item):
                raise ValueError(f"{path}:{reader.line_num}: missing or invalid identifier")
            pairs.add((user, item))
    if not pairs:
        raise ValueError(f"{path}: no interactions")
    return pairs


def filter_core(pairs, minimum: int):
    """Iterate user/item degree filtering to a stable k-core of distinct pairs."""
    if minimum < 3:
        raise ValueError("minimum must be at least 3 to support train/validation/test")
    pairs = set(pairs)
    while pairs:
        users = Counter(u for u, _ in pairs)
        items = Counter(v for _, v in pairs)
        kept = {(u, v) for u, v in pairs if users[u] >= minimum and items[v] >= minimum}
        if len(kept) == len(pairs):
            return kept
        pairs = kept
    raise ValueError("Filtering removed all interactions; inspect the input and threshold")


def index_domain(pairs):
    users = {u: i for i, u in enumerate(sorted({u for u, _ in pairs}))}
    items = {v: i for i, v in enumerate(sorted({v for _, v in pairs}))}
    histories = defaultdict(list)
    for u, v in sorted(pairs):
        histories[users[u]].append(items[v])
    return users, items, dict(histories)


def split_histories(histories, seed: int, domain: str, target: bool):
    """Deterministic per-user random splitting, with nonempty held-out sets."""
    train, valid, test = {}, {}, {}
    for user, history in sorted(histories.items()):
        items = sorted(set(history))
        digest = hashlib.sha256(f"{seed}:{domain}:{user}".encode()).digest()
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(items)
        n = len(items)
        if n < (3 if target else 2):
            raise ValueError(f"User {user} has too few interactions")
        nt = min(max(1, int(0.8 * n)), n - (2 if target else 1))
        nv = min(max(1, int(0.1 * n)), n - nt - 1) if target else n - nt
        train[user], valid[user] = items[:nt], items[nt:nt + nv]
        if target:
            test[user] = items[nt + nv:]
        assert not (set(train[user]) & set(valid[user]))
        assert not (set(train[user]) & set(test.get(user, [])))
        assert not (set(valid[user]) & set(test.get(user, [])))
    return {"train": train, "valid": valid, "test": test}


def write_mapping(path, mapping):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for raw, idx in sorted(mapping.items(), key=lambda x: x[1]):
            stream.write(f"{raw} {idx}\n")


def write_histories(path, histories):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for user, items in sorted(histories.items()):
            if items:
                stream.write(f"{user} " + " ".join(map(str, sorted(items))) + "\n")


def prepare_douban(movie_file, music_file, output_dir, min_interactions=10, seed=2026,
                   encoding="utf-8-sig"):
    """Create a new output directory. Refuse overwrites to protect existing splits."""
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new or empty output directory: {output}")
    domains = {}
    for name, path, col in [("movie", movie_file, "movie_id"),
                            ("music", music_file, "music_id")]:
        domains[name] = index_domain(filter_core(read_interactions(Path(path), col, encoding),
                                                min_interactions))
    shared = sorted(set(domains["movie"][0]) & set(domains["music"][0]))
    if not shared:
        raise ValueError("The filtered domains have no overlapping user IDs")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": seed, "minimum_degree": min_interactions,
                "filter": "iterative user/item k-core after pair deduplication",
                "feedback": "interaction presence; rating magnitude ignored",
                "overlapping_users": len(shared), "domains": {}, "tasks": {}}
    for name, (users, items, histories) in domains.items():
        folder = output / name
        folder.mkdir()
        write_mapping(folder / "user-map.txt", users)
        write_mapping(folder / "item-map.txt", items)
        manifest["domains"][name] = {"users": len(users), "items": len(items),
                                      "interactions": sum(map(len, histories.values()))}
    overlap_dir = output / "overlap"
    overlap_dir.mkdir()
    for source, target in [("movie", "music"), ("music", "movie")]:
        task = output / f"{source}-{target}"
        task.mkdir()
        mapping = [(domains[source][0][u], domains[target][0][u]) for u in shared]
        with (overlap_dir / f"{source}-{target}-overlap.txt").open("w", encoding="utf-8") as f:
            for a, b in mapping:
                f.write(f"{a} {b}\n")
        for role, name in [("source", source), ("target", target)]:
            users, items, histories = domains[name]
            splits = split_histories(histories, seed, name, target=(role == "target"))
            folder = task / role
            folder.mkdir()
            write_mapping(folder / "user-map.txt", users)
            write_mapping(folder / "item-map.txt", items)
            for split, records in splits.items():
                write_histories(folder / f"{split}.txt", records)
            if role == "target":
                eligible = {b for _, b in mapping}
                for split, records in splits.items():
                    write_histories(task / f"{split}.txt",
                                    {u: v for u, v in records.items() if u in eligible})
                manifest["tasks"][task.name] = {
                    "evaluated_users": len(eligible),
                    "target_counts": {split: sum(len(v) for u, v in records.items() if u in eligible)
                                      for split, records in splits.items()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie", type=Path, required=True)
    parser.add_argument("--music", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-interactions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--encoding", default="utf-8-sig")
    args = parser.parse_args()
    result = prepare_douban(args.movie, args.music, args.output, args.min_interactions,
                            args.seed, args.encoding)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
