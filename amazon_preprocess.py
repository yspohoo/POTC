"""
Amazon-2014 JSON Data Preprocessing for Ranking-based Cross-Domain Recommendation (CDR).

Pipeline
--------
1. `build_domain_mapping`   : map raw user/item JSON ids to contiguous integer
                               indices (single-domain), and write out the full
                               interaction list per user as `interactions.txt`.
2. `build_overlap_mapping`  : find users shared between two domains and save
                               their aligned (mapped) ids as an overlap file.
3. `build_ranking_split`    : for each overlap user, split their item history
                               in the target domain 8:1:1 into train/valid/test
                               and save each as a `user item1 item2 ...` file.

Interaction presence defines implicit feedback; rating magnitudes are ignored.
Encoder graphs contain only the training split of the selected domain role.
"""

import os
import shutil
from douban_preprocess import split_histories, write_histories
from data_protocol import read_rankings, validate_task
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing as pp

from variable import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# --------------------------------------------------------------------------- #
# Step 1: Single-domain id mapping + full interaction list (ranking format)
# --------------------------------------------------------------------------- #
def build_domain_mapping(sfile, domain, inter):
    """
    Map raw user/item JSON ids to contiguous integer indices for a single
    domain, filter out low-activity users/items, and save:
      - user-map.txt / item-map.txt : raw id -> mapped index
      - interactions.txt            : one line per user, "user item1 item2 ..."
                                       listing every interacted item (ranking
                                       format, rating value is not used)

    Args:
        sfile (str): path to the raw Amazon domain JSON file.
        inter (int): minimum number of interactions per user/item to keep.

    Returns:
        user_counts_after (pd.Series): interaction count per user, post-filter.
        item_counts_after (pd.Series): interaction count per item, post-filter.
        df (pd.DataFrame): filtered interactions with mapped id columns
                            ('user_id_idx', 'item_id_idx').
    """
    df = pd.read_json(sfile, lines=True)
    df = df[['reviewerID', 'asin', 'overall']].drop_duplicates(['reviewerID', 'asin'])
    df.columns = ['user_id', 'item_id', 'rating']
    print(len(df))

    # Check the number of interactions per user and per item
    user_counts = df['user_id'].value_counts()
    item_counts = df['item_id'].value_counts()
    print("Before filtering:")
    print(f"Total users: {len(user_counts)}")
    print(f"Total items: {len(item_counts)}")

    # Keep users with at least inter interactions
    df = df[df['user_id'].isin(user_counts[user_counts >= inter].index)]
    # Keep items with at least inter interactions
    df = df[df['item_id'].isin(item_counts[item_counts >= inter + 5].index)]

    # Recalculate stats after filtering
    user_counts_after = df['user_id'].value_counts()
    item_counts_after = df['item_id'].value_counts()
    print("\nAfter filtering:")
    print(f"Total users: {len(user_counts_after)}")
    print(f"Total items: {len(item_counts_after)}")
    print(f"Total interactions: {len(df)}")

    # User & item mapping
    le_user = pp.LabelEncoder()
    le_item = pp.LabelEncoder()
    df['user_id_idx'] = le_user.fit_transform(df['user_id'].values)
    df['item_id_idx'] = le_item.fit_transform(df['item_id'].values)
    n_users = df['user_id_idx'].nunique()
    n_items = df['item_id_idx'].nunique()
    print("Number of Unique Users : ", n_users)
    print("Number of unique Items : ", n_items)

    # Map dict, txt file
    le_user_mapping = dict(zip(le_user.classes_, le_user.transform(le_user.classes_)))
    le_item_mapping = dict(zip(le_item.classes_, le_item.transform(le_item.classes_)))

    Path(user_map_paths[domain]).parent.mkdir(parents=True, exist_ok=True)
    with open(user_map_paths[domain], "w") as output:
        for k, v in le_user_mapping.items():
            output.write(f'{k} {v}\n')
    with open(item_map_paths[domain], "w") as output:
        for k, v in le_item_mapping.items():
            output.write(f'{k} {v}\n')

    # Ranking format: one line per user listing every interacted item,
    # rating value is discarded, existence of an interaction is all that matters
    train_path = os.path.join(str(Path(user_map_paths[domain]).parent), "interactions.txt")
    save_user_item_list(df, train_path)
    print(f"Saved full preprocessing interactions to: {train_path}")

    return user_counts_after, item_counts_after, df


def save_user_item_list(df, out_path):
    """
    Write interactions in ranking format: one line per user,
    "user item1 item2 item3 ...", sorted by user id.

    Args:
        df (pd.DataFrame): must contain 'user_id_idx' and 'item_id_idx'.
        out_path (str): destination file path.
    """
    grouped = df.groupby('user_id_idx')['item_id_idx'].apply(list).sort_index()
    with open(out_path, "w") as output:
        for user, items in grouped.items():
            item_str = ' '.join(str(i) for i in items)
            output.write(f'{user} {item_str}\n')


# --------------------------------------------------------------------------- #
# Step 2: Cross-domain overlap user mapping
# --------------------------------------------------------------------------- #
def load_domain_user_maps(user_map_paths):
    """
    Load every domain's user-map.txt into a dict of DataFrames.

    Args:
        user_map_paths (dict): domain name -> path to that domain's user-map.txt

    Returns:
        dict: domain name -> DataFrame with columns ['original_id', 'mapped_id']
    """
    mappings = {}
    for domain, path in user_map_paths.items():
        df = pd.read_csv(path, sep=r"\s+", header=None, names=["original_id", "mapped_id"])
        mappings[domain] = df
        print(f"Loaded {domain}: {len(df)} users")
    return mappings


def build_overlap_mapping(mappings, domain_a, domain_b):
    """
    Find users shared between two domains and save their aligned mapped ids.

    Args:
        mappings (dict): domain name -> DataFrame from load_domain_user_maps.
        domain_a (str): source domain name.
        domain_b (str): target domain name.

    Returns:
        out_path (str): path to the saved overlap file.
        out_df (pd.DataFrame): overlap mapped ids, columns ['mapped_id_s', 'mapped_id_t'].
    """
    df_a = mappings[domain_a]
    df_b = mappings[domain_b]

    # Inner join on original_id to find overlap
    overlap = pd.merge(
        df_a, df_b, on="original_id", how="inner",
        suffixes=(f"_{domain_a}", f"_{domain_b}")
    )
    overlap.columns = ["original_id", "mapped_id_s", "mapped_id_t"]

    out_path = os.path.join(overlap_save, f"{domain_a}-{domain_b}-overlap.txt")
    Path(overlap_save).mkdir(parents=True, exist_ok=True)
    overlap[["mapped_id_s", "mapped_id_t"]].to_csv(out_path, sep=' ', index=False, header=False)
    print(f"Saved {len(overlap)} overlaps -> {out_path}")

    return out_path, overlap[["mapped_id_s", "mapped_id_t"]]


# --------------------------------------------------------------------------- #
# Step 3: Ranking train/valid/test split (8:1:1, per overlap user)
# --------------------------------------------------------------------------- #
def build_ranking_split(source, target, target_df, save_dir, seed=42):
    """Create role-specific training graphs and overlap-only mapping splits."""
    task = Path(save_dir)
    if task.exists() and any(task.iterdir()):
        raise FileExistsError(f"Use a new or empty task directory: {task}")
    overlap_path = Path(overlap_save) / f"{source}-{target}-overlap.txt"
    overlap_df = pd.read_csv(overlap_path, sep=r"\s+", header=None,
                             names=["mapped_id_s", "mapped_id_t"])
    eligible = set(overlap_df["mapped_id_t"])
    source_histories = read_rankings(Path(user_map_paths[source]).parent / "interactions.txt")
    target_histories = {int(u): sorted(set(map(int, group["item_id_idx"])))
                        for u, group in target_df.groupby("user_id_idx")}
    task.mkdir(parents=True, exist_ok=True)
    result = {}
    for role, name, histories in [("source", source, source_histories),
                                   ("target", target, target_histories)]:
        splits = split_histories(histories, seed, name, target=(role == "target"))
        folder = task / role
        folder.mkdir()
        shutil.copy2(user_map_paths[name], folder / "user-map.txt")
        shutil.copy2(item_map_paths[name], folder / "item-map.txt")
        for split, records in splits.items():
            write_histories(folder / f"{split}.txt", records)
            if role == "target":
                mapping_records = {u: items for u, items in records.items() if u in eligible}
                write_histories(task / f"{split}.txt", mapping_records)
                result[split] = _rows_to_df([(u, np.asarray(items)) for u, items in mapping_records.items()])
    validate_task(task)
    return result


def _rows_to_df(rows):
    """Expand a list of (user, item_array) pairs into a flat DataFrame."""
    if not rows:
        return pd.DataFrame(columns=['user_id_idx', 'item_id_idx'])
    users = np.concatenate([np.full(len(items), user) for user, items in rows])
    items = np.concatenate([items for _, items in rows])
    return pd.DataFrame({'user_id_idx': users, 'item_id_idx': items})


# --------------------------------------------------------------------------- #
# Example usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Step 1: map each domain's JSON ids and write per-domain interactions.txt
    book_user_count, book_item_count, book_df = build_domain_mapping(amazon_book, "book", 15)
    music_user_count, music_item_count, music_df = build_domain_mapping(amazon_music, "music", 10)
    movie_user_count, movie_item_count, movie_df = build_domain_mapping(amazon_movie, "movie", 10)

    # Step 2: build overlap mapping between two domains
    mappings = load_domain_user_maps(user_map_paths)
    overlap_path, overlap_df = build_overlap_mapping(mappings, "book", "movie")
    overlap_path, overlap_df = build_overlap_mapping(mappings, "movie", "book")
    overlap_path, overlap_df = build_overlap_mapping(mappings, "movie", "music")
    overlap_path, overlap_df = build_overlap_mapping(mappings, "music", "movie")

    # Step 3: ranking-style 8:1:1 split for overlap users in the target domain
    splits = build_ranking_split(
        source="book",
        target="movie",
        target_df=movie_df,
        save_dir=book_movie_save,
    )
    splits = build_ranking_split(
            source="movie",
            target="music",
            target_df=music_df,
            save_dir=movie_music_save,
        )
    splits = build_ranking_split(
            source="music",
            target="movie",
            target_df=movie_df,
            save_dir=music_movie_save,
        )
    splits = build_ranking_split(
                source="movie",
                target="book",
                target_df=book_df,
                save_dir=movie_book_save,
            )
    pass