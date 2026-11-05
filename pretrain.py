"""Pretrain domain encoders on the source/target training graphs of one task."""

import os
import argparse
from data_protocol import task_directory, validate_task, graph_fingerprint
from time import time

import numpy as np
import torch
import torch.optim as optim

from models import LightGCN
from utils_edited import (
    Loader,
    UniformSample_pretrain,
    pretrain_minibatch,
    shuffle,
    RecallPrecision_ATk,
    NDCGatK_r,
    getLabel,
)
from variable import *

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, dataset, optimizer, device, batch_size=BATCH_SIZE, weight_decay=WEIGHT_DECAY):
    """Runs one BPR training epoch over freshly sampled triplets."""
    model.train()
    S = UniformSample_pretrain(dataset.allPos, dataset.n_user, dataset.m_item, dataset.trainDataSize)
    users = torch.LongTensor(S[:, 0]).to(device)
    posItems = torch.LongTensor(S[:, 1]).to(device)
    negItems = torch.LongTensor(S[:, 2]).to(device)
    users, posItems, negItems = shuffle(users, posItems, negItems)

    total_loss = 0.
    n_batches = 0
    for batch_users, batch_pos, batch_neg in pretrain_minibatch(
        users, posItems, negItems, batch_size=batch_size
    ):
        loss, reg_loss = model.bpr_loss(batch_users, batch_pos, batch_neg)
        full_loss = loss + reg_loss * weight_decay
        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()
        total_loss += full_loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(model, dataset, device, k=10, test_batch_size=100):
    """
    Recall@k / Precision@k / NDCG@k on held-out interactions.
    Returns None when the validation split is empty.
    """
    test_dict = dataset.validDict
    if len(test_dict) == 0:
        print("No validation data available for this domain, skipping evaluation.")
        return None

    model.eval()
    users = list(test_dict.keys())
    recalls, precisions, ndcgs = [], [], []

    with torch.no_grad():
        for start in range(0, len(users), test_batch_size):
            batch_users = users[start:start + test_batch_size]
            ground_truth = [test_dict[u] for u in batch_users]
            all_pos = dataset.getUserPosItems(batch_users)

            batch_users_t = torch.LongTensor(batch_users).to(device)
            ratings = model.getUsersRating(batch_users_t)
            for i, pos in enumerate(all_pos):
                ratings[i, pos] = -float('inf')  # exclude items already seen in training

            _, top_items = torch.topk(ratings, k=k)
            top_items = top_items.cpu().numpy()

            r = getLabel(ground_truth, top_items)
            pr = RecallPrecision_ATk(ground_truth, r, k)
            recalls.append(pr['recall'])
            precisions.append(pr['precision'])
            ndcgs.append(NDCGatK_r(ground_truth, r, k))

    n = len(users)
    results = {
        'recall': float(np.sum(recalls) / n),
        'precision': float(np.sum(precisions) / n),
        'ndcg': float(np.sum(ndcgs) / n),
    }
    print(f"Recall@{k}: {results['recall']:.5f}  "
          f"Precision@{k}: {results['precision']:.5f}  "
          f"NDCG@{k}: {results['ndcg']:.5f}")
    return results


def pretrain_domain(domain, device, data_path, weight_path, dataset_name='amazon'):
    """Trains and saves a LightGCN model for a single domain."""
    print(f"\n===== Pretraining LightGCN on domain: {domain} =====")

    dataset = Loader(path=data_path, device=device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = LightGCN(dataset, EMBBEDDING_DIM, NUM_LAYERS, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LIGHTGCN_LR)

    if not dataset.validDict:
        raise ValueError("Validation interactions are required for early stopping")
    train_losses = []
    best_recall = -float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    for it in range(PRETRAIN_ITERATIONS):
        loss = train_one_epoch(model, dataset, optimizer, device,
                               weight_decay={'amazon': 1e-06, 'douban': 1e-07}[dataset_name])
        train_losses.append(loss)
        metrics = evaluate(model, dataset, device, k=10)
        score = metrics['recall']
        if not np.isfinite(score):
            raise ValueError("Validation Recall is not finite")
        if score > best_recall + 0.0:
            best_recall = score
            best_epoch = it + 1
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= PRETRAIN_PATIENCE:
            break
    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    torch.save({"state_dict": best_state,
                "graph_fingerprint": graph_fingerprint(data_path),
                "epoch": best_epoch, "validation_recall": best_recall}, weight_path)
    print(f"[{domain}] Saved weights to: {weight_path}")

    return model, dataset, train_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["amazon", "douban"], required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    task = task_directory(args.dataset, args.source, args.target)
    validate_task(task)
    for role, domain in [("source", args.source), ("target", args.target)]:
        pretrain_domain(domain, device, str(task / role),
                        str(task / "weights" / f"{role}.pt"),
                        dataset_name=args.dataset)
