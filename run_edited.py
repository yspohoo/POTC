from data_protocol import task_directory, validate_task, load_encoder_state
from models import *
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from utils_edited import *
import torch.optim as optim
import tempfile
from pathlib import Path
import ray
from ray import tune
from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune import JupyterNotebookReporter
import time


def get_cdr_save_dir(dataset, source_domain, target_domain):
    return str(task_directory(dataset, source_domain, target_domain))


def get_overlap_path(dataset, source_domain, target_domain):
    root = task_directory(dataset, source_domain, target_domain).parent
    return str(root / "overlap" / f"{source_domain}-{target_domain}-overlap.txt")


def evaluate(model, cdr_dataset, tgt2src_tensor, split_dict, device, k=10, test_batch_size=100, phase="valid"):
    """
    Ranks target-domain items for overlap users via their mapped source
    representation (POTC.forward) and reports HR@k / Recall@k / NDCG@k
    against `split_dict` (cdr_dataset.validDict while tuning, cdr_dataset.testDict
    for the final held-out report).
    """
    if phase not in {"valid", "test"}:
        raise ValueError("phase must be valid or test")
    if len(split_dict) == 0:
        return 0.0, 0.0, 0.0

    model.eval()
    users = list(split_dict.keys())
    hits, recalls, ndcgs = [], [], []

    with torch.no_grad():
        for start in range(0, len(users), test_batch_size):
            batch_users = users[start:start + test_batch_size]
            ground_truth = [split_dict[u] for u in batch_users]

            batch_users_t = torch.LongTensor(batch_users).to(device)
            source_uid = tgt2src_tensor[batch_users_t]
            hat_u, _, all_items = model(source_uid, batch_users_t)
            ratings = torch.matmul(hat_u, all_items.t())

            # Validation hides training positives; testing also hides validation positives.
            for i, u in enumerate(batch_users):
                seen = set(cdr_dataset.trainAllPos.get(u, []))
                if phase == "test":
                    seen.update(cdr_dataset.validDict.get(u, []))
                if seen:
                    ratings[i, list(seen)] = -float('inf')

            _, top_items = torch.topk(ratings, k=k)
            top_items = top_items.cpu().numpy()

            r = getLabel(ground_truth, top_items)
            hits.append(HR_ATk(ground_truth, r, k))
            recalls.append(RecallPrecision_ATk(ground_truth, r, k)['recall'])
            ndcgs.append(NDCGatK_r(ground_truth, r, k))

    n = len(users)
    return float(np.sum(hits) / n), float(np.sum(recalls) / n), float(np.sum(ndcgs) / n)


def train(config):
    device = config["device"]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    pre_s_weight_save = config["pre_s_weight_save"]
    pre_t_weight_save = config["pre_t_weight_save"]
    source_domain = config["source_domain"]
    target_domain = config["target_domain"]
    cdr_save_dir = config["cdr_save_dir"]
    overlap_path = config["overlap_path"]
    weight_name = config["weight_name"]

    # ----- Frozen domain encoders (Stage 1: pretraining, done by pretrain.py) -----
    source_dataset = Loader(path=config["source_graph"], device=device)
    target_dataset = Loader(path=config["target_graph"], device=device)

    lightgcn_source = LightGCN(source_dataset, EMBBEDDING_DIM, NUM_LAYERS, device).to(device)
    load_encoder_state(lightgcn_source, pre_s_weight_save, config["source_graph"], device)

    lightgcn_target = LightGCN(target_dataset, EMBBEDDING_DIM, NUM_LAYERS, device).to(device)
    load_encoder_state(lightgcn_target, pre_t_weight_save, config["target_graph"], device)

    # ----- Overlap correspondence + ranking-CDR train/valid/test -----
    overlap_src2tgt = overlap(overlap_path)
    tgt2src_tensor = torch.zeros(target_dataset.n_user, dtype=torch.long, device=device)
    for s_id, t_id in overlap_src2tgt.items():
        tgt2src_tensor[t_id] = s_id

    cdr_dataset = CDRLoader(path=cdr_save_dir)

    # ----- POTC: Stage 2 (prototype prep, done inside __init__) + Stage 3 (mapping) -----
    model = POTC(
        sencoder=lightgcn_source, tencoder=lightgcn_target,
        cluster_size=config["cluster_size"], lambda_e=LAMBDA_E, maxiter=MAXITER,
        rho=config["rho"], overlap_src2tgt=overlap_src2tgt, device=device,
        skip_ot=config["skip_ot"],
    ).to(device)

    # The optimizer's weight decay supplies the L2 regularization on E and F.
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"],
    )

    best_recall = -float("inf")
    stale_epochs = 0
    if not cdr_dataset.validDict:
        raise ValueError("Validation interactions are required for early stopping")
    for it in range(ITERATIONS):
        model.train()

        # BPR triplets from overlapping users' target training histories
        S = UniformSample_cdr(cdr_dataset, target_dataset.m_item)
        users_t = torch.LongTensor(S[:, 0]).to(device)
        pos_t = torch.LongTensor(S[:, 1]).to(device)
        neg_t = torch.LongTensor(S[:, 2]).to(device)
        users_t, pos_t, neg_t = shuffle(users_t, pos_t, neg_t)

        running_loss = 0.0
        running_bpr_loss = 0.0
        running_ot_loss = 0.0
        running_align_loss = 0.0
        epoch_steps = 0
        for batch_users, batch_pos, batch_neg in pretrain_minibatch(
            users_t, pos_t, neg_t, batch_size=BATCH_SIZE
        ):
            source_uid = tgt2src_tensor[batch_users]
            hat_u, (ot_loss, align_loss), item_emb = model(source_uid, batch_users)

            pos_scores = torch.sum(hat_u * item_emb[batch_pos], dim=1)
            neg_scores = torch.sum(hat_u * item_emb[batch_neg], dim=1)
            bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

            total_loss = (
                bpr_loss
                + config["lambda_ot"] * ot_loss
                + config["lambda_align"] * align_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_bpr_loss += bpr_loss.item()
            running_ot_loss += ot_loss.item()
            running_align_loss += align_loss.item()
            epoch_steps += 1

        newhr, newrecall, newndcg = evaluate(
            model, cdr_dataset, tgt2src_tensor, cdr_dataset.validDict, device, k=10,
        )
        if not all(np.isfinite(v) for v in (newhr, newrecall, newndcg)):
            raise ValueError("Validation metrics are not finite")
        if newrecall > best_recall + 0.0:
            best_recall = newrecall
            stale_epochs = 0
        else:
            stale_epochs += 1

        # Save checkpoint and report metrics
        checkpoint_data = {
            "epoch": it + 1,
            "model": model,
            # "weight_name": model.state_dict(),
            # "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
            torch.save(checkpoint_data, checkpoint_path)
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            tune.report(
                {
                    "train_loss": running_loss / epoch_steps,
                    "iter": it + 1,
                    "bpr_loss": running_bpr_loss / epoch_steps,
                    "ot_loss": running_ot_loss / epoch_steps,
                    "align_loss": running_align_loss / epoch_steps,
                    "HR@10": newhr,
                    "Recall@10": newrecall,
                    "NDCG@10": newndcg,
                },
                checkpoint=checkpoint,
            )
        if stale_epochs >= MAPPING_PATIENCE:
            break


def main(source_domain, target_domain, lrlist, rho, clusterlist,
         ot_weightlist, align_weightlist, device, dataset="amazon",
         gpus_per_trial=1, cpus_per_trial=1, skip_ot=False):
    dataset = dataset.lower()
    if dataset not in {"amazon", "douban"}:
        raise ValueError(f"Unsupported dataset: {dataset}. Expected 'amazon' or 'douban'.")

    cdr_save_dir = get_cdr_save_dir(dataset, source_domain, target_domain)
    validate_task(cdr_save_dir)
    pre_s_weight_save = os.path.join(cdr_save_dir, "weights", "source.pt")
    pre_t_weight_save = os.path.join(cdr_save_dir, "weights", "target.pt")
    overlap_path = get_overlap_path(dataset, source_domain, target_domain)

    print(f"Starting hyperparameter tuning on dataset={dataset}, "
          f"source={source_domain}, target={target_domain}.")
    ray.init(include_dashboard=False)

    config = {
        "lr": tune.grid_search(lrlist),
        "cluster_size": tune.grid_search(clusterlist),
        "rho": rho,
        "lambda_ot": tune.grid_search(ot_weightlist),
        "lambda_align": tune.grid_search(align_weightlist),
        "skip_ot": skip_ot,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "pre_s_weight_save": pre_s_weight_save,
        "pre_t_weight_save": pre_t_weight_save,
        "cdr_save_dir": cdr_save_dir,
        "overlap_path": overlap_path,
        "weight_name": "main_",
        "device": device,
        "source_graph": os.path.join(cdr_save_dir, "source"),
        "target_graph": os.path.join(cdr_save_dir, "target"),
        "weight_decay": {'amazon': 1e-06, 'douban': 1e-07}[dataset],
    }
    scheduler = ASHAScheduler(
        time_attr="iter",
        max_t=ITERATIONS,
        grace_period=ASHA_GRACE_PERIOD,
        reduction_factor=2,
    )
    parameter_columns=["lr", "cluster_size", "lambda_ot", "lambda_align", "rho", "skip_ot"]
    metric_columns=["train_loss", "bpr_loss", "ot_loss", "align_loss", "HR@10", "Recall@10", "NDCG@10", "iter"]
    reporter = JupyterNotebookReporter(
        max_progress_rows=100,
        parameter_columns=parameter_columns,
        metric_columns=metric_columns,
    )
    tuner = tune.Tuner(
        tune.with_resources(
            train,
            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        ),
        run_config=tune.RunConfig(
            progress_reporter=reporter,
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=1,                       # only keep the best checkpoint
                checkpoint_score_attribute="HR@10",  # ranking quality, not raw loss
                checkpoint_score_order='max',
            ),
        ),
        tune_config=tune.TuneConfig(
            metric='HR@10',
            mode='max',
            scheduler=scheduler,
            num_samples=1,
        ),
        param_space=config,
    )
    results = tuner.fit()
    best_result = results.get_best_result('HR@10', 'max', scope="all")
    best_checkpoint, best_metrics = max(best_result.best_checkpoints, key=lambda entry: entry[1]['HR@10'])
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial HR@10: {best_metrics['HR@10']}")
    print(f"Best trial Recall@10: {best_metrics['Recall@10']}")
    print(f"Best trial NDCG@10: {best_metrics['NDCG@10']}")

    # get trained model
    with best_checkpoint.as_directory() as checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
        checkpoint_data = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        best_model = checkpoint_data["model"]


    # Evaluation
    start_test = time.time()
    target_dataset = Loader(path=os.path.join(cdr_save_dir, "target"), device=device)
    overlap_src2tgt = overlap(overlap_path)
    tgt2src_tensor = torch.zeros(target_dataset.n_user, dtype=torch.long, device=device)
    for s_id, t_id in overlap_src2tgt.items():
        tgt2src_tensor[t_id] = s_id
    cdr_dataset = CDRLoader(path=cdr_save_dir)
    newhr, newrecall, newndcg = evaluate(
        best_model, cdr_dataset, tgt2src_tensor, cdr_dataset.testDict, device, k=10, phase="test",
    )
    print(f"Evaluation metrics: HR@10={newhr:.4f}, Recall@10={newrecall:.4f}, NDCG@10={newndcg:.4f}")
    print(f"Elapsed test time: {time.time() - start_test:.4f}s")

    # run_transport_analysis(best_model, filename=f"transport_analysis_{source_domain}-{target_domain}_rho-{rho}.svg")

    # from utils_edited import _model_run_tag
    # # ---- save the trial table as CSV ----
    # out_dir = os.path.join(
    #     visualize_path,
    #     _model_run_tag(best_model, source_domain, target_domain),
    # )
    # os.makedirs(out_dir, exist_ok=True)
    # rows = []
    # for r in results:
    #     row = {}
    #     # trial identifier (falls back to the folder name if not in metrics)
    #     row["trial_id"] = r.metrics.get("trial_id", os.path.basename(r.path or ""))
    #     # hyperparameters -> from r.config
    #     for c in parameter_columns:
    #         row[c] = r.config.get(c)
    #     # metrics -> from r.metrics
    #     for c in metric_columns:
    #         row[c] = r.metrics.get(c)
    #     rows.append(row)
    # df = pd.DataFrame(rows)
    # output_path = os.path.join(out_dir, "raytune_results.csv")
    # df.to_csv(output_path, index=False)
    # print(f"Saved Ray Tune results to {output_path}")

    # overlap_src2tgt = overlap(overlap_path)
    # target_dataset = Loader(path=os.path.join(cdr_save_dir, "target"), device=device)
    # tgt2src_tensor = torch.zeros(target_dataset.n_user, dtype=torch.long, device=device)
    # for s_id, t_id in overlap_src2tgt.items():
    #     tgt2src_tensor[t_id] = s_id
    # cdr_dataset = CDRLoader(path=cdr_save_dir)
    # run_correspondence_metric_analysis(
    #     source_domain,
    #     target_domain,
    #     best_model,
    #     cdr_dataset,
    #     tgt2src_tensor,
    #     k=10,
    # )
    # save all hyperparameter & parameter of model
    # best_checkpoint = best_result.checkpoint
    # with best_checkpoint.as_directory() as checkpoint_dir:
    #     checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
    #     best_checkpoint_data = torch.load(checkpoint_path)
        # torch.save(best_checkpoint, weight_save + "checkpoint_" + config["weight_name"])
        # torch.save(best_checkpoint_data[config["weight_name"]], weight_save + config["weight_name"])
    return best_result



# ==================== End POTC Analysis Notebook Helpers ====================
# ==================== Analysis wrappers ====================
# Existing training/evaluation code above is intentionally unchanged.
# These wrappers simply expose the standalone analysis functions from utils.py.
# =================================================================

def run_transport_analysis(model, filename=None):
    """Run Experiment 1 for the CURRENT trained rho configuration."""
    return save_potc_transport_analysis_svg(model, filename=filename)


def run_correspondence_metric_analysis(
    source_domain,
    target_domain,
    model,
    cdr_dataset,
    tgt2src_tensor,
    k=10,
    test_batch_size=100,
    filename_2=None,
    filename_3=None
):
    """Run Experiment 2 + 3."""
    result_3 = save_mapped_source_target_embeddings_csv(
        source_domain,
        target_domain,
        model,
        cdr_dataset,
        tgt2src_tensor,
        filename_3,
    )
    result_2 = save_correspondence_metric_analysis_csv(
        source_domain=source_domain,
        target_domain=target_domain,
        model=model,
        cdr_dataset=cdr_dataset,
        tgt2src_tensor=tgt2src_tensor,
        k=k,
        test_batch_size=test_batch_size,
        filename=filename_2,
    )
    return result_2, result_3
