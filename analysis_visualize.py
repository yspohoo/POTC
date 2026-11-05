from data_protocol import task_directory, validate_task, load_encoder_state
"""Visualize POTC embeddings and confidence-stratified ranking results."""

import os

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from models import LightGCN
from utils_edited import CDRLoader, Loader, overlap
from variable import *


SOURCE_BLUE = "#1683DC"
TARGET_PURPLE = "#E96DC4"


def _embedding_columns(frame, prefix):
    """Return dimension-sorted columns matching an embedding prefix."""
    return sorted((col for col in frame.columns if col.startswith(prefix)), key=lambda col: int(col.rsplit("_", 1)[1]))


def tsne2d(csv_path, prototype_csv_path=None, index=None, perplexity=30, n_iter=1000, output_path=None):
    """Plot paired user embeddings and optional mapped-source/target prototypes."""
    df = pd.read_csv(csv_path)
    mapped_source_cols = [col for col in df.columns if col.startswith("mapped_source_emb_")]
    source_cols = _embedding_columns(df, "source_emb_")
    target_cols = _embedding_columns(df, "target_emb_")
    source_is_mapped = bool(mapped_source_cols)
    source_cols = _embedding_columns(df, "mapped_source_emb_") if source_is_mapped else source_cols
    if not source_cols:
        raise ValueError("No mapped_source_emb_* or source_emb_* columns found.")
    if not target_cols:
        raise ValueError("No target_emb_* columns found.")

    source_emb = df[source_cols].to_numpy(dtype=np.float64)
    target_emb = df[target_cols].to_numpy(dtype=np.float64)
    if source_emb.shape != target_emb.shape:
        raise ValueError("Source and target embedding shapes do not match.")
    n = len(source_emb)
    if n < 2:
        raise ValueError("At least two users are required for t-SNE.")

    # Output Mean Cosine Similarity between source and target embeddings, excluding zero vectors
    source_norms = np.linalg.norm(source_emb, axis=1)
    target_norms = np.linalg.norm(target_emb, axis=1)
    valid = (source_norms > 0) & (target_norms > 0)
    if np.any(valid):
        cosine_similarities = np.sum(
            (source_emb[valid] / source_norms[valid, None])
            * (target_emb[valid] / target_norms[valid, None]),
            axis=1,
        )
        print(f"Mean source-target cosine similarity: {cosine_similarities.mean():.6f}")
    else:
        print("Mean source-target cosine similarity: undefined (all pairs contain zero vectors).")
    if np.any(~valid):
        print(f"Excluded {np.sum(~valid)} pairs containing zero vectors.")

    if index is None:
        selected_indices = []
    elif isinstance(index, (list, tuple, np.ndarray)):
        selected_indices = [int(i) for i in index]
    else:
        selected_indices = [int(index)]
    if len(selected_indices) > 3:
        raise ValueError("At most three selected indices are supported.")
    for i in selected_indices:
        if i < 0 or i >= n:
            raise IndexError(f"index must be within [0, {n - 1}], got {i}.")

    source_prototypes = target_prototypes = None
    if prototype_csv_path is not None:
        proto_df = pd.read_csv(prototype_csv_path)
        source_proto_cols = _embedding_columns(proto_df, "mapped_source_prototype_emb_")
        target_proto_cols = _embedding_columns(proto_df, "target_prototype_emb_")
        if not source_proto_cols and not target_proto_cols:
            raise ValueError("No mapped_source_prototype_emb_* or target_prototype_emb_* columns found.")
        if source_proto_cols:
            source_prototypes = proto_df[source_proto_cols].to_numpy(dtype=np.float64)
            if source_prototypes.shape[1] != source_emb.shape[1]:
                raise ValueError("Mapped source prototype and user embedding dimensions do not match.")
        if target_proto_cols:
            target_prototypes = proto_df[target_proto_cols].to_numpy(dtype=np.float64)
            if target_prototypes.shape[1] != target_emb.shape[1]:
                raise ValueError("Target prototype and user embedding dimensions do not match.")

    point_groups = [source_emb, target_emb]
    if source_prototypes is not None:
        point_groups.append(source_prototypes)
    if target_prototypes is not None:
        point_groups.append(target_prototypes)
    combined = StandardScaler().fit_transform(np.vstack(point_groups))
    effective_perplexity = min(float(perplexity), float(len(combined) - 1))
    projected = TSNE(n_components=2, perplexity=effective_perplexity, max_iter=n_iter,
                     init="pca", learning_rate="auto", random_state=42).fit_transform(combined)

    source_projected, target_projected = projected[:n], projected[n:2 * n]
    offset = 2 * n
    source_proto_projected = None
    target_proto_projected = None
    if source_prototypes is not None:
        count = len(source_prototypes)
        source_proto_projected = projected[offset:offset + count]
        offset += count
    if target_prototypes is not None:
        target_proto_projected = projected[offset:offset + len(target_prototypes)]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    source_label = "Mapped source overlap" if source_is_mapped else "Source overlap"
    ax.scatter(*source_projected.T, s=12, alpha=0.4, color=SOURCE_BLUE, edgecolors="none", label=source_label)
    ax.scatter(*target_projected.T, s=12, alpha=0.4, color=TARGET_PURPLE, edgecolors="none", label="Target overlap")
    if source_proto_projected is not None:
        ax.scatter(*source_proto_projected.T, s=90, marker="s", facecolors="none", edgecolors=SOURCE_BLUE,
                   linewidths=0.7, label="Mapped source prototypes", zorder=4)
    if target_proto_projected is not None:
        ax.scatter(*target_proto_projected.T, s=90, marker="s", facecolors="none", edgecolors=TARGET_PURPLE,
                   linewidths=0.7, label="Target prototypes", zorder=4)

    for selected_idx, suffix in zip(selected_indices, ("x", "y", "z")):
        sx, sy = source_projected[selected_idx]
        tx, ty = target_projected[selected_idx]
        first = selected_idx == selected_indices[0]
        ax.plot([sx, tx], [sy, ty], color="black", linestyle="--", linewidth=2.0,
                label="Corresponding pair" if first else None, zorder=5)
        ax.scatter([sx], [sy], s=40, color=SOURCE_BLUE, edgecolors="black", linewidths=0.5,
                   label=("Selected mapped source" if source_is_mapped else "Selected source") if first else None,
                   zorder=6)
        ax.scatter([tx], [ty], s=40, color=TARGET_PURPLE, edgecolors="black", linewidths=0.5,
                   label="Selected target" if first else None, zorder=6)
        source_text = rf"$\hat{{u}}_s^{{{suffix}}}$" if source_is_mapped else rf"$u_s^{{{suffix}}}$"
        text_style = dict(xytext=(0, 0), textcoords="offset points", fontsize=12, color="white", zorder=7,
                          path_effects=[PathEffects.withStroke(linewidth=1, foreground="black")])
        ax.annotate(source_text, (sx, sy), ha="right", va="bottom", **text_style)
        ax.annotate(rf"$u_t^{{{suffix}}}$", (tx, ty), ha="left", va="bottom", **text_style)

    ax.set(xticks=[], yticks=[], xlabel="", ylabel="", title="")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_alpha(1)
    ax.legend(frameon=True, framealpha=0.6, handletextpad=0.3, borderpad=0.25,
              labelspacing=0.25, borderaxespad=0.25, loc="upper left")
    if output_path is None:
        raise ValueError("output_path must be provided.")
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE visualization: {output_path}")
    return {"svg_path": output_path, "source_projected": source_projected, "target_projected": target_projected,
            "source_proto_projected": source_proto_projected, "target_proto_projected": target_proto_projected}


def plot_confidence_hr_ndcg_from_csv(csv_path, output_svg, table_svg=None, bins=20):
    """Plot and tabulate HR/NDCG across equal-count correspondence-confidence bins."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if bins < 2:
        raise ValueError("bins must be >= 2.")
    df = pd.read_csv(csv_path)
    if "correspondence_confidence" not in df.columns:
        raise ValueError("Missing required column: correspondence_confidence")
    hr_cols = [col for col in df.columns if col.startswith("potc_hr_at_")]
    ndcg_cols = [col for col in df.columns if col.startswith("potc_ndcg_at_")]
    if not hr_cols:
        raise ValueError("No potc_hr_at_K column found.")
    if not ndcg_cols:
        raise ValueError("No potc_ndcg_at_K column found.")
    hr_col, ndcg_col = hr_cols[0], ndcg_cols[0]
    hr_k, ndcg_k = hr_col.rsplit("_", 1)[-1], ndcg_col.rsplit("_", 1)[-1]
    if hr_k != ndcg_k:
        raise ValueError(f"HR and NDCG use different K values: {hr_k} vs {ndcg_k}")

    values = [pd.to_numeric(df[col], errors="raise").to_numpy(dtype=np.float64)
              for col in ("correspondence_confidence", hr_col, ndcg_col)]
    valid = np.logical_and.reduce([np.isfinite(array) for array in values])
    confidence, hr, ndcg = (array[valid] for array in values)
    if len(confidence) < bins:
        raise ValueError(f"Need at least {bins} valid users, but only {len(confidence)} were found.")

    order = np.argsort(confidence, kind="stable")
    confidence, hr, ndcg = (array[order] for array in (confidence, hr, ndcg))
    groups = np.array_split(np.arange(len(confidence)), bins)
    mean_conf = np.asarray([confidence[group].mean() for group in groups])
    mean_hr = np.asarray([hr[group].mean() for group in groups])
    mean_ndcg = np.asarray([ndcg[group].mean() for group in groups])
    counts = np.asarray([len(group) for group in groups], dtype=np.int64)
    x = np.arange(1, bins + 1)
    rho_hr, _ = spearmanr(mean_conf, mean_hr)
    rho_ndcg, _ = spearmanr(mean_conf, mean_ndcg)

    fig, ax_hr = plt.subplots(figsize=(7, 4.5))
    ax_ndcg = ax_hr.twinx()
    hr_bars = ax_hr.bar(x, mean_hr, width=0.72, alpha=1, label=rf"HR@{hr_k}", color=SOURCE_BLUE,
                         edgecolor="black", linewidth=0.6)
    ndcg_line, = ax_ndcg.plot(x, mean_ndcg, marker="o", markersize=9, linewidth=1.5,
                              color="#FF8316", label=rf"NDCG@{hr_k}")
    ax_hr.set_xlim(0.35, bins + 0.65)
    x_ticks = np.arange(1, bins + 1, 2)
    ax_hr.set_xticks(x_ticks)
    ax_hr.set_xticklabels([f"Q{i}" for i in x_ticks], fontsize=13)
    ax_hr.set_xlabel("Cross-domain correspondence confidence", fontsize=17, labelpad=2)

    def limits(data):
        low, high = float(data.min()), float(data.max())
        spread = high - low
        margin = max(abs(high) * 0.05, 0.01) if np.isclose(spread, 0.0) else 0.06 * spread
        return low - margin, high + margin

    hr_limits, ndcg_limits = limits(mean_hr), limits(mean_ndcg)
    ax_hr.set_ylim(*hr_limits)
    ax_ndcg.set_ylim(*ndcg_limits)
    hr_ticks, ndcg_ticks = np.linspace(*hr_limits, 5), np.linspace(*ndcg_limits, 5)
    ax_hr.set_yticks(hr_ticks)
    ax_ndcg.set_yticks(ndcg_ticks)
    ax_hr.set_yticklabels([f"{value:.3f}" for value in hr_ticks], fontsize=13,
                          rotation=90, va="center", ha="center")
    ax_ndcg.set_yticklabels([f"{value:.3f}" for value in ndcg_ticks], fontsize=13,
                            rotation=90, va="center", ha="center")
    ax_hr.tick_params(axis="y", pad=6)
    ax_ndcg.tick_params(axis="y", pad=6)
    ax_hr.set_ylabel(rf"HR@{hr_k}", fontsize=17, labelpad=2)
    ax_ndcg.set_ylabel(rf"NDCG@{hr_k}", fontsize=17, labelpad=2)
    ax_hr.spines["top"].set_visible(False)
    ax_ndcg.spines["top"].set_visible(False)
    ax_hr.legend([hr_bars, ndcg_line], [rf"HR@{hr_k}", rf"NDCG@{hr_k}"], loc="best", fontsize=17,
                 borderpad=0.2, labelspacing=0.25, handletextpad=0.35, handlelength=1.5,
                 columnspacing=0.5, framealpha=1)
    rho_text = rf"spearman-$\rho_{{HR}}$={rho_hr:.3f}" + "\n" + rf"spearman-$\rho_{{NDCG}}$={rho_ndcg:.3f}"
    ax_hr.text(0.022, 0.72, rho_text, transform=ax_hr.transAxes, fontsize=17, ha="left", va="center",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.2, edgecolor="black"))
    ax_hr.margins(x=0.0, y=0.0)
    ax_ndcg.margins(x=0.0, y=0.0)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.13, top=0.96)
    fig.savefig(output_svg, format="svg", bbox_inches="tight")
    plt.close(fig)

    if table_svg is None:
        table_svg = os.path.splitext(output_svg)[0] + "_table.svg"
    table_data = [[f"Q{i + 1}", f"{mean_conf[i]:.4f}", f"{counts[i]}", f"{mean_hr[i]:.4f}",
                   f"{mean_ndcg[i]:.4f}"] for i in range(bins)]
    columns = ["Bin", "Association", "N", rf"HR@{hr_k}", rf"NDCG@{hr_k}"]
    fig_table, ax_table = plt.subplots(figsize=(9.5, 6.0))
    ax_table.axis("off")
    table = ax_table.table(cellText=table_data, colLabels=columns, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.45)
    for cell in table.get_celld().values():
        cell.set_linewidth(0.6)
    fig_table.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    fig_table.savefig(table_svg, format="svg", bbox_inches="tight")
    plt.close(fig_table)


def pretrain_base(source_domain, target_domain, index_list, dataset="amazon"):
    """Export pretrained overlap-user embeddings and their t-SNE visualization."""
    device = "cpu"
    task = task_directory(dataset, source_domain, target_domain)
    validate_task(task)
    source_dataset = Loader(path=str(task / "source"), device=device)
    target_dataset = Loader(path=str(task / "target"), device=device)
    lightgcn_source = LightGCN(source_dataset, EMBBEDDING_DIM, NUM_LAYERS, device).to(device)
    load_encoder_state(lightgcn_source, task / "weights/source.pt", task / "source", device)
    lightgcn_target = LightGCN(target_dataset, EMBBEDDING_DIM, NUM_LAYERS, device).to(device)
    load_encoder_state(lightgcn_target, task / "weights/target.pt", task / "target", device)
    lightgcn_source.eval()
    lightgcn_target.eval()

    overlap_path = str(task.parent / "overlap" / f"{source_domain}-{target_domain}-overlap.txt")
    overlap_src2tgt = overlap(overlap_path)
    tgt2src = torch.zeros(target_dataset.n_user, dtype=torch.long, device=device)
    for source_id, target_id in overlap_src2tgt.items():
        tgt2src[target_id] = source_id

    cdr_dataset = CDRLoader(path=str(task))
    overlap_target_ids = {int(target_id) for target_id in overlap_src2tgt.values()}
    test_overlap_users = [int(uid) for uid in sorted(cdr_dataset.testDict) if int(uid) in overlap_target_ids]
    if not test_overlap_users:
        raise ValueError("No overlapping users were found in the CDR test set.")
    target_ids = torch.as_tensor(test_overlap_users, dtype=torch.long, device=device)
    source_ids = tgt2src[target_ids]

    with torch.no_grad():
        source_users, _ = lightgcn_source.computer()
        target_users, _ = lightgcn_target.computer()
        source_np = source_users[source_ids].cpu().numpy().astype(np.float64)
        target_np = target_users[target_ids].cpu().numpy().astype(np.float64)

    output_dir = os.path.join(visualize_path, f"{source_domain}-{target_domain}-base")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "pretrained_source_target_embeddings.csv")
    rows = []
    source_ids_np = source_ids.cpu().numpy()
    for i, (target_uid, source_uid) in enumerate(zip(test_overlap_users, source_ids_np)):
        row = {"target_user_id": int(target_uid), "source_user_id": int(source_uid)}
        row.update({f"source_emb_{d}": source_np[i, d] for d in range(source_np.shape[1])})
        row.update({f"target_emb_{d}": target_np[i, d] for d in range(target_np.shape[1])})
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    svg_path = os.path.join(output_dir, "pretrained_source_target_tsne.pdf")
    tsne2d(csv_path=csv_path, index=index_list, output_path=svg_path)
    print(f"Saved base embedding CSV: {csv_path}")
    print(f"Saved base t-SNE SVG: {svg_path}")
    return {"csv_path": csv_path, "svg_path": svg_path, "source_user_ids": source_ids_np,
            "target_user_ids": target_ids.cpu().numpy(), "source_embeddings": source_np, "target_embeddings": target_np}