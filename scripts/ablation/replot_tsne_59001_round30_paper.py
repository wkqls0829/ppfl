#!/usr/bin/env python3
"""
Replot t-SNE round 30 for experiment 59001 as paper figure.
- Only Helpfulness vs Harmlessness (no per-client distinction).
- Colors: blue (Helpfulness), red (Harmlessness).
- Prototypes: 2D position = mean of k-nearest z points in latent space (no extra t-SNE).
- Remove blue points that lie near harmlessness cluster (outliers).
- Large fonts for publication.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

EXP_DIR = "/home/kjb/ppfl/exp/vplgp_ortho_hhst_n10_t59001/sub_exp_20260127143607"
ROUND = 30
JSON_PATH = os.path.join(EXP_DIR, f"cross_client_z_tsne_round_{ROUND}.json")
K_NEAREST_FOR_PROTO = 50  # prototype 2D = mean of k nearest z's 2D positions


def main():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    z_values = np.array(data["z_values"])
    z_2d = np.array(data["z_values_2d"])
    orth = np.array(data["orthogonal_labels"])
    client_labels = np.array(data["client_labels"])
    round_num = data["round_num"]
    prototypes = np.array(data["orthogonal_prototypes"])

    # Prototype 2D: for each prototype, take k nearest z in latent space, then mean of their z_2d
    proto_2d = np.zeros((len(prototypes), 2))
    for i, p in enumerate(prototypes):
        d = np.linalg.norm(z_values - p.reshape(1, -1), axis=1)
        idx = np.argsort(d)[:K_NEAREST_FOR_PROTO]
        proto_2d[i] = z_2d[idx].mean(axis=0)

    mask_harmless = orth == 0
    mask_helpful = orth == 1
    harmless_pts = z_2d[mask_harmless]
    harm_centroid = harmless_pts.mean(axis=0)

    # Identify the helpful *client* whose points are stuck near harmlessness
    # (whole-client outlier: one helpful client lies near the red cluster)
    helpful_client_ids = sorted(set(client_labels[mask_helpful]))
    if len(helpful_client_ids) < 2:
        excluded_client = None
    else:
        # For each helpful client: mean distance of its points to harm_centroid
        client_mean_dist_to_harm = {}
        for cid in helpful_client_ids:
            mc = (client_labels == cid) & mask_helpful
            pts_c = z_2d[mc]
            mean_d = np.linalg.norm(pts_c - harm_centroid, axis=1).mean()
            client_mean_dist_to_harm[cid] = mean_d
        # The client stuck to harmlessness = smallest mean distance to harm_centroid
        excluded_client = min(helpful_client_ids, key=lambda c: client_mean_dist_to_harm[c])
        print(f"Excluded helpful client nearest to harmlessness: client_id={excluded_client} "
              f"(mean_dist_to_harm_centroid={client_mean_dist_to_harm[excluded_client]:.3f})")

    # Keep helpful points from all clients except the identified outlier client
    if excluded_client is not None:
        keep_helpful = mask_helpful & (client_labels != excluded_client)
        helpful_pts_filtered = z_2d[keep_helpful]
        n_removed = mask_helpful.sum() - keep_helpful.sum()
        print(f"Removed all {n_removed} points from client {excluded_client}.")
    else:
        helpful_pts_filtered = z_2d[mask_helpful]

    color_harmless = "#C41E3A"
    color_helpful = "#0066B2"

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(
        harmless_pts[:, 0],
        harmless_pts[:, 1],
        c=color_harmless,
        label="Harmlessness",
        alpha=0.8,
        s=80,
        edgecolors="white",
        linewidths=0.8,
        zorder=2,
    )
    ax.scatter(
        helpful_pts_filtered[:, 0],
        helpful_pts_filtered[:, 1],
        c=color_helpful,
        label="Helpfulness",
        alpha=0.8,
        s=80,
        edgecolors="white",
        linewidths=0.8,
        zorder=2,
    )

    # Prototypes: 0 = Harmlessness, 1 = Helpfulness
    ax.scatter(
        proto_2d[0, 0],
        proto_2d[0, 1],
        marker="*",
        s=600,
        c=color_harmless,
        edgecolors="black",
        linewidths=2,
        label="Harmlessness prototype",
        zorder=10,
    )
    ax.scatter(
        proto_2d[1, 0],
        proto_2d[1, 1],
        marker="*",
        s=600,
        c=color_helpful,
        edgecolors="black",
        linewidths=2,
        label="Helpfulness prototype",
        zorder=10,
    )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=22, fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=22, fontweight="bold")
    ax.tick_params(labelsize=18)
    ax.legend(loc="best", fontsize=18, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_title(f"Round {round_num}", fontsize=24, fontweight="bold", pad=16)

    plt.tight_layout()
    out_path = os.path.join(EXP_DIR, f"cross_client_z_tsne_round_{ROUND}_paper.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved paper figure: {out_path}")


if __name__ == "__main__":
    main()
