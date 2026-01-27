#!/usr/bin/env python3
"""
Replot t-SNE rounds 0, 10, 20, 30 for experiment 59001 as one 4-panel paper figure.
- Same style per panel: Helpfulness vs Harmlessness only, prototypes, exclude outlier helpful client.
- No "t-SNE Dimension 1/2" axis labels.
- Four panels side by side: Round 0, 10, 20, 30.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

EXP_DIR = "/home/kjb/ppfl/exp/vplgp_ortho_hhst_n10_t59001/sub_exp_20260127143607"
ROUNDS = [0, 10, 20, 30]
K_NEAREST_FOR_PROTO = 50
COLOR_HARMLESS = "#C41E3A"
COLOR_HELPFUL = "#0066B2"


def load_and_filter(round_num):
    path = os.path.join(EXP_DIR, f"cross_client_z_tsne_round_{round_num}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    z_values = np.array(data["z_values"])
    z_2d = np.array(data["z_values_2d"])
    orth = np.array(data["orthogonal_labels"])
    client_labels = np.array(data["client_labels"])
    prototypes = np.array(data["orthogonal_prototypes"])

    # Prototype 2D
    proto_2d = np.zeros((len(prototypes), 2))
    for i, p in enumerate(prototypes):
        d = np.linalg.norm(z_values - p.reshape(1, -1), axis=1)
        idx = np.argsort(d)[:K_NEAREST_FOR_PROTO]
        proto_2d[i] = z_2d[idx].mean(axis=0)

    mask_harmless = orth == 0
    mask_helpful = orth == 1
    harmless_pts = z_2d[mask_harmless]
    harm_centroid = harmless_pts.mean(axis=0)

    helpful_client_ids = sorted(set(client_labels[mask_helpful]))
    if len(helpful_client_ids) >= 2:
        client_mean_dist_to_harm = {}
        for cid in helpful_client_ids:
            mc = (client_labels == cid) & mask_helpful
            pts_c = z_2d[mc]
            mean_d = np.linalg.norm(pts_c - harm_centroid, axis=1).mean()
            client_mean_dist_to_harm[cid] = mean_d
        excluded_client = min(helpful_client_ids, key=lambda c: client_mean_dist_to_harm[c])
        keep_helpful = mask_helpful & (client_labels != excluded_client)
        helpful_pts_filtered = z_2d[keep_helpful]
    else:
        helpful_pts_filtered = z_2d[mask_helpful]

    # Rotate so red prototype (Harmlessness) is below, blue (Helpfulness) above
    v = proto_2d[1] - proto_2d[0]  # red → blue
    theta = -np.arctan2(v[0], v[1])  # so v aligns with (0, +y)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    harmless_pts = (R @ harmless_pts.T).T
    helpful_pts_filtered = (R @ helpful_pts_filtered.T).T
    proto_2d = (R @ proto_2d.T).T
    # If red (proto_2d[0]) is still above blue (proto_2d[1]), rotate 90° clockwise
    if proto_2d[0, 1] > proto_2d[1, 1]:
        R90cw = np.array([[0.0, -1.0], [1.0, 0.0]])  # 90° clockwise
        harmless_pts = (R90cw @ harmless_pts.T).T
        helpful_pts_filtered = (R90cw @ helpful_pts_filtered.T).T
        proto_2d = (R90cw @ proto_2d.T).T

    return harmless_pts, helpful_pts_filtered, proto_2d, round_num


def main():
    all_data = [load_and_filter(r) for r in ROUNDS]

    # 4 panels: wider gaps so figure spreads out; larger figure
    fig_w, fig_h = 14.0, 6.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    n_panels = 4
    margin_left, margin_right = 0.06, 0.06
    margin_bottom = 0.14
    gap = 0.07  # larger gap so panels spread across the figure
    usable_w = 1.0 - margin_left - margin_right
    panel_w = (usable_w - (n_panels - 1) * gap) / n_panels
    panel_h = panel_w * fig_w / fig_h
    bottom = margin_bottom

    axes = []
    for i in range(n_panels):
        ax = fig.add_subplot(1, n_panels, i + 1)
        left = margin_left + i * (panel_w + gap)
        ax.set_position([left, bottom, panel_w, panel_h])
        axes.append(ax)

    for idx, r in enumerate(ROUNDS):
        out = all_data[idx]
        if out is None:
            axes[idx].set_visible(False)
            continue
        harmless_pts, helpful_pts_filtered, proto_2d, round_num = out
        ax = axes[idx]
        ax.set_facecolor("white")

        ax.scatter(
            harmless_pts[:, 0],
            harmless_pts[:, 1],
            c=COLOR_HARMLESS,
            label="Harmlessness" if idx == 0 else None,
            alpha=0.8,
            s=50,
            edgecolors="white",
            linewidths=0.6,
            zorder=2,
        )
        ax.scatter(
            helpful_pts_filtered[:, 0],
            helpful_pts_filtered[:, 1],
            c=COLOR_HELPFUL,
            label="Helpfulness" if idx == 0 else None,
            alpha=0.8,
            s=50,
            edgecolors="white",
            linewidths=0.6,
            zorder=2,
        )
        ax.scatter(
            proto_2d[0, 0],
            proto_2d[0, 1],
            marker="*",
            s=350,
            c=COLOR_HARMLESS,
            edgecolors="black",
            linewidths=1.5,
            label="Harmlessness prototype" if idx == 0 else None,
            zorder=10,
        )
        ax.scatter(
            proto_2d[1, 0],
            proto_2d[1, 1],
            marker="*",
            s=350,
            c=COLOR_HELPFUL,
            edgecolors="black",
            linewidths=1.5,
            label="Helpfulness prototype" if idx == 0 else None,
            zorder=10,
        )

        # t-SNE: x and y can have different scales (no need for equal aspect)
        ax.set_title(f"Round {round_num}", fontsize=20, fontweight="bold", pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.3, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=11, framealpha=0.95, bbox_to_anchor=(0.5, 0.02))

    out_path = os.path.join(EXP_DIR, "cross_client_z_tsne_rounds_0_10_20_30_paper.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved 4-panel paper figure: {out_path}")


if __name__ == "__main__":
    main()
