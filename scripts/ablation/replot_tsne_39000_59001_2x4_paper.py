#!/usr/bin/env python3
"""
Replot t-SNE: top row = 39000 (VPL, no prototype), bottom row = 59001 (VPL-GP-Ortho, with prototypes).
2x4 layout: Round 0, 10, 20, 30 each. No axis numbers, smaller gap. Legend from bottom row.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

EXP_DIR_39000 = "/home/kjb/ppfl/exp/vpl_hhst_n10_t39000/sub_exp_20260127143603"
EXP_DIR_59001 = "/home/kjb/ppfl/exp/vplgp_ortho_hhst_n10_t59001/sub_exp_20260127143607"
ROUNDS = [0, 10, 20, 30]
K_NEAREST_FOR_PROTO = 50
COLOR_HARMLESS = "#C41E3A"
COLOR_HELPFUL = "#0066B2"


def load_39000(round_num):
    """39000: no prototype, no rotation. Returns (harmless_pts, helpful_pts_filtered, None, round_num)."""
    path = os.path.join(EXP_DIR_39000, f"cross_client_z_tsne_round_{round_num}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    z_2d = np.array(data["z_values_2d"])
    orth = np.array(data["orthogonal_labels"])
    client_labels = np.array(data["client_labels"])
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
    return harmless_pts, helpful_pts_filtered, None, data["round_num"]


def load_59001(round_num):
    """59001: with prototypes, rotation. Returns (harmless_pts, helpful_pts_filtered, proto_2d, round_num)."""
    path = os.path.join(EXP_DIR_59001, f"cross_client_z_tsne_round_{round_num}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    z_values = np.array(data["z_values"])
    z_2d = np.array(data["z_values_2d"])
    orth = np.array(data["orthogonal_labels"])
    client_labels = np.array(data["client_labels"])
    prototypes = np.array(data["orthogonal_prototypes"])
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
    v = proto_2d[1] - proto_2d[0]
    theta = -np.arctan2(v[0], v[1])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    harmless_pts = (R @ harmless_pts.T).T
    helpful_pts_filtered = (R @ helpful_pts_filtered.T).T
    proto_2d = (R @ proto_2d.T).T
    if proto_2d[0, 1] > proto_2d[1, 1]:
        R90cw = np.array([[0.0, -1.0], [1.0, 0.0]])
        harmless_pts = (R90cw @ harmless_pts.T).T
        helpful_pts_filtered = (R90cw @ helpful_pts_filtered.T).T
        proto_2d = (R90cw @ proto_2d.T).T
    return harmless_pts, helpful_pts_filtered, proto_2d, data["round_num"]


def main():
    data_39000 = [load_39000(r) for r in ROUNDS]
    data_59001 = [load_59001(r) for r in ROUNDS]

    n_rows, n_cols = 2, 4
    fig_w, fig_h = 14.0, 6.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # Left margin for algorithm names (slightly reduced so closer to panels)
    margin_left = 0.14
    margin_right = 0.06
    margin_bottom = 0.14   # legend below; reduced to shrink legend–panel gap
    margin_top = 0.06
    gap_h, gap_v = 0.05, 0.04
    usable_w = 1.0 - margin_left - margin_right
    usable_h = 1.0 - margin_bottom - margin_top
    panel_w = (usable_w - (n_cols - 1) * gap_h) / n_cols
    panel_h = (usable_h - gap_v) / 2

    axes = []
    for row in range(n_rows):
        bottom = margin_bottom + (1 - row) * (panel_h + gap_v)
        for col in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            left = margin_left + col * (panel_w + gap_h)
            ax.set_position([left, bottom, panel_w, panel_h])
            axes.append(ax)

    dot_sz = 32  # smaller dots
    star_sz = 280  # smaller prototype stars

    def draw_panel(ax, harmless_pts, helpful_pts_filtered, proto_2d, round_num, show_round=True, add_legend_labels=False):
        ax.set_facecolor("white")
        ax.scatter(harmless_pts[:, 0], harmless_pts[:, 1], c=COLOR_HARMLESS,
                  label="Harmlessness" if add_legend_labels else None,
                  alpha=0.8, s=dot_sz, edgecolors="white", linewidths=0.5, zorder=2)
        ax.scatter(helpful_pts_filtered[:, 0], helpful_pts_filtered[:, 1], c=COLOR_HELPFUL,
                  label="Helpfulness" if add_legend_labels else None,
                  alpha=0.8, s=dot_sz, edgecolors="white", linewidths=0.5, zorder=2)
        if proto_2d is not None:
            ax.scatter(proto_2d[0, 0], proto_2d[0, 1], marker="*", s=star_sz, c=COLOR_HARMLESS,
                      edgecolors="black", linewidths=1.2,
                      label="Harmlessness prototype" if add_legend_labels else None, zorder=10)
            ax.scatter(proto_2d[1, 0], proto_2d[1, 1], marker="*", s=star_sz, c=COLOR_HELPFUL,
                      edgecolors="black", linewidths=1.2,
                      label="Helpfulness prototype" if add_legend_labels else None, zorder=10)
        if show_round:
            ax.set_title(f"Round {round_num}", fontsize=20, fontweight="bold", pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.3, linestyle="--")

    for col in range(n_cols):
        out = data_39000[col]
        if out is not None:
            draw_panel(axes[col], out[0], out[1], out[2], out[3], show_round=True, add_legend_labels=False)
        else:
            axes[col].set_visible(False)

    for col in range(n_cols):
        out = data_59001[col]
        if out is not None:
            draw_panel(axes[4 + col], out[0], out[1], out[2], out[3], show_round=False, add_legend_labels=(col == 0))
        else:
            axes[4 + col].set_visible(False)

    # Row labels in left band (center of 0..margin_left) so they don’t overlap panels
    label_x = margin_left * 0.5
    y_top = margin_bottom + panel_h + gap_v + panel_h / 2
    y_bot = margin_bottom + panel_h / 2
    fig.text(label_x, y_top, "FedVPL", fontsize=14, fontweight="bold", va="center", ha="center")
    fig.text(label_x, y_bot, "FedVPA-GP", fontsize=14, fontweight="bold", va="center", ha="center")

    # Legend below the figure, a bit closer to bottom row
    handles, labels = axes[4].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=11, framealpha=0.95)

    out_path = os.path.join(EXP_DIR_59001, "cross_client_z_tsne_39000_59001_2x4_paper.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved 2x4 paper figure: {out_path}")


if __name__ == "__main__":
    main()
