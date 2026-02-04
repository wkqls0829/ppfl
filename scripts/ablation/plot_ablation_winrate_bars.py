#!/usr/bin/env python3
"""
Ablation bar graph for paper: Helpfulness + Harmlessness win rate in one figure.
Left: Helpfulness (4 bars). Right: Harmlessness (4 bars).
Compares: FedVPL, FedVPL+Ortho, FedVPL+GB Prior, FedVPA-GP.

수치는 아래 VALUES 딕셔너리에서 바꾸면 됩니다.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- 수치 (Qwen) ---
# 순서: FedVPL, FedVPL+Ortho, FedVPL+GB Prior, FedVPA-GP
VALUES_QWEN = {
    "helpfulness_winrate": [63.24, 64.1, 64.5, 66.45],
    "harmlessness_winrate": [84.56, 85.7, 86.0, 89.21],
}
# --- 수치 (Gemma) ---
VALUES_GEMMA = {
    "helpfulness_winrate": [66.82, 67.8, 68.2, 75.21],
    "harmlessness_winrate": [89.15, 90.2, 90.8, 96.34],
}
STD = {"helpfulness_winrate": None, "harmlessness_winrate": None}

METHOD_LABELS = ["FedVPL", "FedVPL+Ortho", "FedVPL+GB Prior", "FedVPA-GP (ours)"]

# 출력 경로
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# 논문용 스타일 (좌우로 줄여서 두 피겨 나란히 배치, 글자 확대)
FIG_WIDTH = 4.2
FIG_HEIGHT = 3.0
FONTSIZE = 12       # 축·라벨·눈금
BAR_LABEL_FONTSIZE = 9   # 막대 위 숫자 (고정)
TITLE_FONTSIZE = 12
Y_LABEL = "Win rate (%)"
Y_RANGE = (0, 100)
BAR_COLORS = ["#4d7ea8", "#7ba06b", "#c9a227", "#b64f70"]
GAP = 1.5   # 두 그룹 사이 간격


def plot_combined(values, std, output_basename, ylim=(55, 95), show_legend=False):
    n = len(METHOD_LABELS)
    x_left = np.arange(n)
    x_right = np.arange(n) + n + GAP
    width = 1.0

    helpful = np.array(values["helpfulness_winrate"])
    harm = np.array(values["harmlessness_winrate"])
    std_h = std.get("helpfulness_winrate")
    std_a = std.get("harmlessness_winrate")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    bars_left = ax.bar(x_left, harm, width, color=BAR_COLORS, edgecolor="black", linewidth=0.5)
    bars_right = ax.bar(x_right, helpful, width, color=BAR_COLORS, edgecolor="black", linewidth=0.5)
    if std_a is not None and len(std_a) == n:
        ax.errorbar(x_left, harm, yerr=std_a, fmt="none", color="black", capsize=2, capthick=0.6)
    if std_h is not None and len(std_h) == n:
        ax.errorbar(x_right, helpful, yerr=std_h, fmt="none", color="black", capsize=2, capthick=0.6)

    ax.set_ylabel(Y_LABEL, fontsize=FONTSIZE)
    ax.set_ylim(ylim[0], ylim[1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks([(x_left[0] + x_left[-1]) / 2, (x_right[0] + x_right[-1]) / 2])
    ax.set_xticklabels(["Harmlessness", "Helpfulness"], fontsize=FONTSIZE)
    ax.tick_params(axis="both", labelsize=FONTSIZE)
    if show_legend:
        legend_handles = [mpatches.Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=lab) for c, lab in zip(BAR_COLORS, METHOD_LABELS)]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=10, frameon=True, framealpha=0.95)

    for bar, v in zip(bars_left, harm):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{v:.2f}", ha="center", va="bottom", fontsize=BAR_LABEL_FONTSIZE)
    for bar, v in zip(bars_right, helpful):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{v:.2f}", ha="center", va="bottom", fontsize=BAR_LABEL_FONTSIZE)

    fig.tight_layout()
    path_pdf = os.path.join(OUT_DIR, f"{output_basename}.pdf")
    path_png = os.path.join(OUT_DIR, f"{output_basename}.png")
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path_pdf} and {path_png}")


if __name__ == "__main__":
    plot_combined(VALUES_QWEN, STD, "ablation_winrate_combined", ylim=(55, 95), show_legend=False)
    plot_combined(VALUES_GEMMA, STD, "ablation_winrate_combined_gemma", ylim=(55, 98), show_legend=True)
