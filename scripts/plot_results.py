"""Figures for the GSM8K fine-tuning writeup.

Every number here is a measured lm-eval strict-match (or flexible-extract) result
from the October 2026 Overmind challenge. Validation is the 500-row holdout
carved from the GSM8K train split. The test set is the official 1,319-question
split and was scored twice, on two checkpoints chosen before each run.

Sources: evals/*.json and autoresearch/results.tsv in the lab notebook.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

INK = "#1c1917"
MUTED = "#78716c"
GRID = "#e7e5e4"
BLUE = "#1d4ed8"
TEAL = "#0f766e"
AMBER = "#b45309"
STONE = "#a8a29e"
GREEN = "#047857"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.edgecolor": "#d6d3d1",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 160,
})


def clean(ax, grid="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid, color=GRID, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print(name)


# Morning trainer: step-1 loss about 0.92. Evening trainer: step-1 loss about 0.50.
# Same nominal learning rate is not the same update across that shift.
MORNING_LR = [
    (5e-6, 91.0),
    (2e-5, 91.4),
    (4e-5, 91.6),
    (1e-4, 82.6),
]
EVENING_LR = [
    (5e-6, 91.0),
    (1e-5, 90.2),
    (2e-5, 87.2),
    (6e-5, 80.2),
    (8e-5, 78.6),
]

# Same Qwen3.5-4B pool, evening trainer. Dose proxy is learning rate times optimizer steps.
DOSE = [
    ("5e-6 · 26 steps", 5e-6 * 26, 91.0),
    ("1e-5 · 26 steps", 1e-5 * 26, 90.2),
    ("2e-5 · 26 steps", 2e-5 * 26, 87.2),
    ("2e-5 · 82 steps\n(batch 4)", 2e-5 * 82, 81.2),
    ("2e-5 · 328 steps\n(batch 1)", 2e-5 * 328, 84.0),
]

# Every Qwen3.5-4B run with a 500-row validation score.
# (label, strict %, series)  series: morning, evening, selected
RUNS_4B = [
    ("LoRA 4e-5, morning", 91.6, "morning"),
    ("LoRA 2e-5, morning", 91.4, "morning"),
    ("LoRA 2e-5, QA format (submitted)", 91.4, "selected"),
    ("LoRA 5e-6, morning", 91.0, "morning"),
    ("LoRA 5e-6, evening", 91.0, "evening"),
    ("Hard examples only, 5e-6", 90.6, "evening"),
    ("LoRA 1e-5, evening", 90.2, "evening"),
    ("Hard examples upsampled 4×", 90.2, "evening"),
    ("Full fine-tune, 2e-6", 89.2, "morning"),
    ("Control, shuffled order", 87.4, "evening"),
    ("Control, evening LR 2e-5", 87.2, "evening"),
    ("Long answers, 3 epochs", 84.4, "evening"),
    ("Long answers, 1 epoch", 84.0, "evening"),
    ("Batch size 1 (328 steps)", 84.0, "evening"),
    ("Five-shot conversations", 83.6, "evening"),
    ("Grouped few-shot, LR 5e-5", 83.2, "evening"),
    ("LoRA 1e-4 (first run)", 82.6, "morning"),
    ("Decimal-money, LR 4e-5, 2 epochs", 82.8, "evening"),
    ("Full fine-tune, 5e-6", 82.0, "morning"),
    ("Decimal-money, LR 4e-5, 1 epoch", 81.8, "evening"),
    ("Grouped few-shot, LR 2e-5", 81.6, "evening"),
    ("Most calculator-step rows", 81.4, "evening"),
    ("Batch size 4 (82 steps)", 81.2, "evening"),
    ("Decimal-money, LR 2e-5, 1 epoch", 80.8, "evening"),
    ("LoRA 6e-5, evening", 80.2, "evening"),
    ("Decimal-money, LR 2e-5, 2 epochs", 80.0, "evening"),
    ("LoRA 8e-5, evening", 78.6, "evening"),
]

# Best 500-row validation per base model. Score uses the stored parameter count.
CATALOG = [
    ("LFM2.5-230M", 0.233, 32.6, 32.37),
    ("LFM2.5-350M", 0.361, 29.4, 29.04),
    ("Qwen3.5-0.8B", 0.882, 41.2, 40.32),
    ("LFM2.5-1.2B", 1.183, 75.2, 74.02),
    ("Llama-3.2-1B", 1.247, 52.8, 51.55),
    ("Qwen3.5-2B", 2.290, 65.6, 63.31),
    ("Llama-3.2-3B", 3.237, 80.0, 76.76),
    ("Qwen3.5-4B", 4.681, 91.4, 86.72),
    ("Gemma-4-E2B", 5.142, 62.0, 56.86),
    ("Gemma-4-E4B", 8.032, 76.2, 68.17),
    ("Qwen3-8B", 8.234, 83.8, 75.57),
    ("Qwen3.5-9B", 9.682, 92.0, 82.32),
]

# Official test misses of the submitted checkpoint. 1204 correct + these = 1319.
ERRORS = [
    ("Wrong number", 84),
    ("Gold elsewhere\nin the trace", 15),
    ("No #### line", 8),
    ("#### 75.00\nvs gold 75", 8),
]

BOARD = [
    ("Aman", 89.95),
    ("Cursive", 88.91),
    ("Daniel", 88.09),
    ("Martin", 87.84),
    ("Chinese", 87.34),
    ("Nodolas", 87.20),
    ("Shiven Tripathi", 87.07),
    ("Kartik", 86.67),
    ("Seif", 86.45),
    ("Ayush Nigade", 86.24),
]


def lr_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    panels = [
        (axes[0], MORNING_LR, "Morning trainer", "Step-1 loss ≈ 0.92"),
        (axes[1], EVENING_LR, "Evening trainer", "Step-1 loss ≈ 0.50"),
    ]
    for ax, pts, title, note in panels:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=BLUE, marker="o", markersize=7, linewidth=1.8, zorder=3)
        for x, y in pts:
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, color=INK)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Learning rate")
        ax.text(0.02, 0.04, note, transform=ax.transAxes, color=MUTED, fontsize=9)
        clean(ax, "y")
        ax.set_ylim(74, 96)
        ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    axes[0].set_ylabel("Validation strict-match (%)")
    fig.suptitle("Qwen3.5-4B LoRA: validation accuracy against learning rate", fontsize=14, y=1.02)
    fig.text(0.5, -0.04, "500-question holdout. One epoch, rank 16, all linear modules. The two panels are not one curve: the trainer’s step-1 loss changed between them.",
             ha="center", color=MUTED, fontsize=9)
    save(fig, "lr_sweep.png")


def dose():
    order = sorted(DOSE, key=lambda d: d[1])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = list(range(len(order)))
    ys = [d[2] for d in order]
    ax.plot(xs, ys, color=TEAL, marker="o", markersize=8, linewidth=1.8, zorder=3)
    for x, (_, _, y) in zip(xs, order):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    ax.set_xticks(xs, [d[0].replace("\n", " ") for d in order])
    ax.tick_params(axis="x", labelrotation=18)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_ylim(78, 96)
    ax.set_ylabel("Validation strict-match (%)")
    ax.set_xlabel("Left to right, learning rate × steps gets larger")
    ax.set_title("On the evening trainer, a larger update scored worse")
    clean(ax, "y")
    fig.tight_layout()
    fig.text(0.01, -0.02, "Same 6,473-row pool and Qwen3.5-4B. Smaller batches raised the step count. Batch 1 is the point that breaks the line.",
             color=MUTED, fontsize=9)
    save(fig, "dose.png")


def runs_4b():
    rows = sorted(RUNS_4B, key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9.2, 8.2))
    colors = {"morning": BLUE, "evening": STONE, "selected": GREEN}
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows], color=[colors[r[2]] for r in rows], height=0.72, zorder=3)
    ax.set_yticks(list(ys), [r[0] for r in rows])
    ax.set_xlim(74, 96)
    ax.set_xlabel("Validation strict-match (%)")
    ax.set_title("Every Qwen3.5-4B validation run")
    for y, row in zip(ys, rows):
        ax.text(row[1] + 0.25, y, f"{row[1]:.1f}", va="center", fontsize=8, color=INK)
    clean(ax, "x")
    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor=GREEN, label="Submitted checkpoint (val 91.4, test 91.28)"),
        Patch(facecolor=BLUE, label="Morning trainer"),
        Patch(facecolor=STONE, label="Evening trainer"),
    ], frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    save(fig, "qwen35_4b_runs.png")


def catalog():
    offsets = {
        "LFM2.5-230M": (8, 8),
        "LFM2.5-350M": (8, -12),
        "Qwen3.5-0.8B": (8, -2),
        "LFM2.5-1.2B": (8, 6),
        "Llama-3.2-1B": (8, -12),
        "Qwen3.5-2B": (8, 6),
        "Llama-3.2-3B": (8, 8),
        "Gemma-4-E2B": (8, -4),
        "Gemma-4-E4B": (8, -12),
        "Qwen3-8B": (8, 8),
        "Qwen3.5-9B": (-78, 8),
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for name, params, acc, score in CATALOG:
        if name == "Qwen3.5-4B":
            continue
        ax.scatter(params, score, s=42, color=BLUE, zorder=3)
        dx, dy = offsets[name]
        ax.annotate(name, (params, score), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, color=MUTED)
    ax.scatter([4.681], [86.72], s=80, color=GREEN, zorder=4)
    ax.annotate("Qwen3.5-4B val 86.72", (4.681, 86.72),
                textcoords="offset points", xytext=(10, -18), fontsize=8, color=GREEN)
    ax.scatter([4.206], [87.07], s=160, marker="*", color=AMBER, zorder=5)
    ax.annotate("submitted test 87.07\ntext-only count", (4.206, 87.07),
                textcoords="offset points", xytext=(-12, 14), ha="right", fontsize=8, color=AMBER)
    ax.set_xlabel("Parameters (billions, stored count)")
    ax.set_ylabel("Score  =  accuracy %  −  parameters (B)")
    ax.set_title("Best validation score of each base model")
    ax.set_xlim(-0.3, 11.2)
    ax.set_ylim(22, 98)
    clean(ax)
    fig.text(0.0, -0.02, "Circles are 500-question validation at the stored parameter count. The star is the submitted test score, same answers, text-only count.",
             color=MUTED, fontsize=9)
    save(fig, "model_scale.png")


def errors():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = [e[0] for e in ERRORS]
    vals = [e[1] for e in ERRORS]
    colors = [AMBER, STONE, BLUE, TEAL]
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.62, zorder=3)
    ax.set_xlabel("Test questions (of 1,319)")
    ax.set_title("Where the submitted model missed")
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + 1.2, bar.get_y() + bar.get_height() / 2, str(v), va="center", fontsize=10)
    ax.set_xlim(0, 110)
    clean(ax, "x")
    fig.text(0.0, -0.04, "1,204 correct. The four miss types partition the other 115. Eight of them are the right integer written as N.00.",
             color=MUTED, fontsize=9)
    save(fig, "errors.png")


def board():
    rows = list(reversed(BOARD))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    colors = [GREEN if name == "Shiven Tripathi" else STONE for name, _ in rows]
    ax.barh(range(len(rows)), [s for _, s in rows], color=colors, height=0.7, zorder=3)
    ax.set_yticks(range(len(rows)), [n for n, _ in rows])
    ax.set_xlim(84, 92)
    ax.set_xlabel("Submitted score")
    ax.set_title("Final leaderboard, top 10 of 28")
    for i, (_, s) in enumerate(rows):
        ax.text(s + 0.08, i, f"{s:.2f}", va="center", fontsize=9)
    clean(ax, "x")
    fig.text(0.0, -0.04, "Final board, 7 October 2026. Axis starts at 84 so the gaps are readable. Rankings use unrounded scores. 28 teams in all.",
             color=MUTED, fontsize=9)
    save(fig, "leaderboard.png")


def export_csv():
    path = ROOT / "data" / "runs.csv"
    lines = ["series,name,val_strict,params_b,val_score_official,notes"]
    for name, acc, kind in RUNS_4B:
        lines.append(f"qwen3.5-4b,{name},{acc},4.681,{acc - 4.681:.2f},{kind}")
    for name, params, acc, score in CATALOG:
        if name == "Qwen3.5-4B":
            continue
        lines.append(f"catalog,{name},{acc},{params},{score},best 500-row validation")
    lines.append("test,submitted official count,91.2813,4.6811,86.60,1204/1319")
    lines.append("test,submitted text-only count,91.2813,4.2058,87.07,same 1204/1319")
    lines.append("test,first checkpoint,81.6528,4.6811,76.97,1077/1319 superseded")
    path.write_text("\n".join(lines) + "\n")
    print(path.name)


if __name__ == "__main__":
    lr_sweep()
    dose()
    runs_4b()
    catalog()
    errors()
    board()
    export_csv()
