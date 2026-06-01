"""Plot Space-L vs Space-H alignment and H7 beta_d2 comparison for the report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import PROJECT_ROOT  # noqa: E402

SUMMARY = PROJECT_ROOT / ".cache" / "space_l" / "space_l_summary.json"
OUT = PROJECT_ROOT / "results" / "figures" / "space_l_validation.png"

H7 = [
    ("4B", -0.641, -0.121),
    ("8B", -0.744, -0.196),
    ("14B", -0.432, -0.175),
]
PERSONA_LABELS = ["RC", "EC", "RA", "EA"]


def main() -> int:
    if not SUMMARY.exists():
        print(f"Missing {SUMMARY}; run: python -m src.run_experiment build_space_l", file=sys.stderr)
        return 1

    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    proc = data["procrustes_to_h"]
    h = np.asarray(proc["aligned_H"], dtype=float)
    l = np.asarray(proc["aligned_L"], dtype=float)
    disp = float(proc["disparity"])
    ax_corr = proc["axis_correlations"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=150)

    ax = axes[0]
    ax.scatter(h[:, 0], h[:, 1], s=80, marker="D", c="#2166ac", label="Space-H (aligned)", zorder=3)
    ax.scatter(l[:, 0], l[:, 1], s=80, marker="o", c="#b2182b", label="Space-L (PCA)", zorder=3)
    for i, lab in enumerate(PERSONA_LABELS):
        ax.annotate(lab, (h[i, 0], h[i, 1]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.plot([h[i, 0], l[i, 0]], [h[i, 1], l[i, 1]], "k-", alpha=0.25, lw=0.8)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("$S$ (aligned)")
    ax.set_ylabel("$R$ (aligned)")
    ax.set_title(f"Persona layout (Procrustes $d={disp:.3f}$)")
    ax.legend(loc="best", fontsize=7)
    ax.set_aspect("equal", adjustable="box")

    ax2 = axes[1]
    labels = [x[0] for x in H7]
    x = np.arange(len(labels))
    w = 0.35
    ax2.bar(x - w / 2, [x[1] for x in H7], width=w, label=r"$d_H$", color="#2166ac")
    ax2.bar(x + w / 2, [x[2] for x in H7], width=w, label=r"$d_L$", color="#b2182b")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Qwen3-{lb}" for lb in labels], fontsize=8)
    ax2.set_ylabel(r"MixedLM $\beta_{d^2}$ on $C_{\mathrm{auto}}$")
    ax2.set_title("H7: same negative sign (T3, $n{=}720$)")
    ax2.legend(fontsize=7)

    evr = data.get("explained_variance_ratio", [0.507, 0.369])
    fig.suptitle(
        f"Space-L: PC1$\\leftrightarrow R$ $r={abs(ax_corr['PC1_vs_R']):.2f}$, "
        f"PC2$\\leftrightarrow S$ $r={ax_corr['PC2_vs_S']:.2f}$; PCA var. {evr[0]:.0%}+{evr[1]:.0%}",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_space_l_validation] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
