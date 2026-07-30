from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CaseConfig


def write_diagnostics(config: CaseConfig) -> list[Path]:
    matrix_path = config.output_path("matrix_npz")
    if not matrix_path.exists():
        raise RuntimeError("Matrix NPZ not found. Run assemble first.")

    payload = np.load(matrix_path, allow_pickle=True)
    matrix = payload["T"]
    region_ids = payload["region_ids"].astype(str)
    diag_dir = config.output_path("diagnostics_dir")
    diag_dir.mkdir(parents=True, exist_ok=True)

    column_sums = matrix.sum(axis=1)
    summary = pd.DataFrame(
        {
            "hour": np.repeat(np.arange(matrix.shape[0]), matrix.shape[2]),
            "source_region": np.tile(region_ids, matrix.shape[0]),
            "within_domain_sum": column_sums.reshape(-1),
        }
    )
    summary_path = diag_dir / "column_mass_summary.csv"
    summary.to_csv(summary_path, index=False)

    outputs = [summary_path]
    try:
        import matplotlib.pyplot as plt

        for hour in range(matrix.shape[0]):
            fig, ax = plt.subplots(figsize=(7, 6))
            image = ax.imshow(matrix[hour], origin="lower", aspect="auto")
            ax.set_title(f"Transfer matrix hour {hour:02d}")
            ax.set_xlabel("source region")
            ax.set_ylabel("target region")
            fig.colorbar(image, ax=ax, label="response fraction")
            out = diag_dir / f"matrix_heatmap_hour_{hour:02d}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            outputs.append(out)
    except ImportError:
        pass
    return outputs

