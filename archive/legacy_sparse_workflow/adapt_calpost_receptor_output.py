from __future__ import annotations

import argparse
from pathlib import Path

from transfer_matrix.calpost_adapter import adapt_calpost_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize a verified CALPOST CSV export using an explicit receptor manifest."
    )
    parser.add_argument("--input", type=Path, required=True, help="CSV exported from CALPOST or a verified converter.")
    parser.add_argument("--receptor-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Assembler-compatible receptors.csv.")
    parser.add_argument("--value-column", default="concentration")
    parser.add_argument("--value-unit", default=None)
    args = parser.parse_args()
    output = adapt_calpost_csv(
        args.input,
        args.receptor_manifest,
        args.output,
        value_column=args.value_column,
        value_unit=args.value_unit,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
