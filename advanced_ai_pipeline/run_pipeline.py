"""Run the advanced AI pipeline from the command line."""

from __future__ import annotations

import argparse
import json

from .api_handler import handle_doctor_input


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Advanced AI Doctor Pipeline")
    parser.add_argument("--drug1", required=True, help="First drug name or SMILES string")
    parser.add_argument("--drug2", required=True, help="Second drug name or SMILES string")
    parser.add_argument("--description", required=False, default="Doctor submitted interaction.", help="Short interaction description")
    args = parser.parse_args()

    result = handle_doctor_input(args.drug1, args.drug2, args.description)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
