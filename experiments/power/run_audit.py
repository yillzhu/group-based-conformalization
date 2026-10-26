#!/usr/bin/env python3
"""Run the IEEE 39-bus residual-geography falsification audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conformal_cl.power.audit import PowerAuditConfig, run_power_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = (
        PowerAuditConfig.smoke(args.seed)
        if args.smoke
        else PowerAuditConfig(seed=args.seed)
    )
    result = run_power_audit(config)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
