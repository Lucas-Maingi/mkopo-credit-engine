"""Command line entry point for the Mkopo Score pipeline.

Each pipeline stage is a subcommand so the whole thing is reproducible from a
shell (and from CI) without a notebook in sight::

    mkopo generate-data
    mkopo build-features
    mkopo train
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys

from mkopo import __version__

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

#: subcommand -> "module:function" resolved lazily so that ``mkopo --help`` stays
#: fast and does not import LightGBM/MLflow just to print usage.
COMMANDS: dict[str, tuple[str, str]] = {
    "generate-data": ("mkopo.data.generator", "Generate the synthetic mobile-money book"),
    "build-features": ("mkopo.features.pipeline", "Build the WOE feature matrix"),
    "fit-scorecard": ("mkopo.models.scorecard", "Fit the interpretable logistic scorecard"),
    "train": ("mkopo.models.gbm", "Train LightGBM/XGBoost with Optuna + MLflow"),
    "reject-inference": ("mkopo.models.reject_inference", "Correct approval selection bias"),
    "calibrate": ("mkopo.models.calibration", "Calibrate, scale to points, price the cut-off"),
    "fairness": ("mkopo.evaluation.fairness", "Run the fairness audit"),
    "export-onnx": ("mkopo.models.export_onnx", "Export champion to ONNX and benchmark"),
}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mkopo", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"mkopo {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, (_module, help_text) in COMMANDS.items():
        sub.add_parser(name, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    args, extra = build_parser().parse_known_args(argv)
    _configure_logging(args.verbose)

    module_path, _ = COMMANDS[args.command]
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:  # stage not implemented yet
        logging.error("stage '%s' is not available: %s", args.command, exc)
        return 2

    entry = getattr(module, "main", None)
    if entry is None:
        logging.error("module %s has no main()", module_path)
        return 2
    return int(entry(extra) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
