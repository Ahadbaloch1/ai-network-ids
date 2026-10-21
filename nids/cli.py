"""Command-line interface: generate, train, predict."""

import argparse
import os
import sys

import pandas as pd

from . import __version__
from .data import generate
from .model import feature_importance, load_model, predict, save_model, train
from .report import save_reports


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def build_parser():
    p = argparse.ArgumentParser(prog="nids", description="AI-based network intrusion detection")
    p.add_argument("--version", action="version", version="nids " + __version__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Create a synthetic flow dataset")
    g.add_argument("--samples", type=int, default=20000)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default="data/flows.csv")
    g.add_argument("--unlabeled", action="store_true", help="Omit the label column")

    t = sub.add_parser("train", help="Train and evaluate the models")
    t.add_argument("--data", default="data/flows.csv")
    t.add_argument("--model", default="models/nids.joblib")
    t.add_argument("--report", default="reports", help="Folder for evaluation plots")
    t.add_argument("--seed", type=int, default=42)

    q = sub.add_parser("predict", help="Classify flows from a CSV file")
    q.add_argument("--model", default="models/nids.joblib")
    q.add_argument("--data", required=True)
    q.add_argument("--out", default="predictions.csv")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            df = generate(args.samples, args.seed, labeled=not args.unlabeled)
            _ensure_parent(args.out)
            df.to_csv(args.out, index=False)
            print("Wrote %d flows to %s" % (len(df), args.out))
            if "label" in df.columns:
                print(df["label"].value_counts().to_string())
        elif args.command == "train":
            df = pd.read_csv(args.data)
            bundle, metrics = train(df, seed=args.seed)
            _ensure_parent(args.model)
            save_model(bundle, args.model)
            print("Training rows: %d, test rows: %d" % (metrics["n_train"], metrics["n_test"]))
            print("Cross-validation accuracy: %.3f (+/- %.3f)"
                  % (metrics["cv_accuracy_mean"], metrics["cv_accuracy_std"]))
            print("Test accuracy: %.3f\n" % metrics["accuracy"])
            print(metrics["report"])
            for path in save_reports(metrics, feature_importance(bundle), args.report):
                print("Saved " + path)
            print("Saved model to " + args.model)
        elif args.command == "predict":
            bundle = load_model(args.model)
            result = predict(bundle, pd.read_csv(args.data))
            _ensure_parent(args.out)
            result.to_csv(args.out, index=False)
            print("Classified %d flows. Alerts: %d" % (len(result), int(result["alert"].sum())))
            print(result["prediction"].value_counts().to_string())
            print("Saved " + args.out)
    except (ValueError, OSError, KeyError) as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1
    return 0
