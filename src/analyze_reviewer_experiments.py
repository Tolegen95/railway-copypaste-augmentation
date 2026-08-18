"""Aggregate repeated runs and compute confidence intervals and paired tests."""
import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="full_reviewer_experiments/results.csv")
    ap.add_argument("--out", default="full_reviewer_experiments/statistics")
    args = ap.parse_args()
    df = pd.read_csv(args.results)
    df = df[df.status == "complete"].copy()
    metrics = ["precision", "recall", "mAP50", "mAP50_95", "inference_ms"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = df.groupby(["model", "variant"])[metrics].agg(["count", "mean", "std"])
    summary.to_csv(out / "summary_mean_sd.csv")
    comparisons = []
    for model in sorted(df.model.unique()):
        base = df[(df.model == model) & (df.variant == "baseline")].set_index("seed")
        for variant in sorted(set(df[df.model == model].variant) - {"baseline"}):
            alt = df[(df.model == model) & (df.variant == variant)].set_index("seed")
            common = sorted(set(base.index) & set(alt.index))
            for metric in metrics[:-1]:
                a, b = base.loc[common, metric], alt.loc[common, metric]
                d = b - a; n = len(d)
                if n < 2: continue
                sem = stats.sem(d)
                ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=sem)
                tt = stats.ttest_rel(b, a)
                try: wx = stats.wilcoxon(b, a, method="auto")
                except ValueError: wx = None
                comparisons.append({
                    "model": model, "variant": variant, "metric": metric, "n": n,
                    "mean_paired_difference": d.mean(), "ci95_low": ci[0], "ci95_high": ci[1],
                    "paired_t_p": tt.pvalue, "wilcoxon_p": None if wx is None else wx.pvalue,
                })
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.csv", index=False)
    print(summary)


if __name__ == "__main__": main()
