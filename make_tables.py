"""
make_tables.py — Build every results table in the paper from runs/*/metrics.json
================================================================================
No number in the paper's results tables is typed by hand: this script reads
the per-seed metrics files, computes mean +/- sample std over seeds and
Welch t-tests against QGenNet (ZNE, p = 0.001) with Bonferroni correction
over the comparisons in each table, and writes LaTeX to ../tables/.
It also writes results/summary.json with all aggregated numbers.

    python make_tables.py [--out DIR]     (default: outputs/)
"""

import glob
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
TAB = os.path.join(HERE, "outputs", "tables")   # overridden by --out
REF = "qgennet_zne_p0.001"


def collect(domain, key, which="final"):
    """{run_name: [values over seeds]} for a metric key."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RUNS, domain, "*", "seed*", "metrics.json"))):
        rn = d.split(os.sep)[-3]
        if "pilot" in rn or "smoke" in rn:
            continue
        m = json.load(open(d))
        if which in m and key in m[which]:
            out.setdefault(rn, []).append(float(m[which][key]))
    return out


def fmt(v, digits):
    return f"{np.mean(v):.{digits}f} $\\pm$ {np.std(v, ddof=1) if len(v) > 1 else 0:.{digits}f}"


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    r = stats.ttest_ind(a, b, equal_var=False)
    return float(r.statistic), float(r.pvalue)


def pfmt(p):
    if np.isnan(p):
        return "---"
    if p < 1e-4:
        return "$<10^{-4}$"
    return f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}"


LABELS = {
    "qgennet_zne_p0.001": "QGenNet (ZNE)",
    "qgennet_nozne_p0.001": "QGenNet (no ZNE)",
    "qgennet_noent": "QGenNet (no entanglement)",
    "qgennet_zne_p0.001_nonorm": "QGenNet (ZNE, no feature norm.)",
    "decoder_only": "Decoder only (no PQC)",
    "mlp": "MLP WGAN-GP",
    "dcgan": "DCGAN-G WGAN-GP",
    "ctgan": "CTGAN (300 epochs)",
    "ctgan_e1500": "CTGAN (1500 epochs)",
    "timegan": "TimeGAN",
}


def table(domain, cols, rows, caption, label, summary, test_key=None, ref_cells=None):
    """cols: list of (metric_key, header, digits, lower_is_better).  The Welch
    test uses ``test_key`` (default: first column).  ``ref_cells`` is an
    optional list of strings for a real-data reference row."""
    data = {k: collect(domain, k) for k, *_ in cols}
    test_key = test_key or cols[0][0]
    rows = [r for r in rows if r in data[test_key]]
    n_cmp = max(1, len([r for r in rows if r != REF]))
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{l" + "c" * len(cols) + "c}", r"\toprule",
             "Model & " + " & ".join(h for _, h, _, _ in cols) + r" & $p_{\mathrm{Bonf}}$ \\", r"\midrule"]
    summary[domain] = {}
    for r in rows:
        cells = []
        summary[domain][r] = {}
        for k, h, dg, _ in cols:
            v = data[k].get(r, [])
            cells.append(fmt(v, dg) if v else "---")
            summary[domain][r][k] = {"mean": float(np.mean(v)) if v else None,
                                     "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                     "n": len(v), "values": v}
        if r == REF:
            pcell = "---"
        else:
            t, p = welch(data[test_key].get(REF, []), data[test_key].get(r, []))
            p_b = min(1.0, p * n_cmp) if not np.isnan(p) else np.nan
            summary[domain][r]["welch_vs_ref"] = {"metric": test_key, "t": t, "p": p,
                                                  "p_bonferroni": p_b, "n_comparisons": n_cmp}
            pcell = pfmt(p_b)
        lines.append(f"{LABELS.get(r, r)} & " + " & ".join(cells) + f" & {pcell} \\\\")
    nseeds = sorted({len(v) for v in data[test_key].values()})
    if ref_cells:
        lines += [r"\midrule",
                  r"Real data (reference)$^{\dagger}$ & " + " & ".join(ref_cells) + r" & --- \\"]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, f"{label.replace(':', '_')}.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{label}: seeds per row {nseeds}")


def params_table():
    """Generator / critic parameter counts read from config.json files."""
    order = ["qgennet_zne_p0.001", "qgennet_noent", "decoder_only", "mlp", "dcgan"]
    doms = ["mnist", "tabular", "stocks"]
    cnt = {}
    for dom in doms:
        for rn in order:
            fs = sorted(glob.glob(os.path.join(RUNS, dom, rn, "seed*", "config.json")))
            if fs:
                c = json.load(open(fs[0]))
                cnt[(dom, rn)] = (c["gen_params"], c["critic_params"])
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Trainable generator parameters (critic parameters in the last row). "
             r"QGenNet counts include the 144 circuit angles.}",
             r"\label{tab:params}", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Generator & MNIST & Tabular & Equities \\", r"\midrule"]
    for rn in order:
        cells = [f"{cnt[(d, rn)][0]:,}" if (d, rn) in cnt else "---" for d in doms]
        lines.append(f"{LABELS[rn]} & " + " & ".join(cells) + r" \\")
    crit = [f"{cnt[(d, order[0])][1]:,}" if (d, order[0]) in cnt else "---" for d in doms]
    lines += [r"\midrule", "Critic (all WGAN-GP models) & " + " & ".join(crit) + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TAB, "tab_params.tex"), "w") as fh:
        fh.write("\n".join(lines).replace(",", "{,}") + "\n")


def ideal_table(summary):
    """Same trained generator evaluated on the noisy circuit it was trained
    with and on the ideal circuit (tabular TVD for every p; equities MAE)."""
    def pair(domain, rn, key):
        a, b = [], []
        for f in sorted(glob.glob(os.path.join(RUNS, domain, rn, "seed*", "metrics.json"))):
            m = json.load(open(f))
            if "final_ideal_circuit" in m:
                a.append(m["final"][key])
                b.append(m["final_ideal_circuit"][key])
        return np.array(a), np.array(b)

    rows = [("tabular", "mean_tvd", p) for p in ("0.001", "0.005", "0.01", "0.02")] + \
           [("stocks", "pred_mae", "0.001"), ("mnist", "fd_lenet", "0.001")]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{The same trained QGenNet generators evaluated on the noisy circuit "
             r"they were trained with and on the ideal (noise-free) circuit (mean $\pm$ std "
             r"over seeds; $n$ seeds). $p_{\mathrm{pair}}$: paired $t$-test, noisy vs.\ ideal, "
             r"uncorrected.}",
             r"\label{tab:ideal}", r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{llcccc}", r"\toprule",
             r"Metric & $p$ & Training & Noisy circuit & Ideal circuit & $p_{\mathrm{pair}}$ \\",
             r"\midrule"]
    summary["ideal_eval"] = {}
    for dom, key, p in rows:
        for zne in (True, False):
            rn = f"qgennet_{'zne' if zne else 'nozne'}_p{p}"
            a, b = pair(dom, rn, key)
            if len(a) < 2:
                continue
            pp = float(stats.ttest_rel(a, b).pvalue)
            summary["ideal_eval"][f"{dom}/{rn}/{key}"] = {
                "noisy_mean": float(a.mean()), "noisy_std": float(a.std(ddof=1)),
                "ideal_mean": float(b.mean()), "ideal_std": float(b.std(ddof=1)),
                "n": len(a), "p_paired": pp}
            mname = {"tabular": "Tab.\\ TVD", "stocks": "Eq.\\ pred.\\ MAE",
                     "mnist": "MNIST FD$_{\\mathrm{LeNet}}$"}[dom]
            dg = 1 if dom == "mnist" else 3
            lines.append(f"{mname} & {p} & {'ZNE' if zne else 'no ZNE'} ($n$={len(a)}) & "
                         f"{a.mean():.{dg}f} $\\pm$ {a.std(ddof=1):.{dg}f} & "
                         f"{b.mean():.{dg}f} $\\pm$ {b.std(ddof=1):.{dg}f} & {pfmt(pp)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(os.path.join(TAB, "tab_ideal.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    os.makedirs(TAB, exist_ok=True)
    params_table()
    summary = {}
    ref = json.load(open(os.path.join(HERE, "results", "reference.json")))
    ms = lambda d, dg: f"{d['mean']:.{dg}f} $\\pm$ {d['std']:.{dg}f}"
    table("mnist",
          [("fd_lenet", r"FD$_{\mathrm{LeNet}}$ $\downarrow$", 1, True),
           ("ndb_k", r"NDB/$K$ $\downarrow$", 2, True)],
          ["qgennet_zne_p0.001", "qgennet_nozne_p0.001", "qgennet_noent",
           "decoder_only", "mlp", "dcgan"],
          r"MNIST results (mean $\pm$ std over seeds). $p_{\mathrm{Bonf}}$: Welch $t$-test on "
          r"FD$_{\mathrm{LeNet}}$ vs.\ QGenNet (ZNE), Bonferroni-corrected. "
          r"$^{\dagger}$10\,000 real training images vs.\ the test set.",
          "tab:mnist", summary,
          ref_cells=[f"{ref['mnist']['fd_lenet_train_vs_test']:.1f}",
                     f"{ref['mnist']['ndb_k_train_vs_test']:.2f}"])
    table("tabular",
          [("mean_tvd", r"Mean TVD $\downarrow$", 3, True),
           ("corr_diff", r"Corr.\ diff.\ $\downarrow$", 3, True),
           ("ml_eff", r"ML efficacy (AUROC) $\uparrow$", 3, False)],
          ["qgennet_zne_p0.001", "qgennet_nozne_p0.001", "qgennet_noent",
           "qgennet_zne_p0.001_nonorm", "decoder_only", "mlp", "ctgan", "ctgan_e1500"],
          r"Breast Cancer Wisconsin results (mean $\pm$ std over seeds). $p_{\mathrm{Bonf}}$: "
          r"Welch $t$-test on mean TVD vs.\ QGenNet (ZNE), Bonferroni-corrected. "
          r"$^{\dagger}$TVD and correlation difference: real test split (114 rows) vs.\ "
          r"training split, inflated by the small test size; ML efficacy: classifiers "
          r"trained on the real training split (5 seeds).",
          "tab:tabular", summary,
          ref_cells=[f"{ref['tabular']['mean_tvd_test_vs_train']:.3f}",
                     f"{ref['tabular']['corr_diff_test_vs_train']:.3f}",
                     ms(ref['tabular']['ml_eff_trtr'], 3)])
    table("stocks",
          [("pred_mae", r"Pred.\ MAE $\downarrow$", 3, True),
           ("corr_diff", r"Corr.\ diff.\ $\downarrow$", 3, True),
           ("disc_score", r"Disc.\ score $\downarrow$", 3, True)],
          ["qgennet_zne_p0.001", "qgennet_nozne_p0.001", "qgennet_noent",
           "decoder_only", "mlp", "timegan"],
          r"Equity log-return windows (mean $\pm$ std over seeds). $p_{\mathrm{Bonf}}$: Welch "
          r"$t$-test on predictive MAE vs.\ QGenNet (ZNE), Bonferroni-corrected. "
          r"$^{\dagger}$Predictive MAE: GRU trained on real training windows (5 seeds); "
          r"correlation difference and discriminative score: real test-period windows vs.\ "
          r"training-period windows.",
          "tab:stocks", summary,
          ref_cells=[ms(ref['stocks']['pred_mae_trtr'], 3),
                     f"{ref['stocks']['corr_diff_test_vs_train']:.3f}",
                     ms(ref['stocks']['disc_score_test_vs_train'], 3)])
    ideal_table(summary)
    # least-squares slopes of log2(Var) vs n for n >= 8 (quoted in Sec. VII)
    g = json.load(open(os.path.join(HERE, "results", "gradvar.json")))
    summary["gradvar_slopes_log2_per_qubit_n_ge_8"] = {}
    for init in ("uniform_pi", "small_0.1pi"):
        r = [x for x in g["rows"] if x["init"] == init and x["n"] >= 8]
        n = np.array([x["n"] for x in r])
        for k in ("var_local", "var_global"):
            summary["gradvar_slopes_log2_per_qubit_n_ge_8"][f"{init}/{k}"] = \
                float(np.polyfit(n, np.log2([x[k] for x in r]), 1)[0])
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"),
                    help="tables are written to <out>/tables (use .. inside the paper folder)")
    TAB = os.path.join(ap.parse_args().out, "tables")
    main()
    print("tables written to", os.path.abspath(TAB))
