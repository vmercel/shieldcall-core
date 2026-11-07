#!/usr/bin/env python3
"""Phase 2c - metrics from the frozen score files. No fitting on test: thresholds, Platt calibration
and abstention bands come from dev. Output: docs/results/phase2_benchmark.json (+ markdown tables)."""
from __future__ import annotations
import _repo  # noqa: F401
import json, gzip, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

SEED = 20261107
RES = Path("docs/results")
WORK = Path("data/work")
ACOUSTIC = ["gmmA", "lrA", "gmmB", "lrB"]
LANG = ["repo", "wide_bag", "tfidf_lr"]
ABSTAIN_FRAC = 0.20


def roc_eer(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    o = np.argsort(-s, kind="mergesort"); ys = y[o]
    P, N = ys.sum(), len(ys) - ys.sum()
    if P == 0 or N == 0:
        return float("nan"), float("nan"), float("nan")
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    tpr = np.r_[0, tp / P]; fpr = np.r_[0, fp / N]
    auc = float(np.trapz(tpr, fpr))
    fnr = 1 - tpr
    i = np.argmin(np.abs(fnr - fpr))
    thr = s[o][min(i, len(s) - 1)] if i > 0 else s[o][0] + 1e-9
    return float((fnr[i] + fpr[i]) / 2), auc, float(thr)


def eer_thr(y, s):
    return roc_eer(y, s)[2]


def ece(y, p, bins=10):
    y = np.asarray(y); p = np.asarray(p)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (p >= lo) & (p < lo + 1 / bins + (1e-9 if lo + 1 / bins >= 1 else 0))
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def boot_eer(y, s, rng, n=200):
    y = np.asarray(y); s = np.asarray(s)
    v = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        v.append(roc_eer(y[i], s[i])[0])
    v = np.array(v)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


class Calib:
    """thr/Platt/abstain band fitted on dev only."""
    def __init__(self, y, s):
        self.thr = eer_thr(y, s)
        s = np.asarray(s, float)
        self.lr = LogisticRegression(C=1e4, max_iter=1000).fit(s[:, None], y)
        self.delta = float(np.quantile(np.abs(s - self.thr), ABSTAIN_FRAC))
        self.dev_eer = roc_eer(y, s)[0]

    def p(self, s):
        return self.lr.predict_proba(np.asarray(s, float)[:, None])[:, 1]


def summarize(y, s, c: Calib, rng=None, ci=False):
    y = np.asarray(y); s = np.asarray(s, float)
    eer, auc, _ = roc_eer(y, s)
    flag = s >= c.thr
    fpr = float(flag[y == 0].mean()) if (y == 0).any() else float("nan")
    fnr = float((~flag[y == 1]).mean()) if (y == 1).any() else float("nan")
    p = c.p(s)
    keep = np.abs(s - c.thr) > c.delta
    out = dict(n=int(len(y)), n_pos=int(y.sum()), eer=eer, auc=auc, fpr=fpr, fnr=fnr,
               ece=ece(y, p), brier=float(np.mean((p - y) ** 2)),
               abstain_rate=float(1 - keep.mean()),
               fpr_retained=float(flag[keep & (y == 0)].mean()) if (keep & (y == 0)).any() else float("nan"),
               fnr_retained=float((~flag[keep & (y == 1)]).mean()) if (keep & (y == 1)).any() else float("nan"))
    if ci:
        out["eer_ci95"] = boot_eer(y, s, rng)
    return out


def acoustic(rng):
    sc = pd.read_csv(RES / "phase2_acoustic_scores.csv.gz")
    dev = pd.DataFrame(json.load(open(WORK / "dev_scores.json")))
    lat = json.load(open(WORK / "latency_acoustic.json"))
    out = dict(by_condition={}, by_attack={}, by_speaker={}, calibration={}, latency=lat)
    cals = {m: Calib(dev.label.values, dev[m].values) for m in ACOUSTIC}
    out["calibration"] = {m: dict(threshold=c.thr, dev_eer=c.dev_eer, abstain_delta=c.delta) for m, c in cals.items()}
    for m in ACOUSTIC:
        out["by_condition"][m] = {cond: summarize(g.label, g[m], cals[m], rng, ci=True) for cond, g in sc.groupby("cond", sort=False)}
    seen = set(dev[dev.label == 1].attack) | {"A16", "A19"}  # A16/A19 repeat the README descriptions of A04/A06 (train attacks)
    ATYPE = dict(A01="TTS neural waveform", A02="TTS vocoder", A03="TTS vocoder", A04="TTS waveform concat", A05="VC vocoder",
                 A06="VC spectral filtering", A07="TTS vocoder+GAN", A08="TTS neural waveform", A09="TTS vocoder", A10="TTS neural waveform",
                 A11="TTS griffin-lim", A12="TTS neural waveform", A13="TTS_VC waveform concat+filtering", A14="TTS_VC vocoder",
                 A15="TTS_VC neural waveform", A16="TTS waveform concat", A17="VC waveform filtering", A18="VC vocoder", A19="VC spectral filtering")
    for m in ACOUSTIC:
        out["by_attack"][m] = {}
        for cond in ("clean_16k", "g711_ulaw", "amr_nb_12k2", "ulaw_snr10"):
            g = sc[sc.cond == cond]; bona = g[g.label == 0]
            d = {}
            for atk, ga in g[g.label == 1].groupby("attack"):
                q = pd.concat([bona, ga]); e = roc_eer(q.label, q[m])[0]
                flag = ga[m].values >= cals[m].thr
                d[atk] = dict(n=int(len(ga)), eer_vs_bonafide=e, fnr=float(1 - flag.mean()), algorithm_seen_in_training=atk in seen, type=ATYPE[atk])
            out["by_attack"][m][cond] = d
        out["by_speaker"][m] = {}
        for cond in ("clean_16k", "g711_ulaw"):
            g = sc[sc.cond == cond]; flag = g[m] >= cals[m].thr
            g = g.assign(flag=flag.values)
            fp = g[g.label == 0].groupby("speaker").flag.mean(); fn = g[g.label == 1].groupby("speaker").flag.apply(lambda v: 1 - v.mean())
            gl_fpr = float(g[g.label == 0].flag.mean())
            out["by_speaker"][m][cond] = dict(
                n_speakers=int(g.speaker.nunique()), global_fpr=gl_fpr,
                fpr_median=float(fp.median()), fpr_p90=float(fp.quantile(.9)), fpr_max=float(fp.max()),
                fnr_median=float(fn.median()), fnr_p90=float(fn.quantile(.9)), fnr_max=float(fn.max()),
                speakers_fpr_over_2x_global=int((fp > 2 * gl_fpr).sum()) if gl_fpr > 0 else None)
    return out, sc, dev, cals


def language(rng):
    sc = pd.read_csv(RES / "phase2_language_scores.csv.gz")
    out = dict(by_wer={}, by_type={}, calibration={}, meta=json.load(open(WORK / "language_meta.json")))
    dev0 = sc[(sc.part == "dev") & (sc.wer == 0)]
    cals = {m: Calib(dev0.label.values, dev0[m].values) for m in LANG}
    out["calibration"] = {m: dict(threshold=c.thr, dev_eer=c.dev_eer) for m, c in cals.items()}
    test = sc[sc.part == "test"]
    for m in LANG:
        out["by_wer"][m] = {str(w): summarize(g.label, g[m], cals[m], rng, ci=True) for w, g in test.groupby("wer")}
        out["by_type"][m] = {}
        for w in (0.0, 0.30):
            g = test[test.wer == w]
            flag = g[m] >= cals[m].thr
            g = g.assign(flag=flag.values)
            out["by_type"][m][str(w)] = {t: dict(n=int(len(x)), label=int(x.label.iloc[0]),
                                                  error_rate=float((x.flag != x.label).mean())) for t, x in g.groupby("type")}
    return out, sc, cals


def fusion(ac, ac_cals, lg, lg_cals, rng):
    """Paired SYNTHETIC calls: random test audio clip + random test dialogue. Threat = spoof OR scam."""
    A = "gmmB"; L = "repo"; wer = 0.15
    txt = lg[(lg.part == "test") & (lg.wer == wer)].reset_index(drop=True)
    txd = lg[(lg.part == "dev") & (lg.wer == wer)].reset_index(drop=True)
    dev_ac = pd.DataFrame(json.load(open(WORK / "dev_scores.json")))

    def pair(a, t, r):
        ia = r.randint(0, len(a), len(a)); it = r.randint(0, len(t), len(a))
        a = a.iloc[ia].reset_index(drop=True); t = t.iloc[it].reset_index(drop=True)
        return pd.DataFrame(dict(sa=a[A].values, st=t[L].values, ya=a.label.values, yt=t.label.values))
    r = np.random.RandomState(SEED)
    dp = pair(dev_ac, txd, r); dp["y"] = ((dp.ya + dp.yt) > 0).astype(int)
    pa, pt = ac_cals[A].p(dp.sa), lg_cals[L].p(dp.st)
    lrf = LogisticRegression(C=10, max_iter=1000).fit(np.c_[pa, pt], dp.y)
    res = {}
    for cond in ("clean_16k", "g711_ulaw", "amr_nb_12k2", "ulaw_snr10", "ulaw_snr0"):
        g = ac[ac.cond == cond]
        tp = pair(g, txt, np.random.RandomState(SEED + 1)); tp["y"] = ((tp.ya + tp.yt) > 0).astype(int)
        fa = tp.sa >= ac_cals[A].thr; ft = tp.st >= lg_cals[L].thr
        pa_, pt_ = ac_cals[A].p(tp.sa), lg_cals[L].p(tp.st)
        systems = {
            "audio_only": fa.values, "text_only": ft.values, "OR_rule": (fa | ft).values,
            "noisy_OR_p>0.5": (1 - (1 - pa_) * (1 - pt_) > 0.5),
            "logistic_fusion": lrf.predict_proba(np.c_[pa_, pt_])[:, 1] > 0.5,
        }
        res[cond] = {}
        for name, flag in systems.items():
            flag = np.asarray(flag); y = tp.y.values
            cells = {}
            for cn, m in (("human_benign", (tp.ya == 0) & (tp.yt == 0)), ("human_scamscript", (tp.ya == 0) & (tp.yt == 1)),
                          ("synthetic_benign", (tp.ya == 1) & (tp.yt == 0)), ("synthetic_scamscript", (tp.ya == 1) & (tp.yt == 1))):
                cells[cn] = float(flag[m.values].mean())  # human_benign: false-positive rate; others: detection rate
            res[cond][name] = dict(fpr=float(flag[y == 0].mean()), fnr=float((~flag[y == 1]).mean()),
                                   rate_flagged_by_cell=cells, n=int(len(y)))
    return res


def main():
    rng = np.random.RandomState(SEED)
    ac, sc, dev, ac_cals = acoustic(rng)
    lg, lsc, lg_cals = language(rng)
    fu = fusion(sc, ac_cals, lsc, lg_cals, rng)
    lock = json.load(open(RES / "phase1_lock.json"))
    out = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), phase1_lock=dict(
        manifest_sha256=lock["manifest_sha256"], heldout_test_audio_sha256=lock["heldout_test_audio_sha256"]),
        acoustic=ac, language=lg, fusion_synthetic_pairs=fu,
        axes_not_covered=["language (audio corpus is English VCTK speakers; text is English)",
                          "real phone-call recordings", "real (non-LLM-generated) scam transcripts",
                          "voice-cloning systems newer than 2019"])
    (RES / "phase2_benchmark.json").write_text(json.dumps(out, indent=1))
    print("wrote phase2_benchmark.json")


if __name__ == "__main__":
    main()
