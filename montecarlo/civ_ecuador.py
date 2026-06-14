"""
Monte Carlo match simulator — Jump Trading Probability Cup
Match: Cote d'Ivoire vs Ecuador, 2026 World Cup group stage, Philadelphia.

TIGHT, LOW-SCORING — two elite defences, draw very live. Ecuador is the slight
BETTING favourite (counters from a deep block), but Cote d'Ivoire carries more
of the possession/initiative via its wingers. So the two "favourite" senses
split across axes:
  - team A / possession axis ("fav" keys) = Cote d'Ivoire (more initiative)
  - team B / counter axis    ("dog" keys) = Ecuador (more likely to win)
The 1X2 anchor is set accordingly: market_wdl["fav"] = CIV win = 0.28.

Weighted-Brier scoring: calibration over confidence; never 0/100; 42-58% flagged.
Same anchored Dixon-Coles / shared-dominance / half-split engine as prior matches.
Benched players are faded hard but keep a floor (early-injury sub risk).
"""

import time
import numpy as np

CFG = {
    "seed": 20260614,
    "n_sims": 200_000,

    "team_fav": "Cote d'Ivoire",   # team A / possession axis (NOT the betting fav)
    "team_dog": "Ecuador",         # team B / counter axis (betting favourite)

    # De-vigged 1X2 anchor: fav=CIV win, draw, dog=ECU win.
    "market_wdl": {"fav": 0.28, "draw": 0.32, "dog": 0.40},
    "ou_line": 1.5,   # market strongly favours UNDER (under = total <= 1)

    # Input xG (regularizer target). Total 1.85.
    "xg_fav": 0.90,   # CIV
    "xg_dog": 0.95,   # ECU

    # Possession ~ even, CIV slightly more initiative.
    "poss_fav": 0.52,   # CIV
    "poss_dog": 0.48,   # ECU
    "dominance_concentration": 18.0,

    # Per-90 base rates.
    "sot":     {"fav": 3.6, "dog": 3.7},   # GIVEN ~3.7/3.8; trimmed so combined ~7-7.5
                                           #   (CIV AFCON accuracy poor ~18% -> don't inflate)
    "corners": {"fav": 4.6, "dog": 4.2},   # INFERRED (CIV more initiative)
    "fouls":   {"fav": 11.0, "dog": 11.3}, # roughly even (ref ~22 fouls/g); CIV base uncertain
    "cards":   {"fav": 2.1, "dog": 2.2},   # GIVEN combined mean ~4.3 (card-heavy ref Letexier)
    "offsides":{"fav": 1.0, "dog": 0.9},   # GIVEN (CIV pacey wingers run the shoulder)

    # Half splits. Goals 42/58 (most WC goals come in the 2nd half).
    "split_goals_h1":   0.42,
    "split_cards_h1":   0.35,
    "split_corners_h1": 0.48,
    "split_sot_h1":     0.48,
    "trail_push": 1.20,

    "sot_combined_cap": 7.5,
    "sot_combined_target": 7.0,

    # Goal-model grid search. Negative rho (~-0.07) to lift low-score draws.
    "grid_goals_max": 12,
    "reg_weight": 0.012,
    "lam_grid": np.round(np.arange(0.50, 1.61, 0.05), 2),
    "mu_grid":  np.round(np.arange(0.50, 1.61, 0.05), 2),
    "rho_grid": np.round(np.arange(-0.15, 0.051, 0.02), 3),
    "rho_cap": 0.05,

    # Benched player props.
    "sangare":   {"exp_sot": 0.08, "side": "fav"},      # deep holder, late cameo -> ~7-8%
    "sarmiento_involve_per_goal": 0.055,                # benched winger, ~5-6% G+A
    "sangare_target": 0.075, "sarmiento_target": 0.055,
}

WC_NORMS = {"sot": 7.0, "corners": 8.9, "fouls": "22-26", "cards": 3.3}


# ---- goal model engine ----
def _poisson_pmf_vec(lmbda, kmax):
    k = np.arange(kmax + 1)
    logp = k * np.log(lmbda) - lmbda - np.cumsum(np.log(np.r_[1, np.arange(1, kmax + 1)]))
    return np.exp(logp)


def _dc_tau(lmbda, mu, rho, kmax):
    tau = np.ones((kmax + 1, kmax + 1))
    tau[0, 0] = 1.0 - lmbda * mu * rho
    tau[0, 1] = 1.0 + lmbda * rho
    tau[1, 0] = 1.0 + mu * rho
    tau[1, 1] = 1.0 - rho
    return tau


def dc_joint_pmf(lmbda, mu, rho, kmax):
    P = np.outer(_poisson_pmf_vec(lmbda, kmax), _poisson_pmf_vec(mu, kmax)) * _dc_tau(lmbda, mu, rho, kmax)
    P = np.clip(P, 0.0, None)
    P /= P.sum()
    return P


def wdl_from_pmf(P):
    return np.tril(P, -1).sum(), np.trace(P), np.triu(P, 1).sum()


def _grid_search(lam_arr, mu_arr, rho_arr, cfg, px_cache):
    kmax, tgt, reg, rho_cap = cfg["grid_goals_max"], cfg["market_wdl"], cfg["reg_weight"], cfg["rho_cap"]
    best = None
    for lmbda in lam_arr:
        px = px_cache.get(lmbda)
        if px is None:
            px = px_cache[lmbda] = _poisson_pmf_vec(lmbda, kmax)
        reg_lam = reg * (lmbda - cfg["xg_fav"]) ** 2
        for mu in mu_arr:
            py = px_cache.get(-mu)
            if py is None:
                py = px_cache[-mu] = _poisson_pmf_vec(mu, kmax)
            base = np.outer(px, py)
            reg_mu = reg * (mu - cfg["xg_dog"]) ** 2
            for rho in rho_arr:
                if rho > rho_cap:
                    continue
                P = np.clip(base * _dc_tau(lmbda, mu, rho, kmax), 0.0, None)
                P /= P.sum()
                f, d, g = wdl_from_pmf(P)
                loss = ((f - tgt["fav"]) ** 2 + (d - tgt["draw"]) ** 2 +
                        (g - tgt["dog"]) ** 2 + reg_lam + reg_mu)
                if best is None or loss < best[0]:
                    best = (loss, lmbda, mu, rho)
    return best


def anchor_goal_model(cfg):
    kmax = cfg["grid_goals_max"]
    px_cache = {}
    _, lmbda, mu, rho = _grid_search(cfg["lam_grid"], cfg["mu_grid"], cfg["rho_grid"], cfg, px_cache)
    lam_f = np.round(np.arange(max(0.05, lmbda - 0.03), lmbda + 0.031, 0.0125), 4)
    mu_f = np.round(np.arange(max(0.05, mu - 0.03), mu + 0.031, 0.0125), 4)
    rho_f = np.round(np.arange(rho - 0.02, rho + 0.0201, 0.005), 4)
    _, lmbda, mu, rho = _grid_search(lam_f, mu_f, rho_f, cfg, px_cache)
    return lmbda, mu, rho, dc_joint_pmf(lmbda, mu, rho, kmax)


def sample_goals(P, n, rng):
    kmax = P.shape[0] - 1
    idx = rng.choice(P.size, size=n, p=P.ravel())
    return (idx // (kmax + 1)).astype(np.int64), (idx % (kmax + 1)).astype(np.int64)


def split_with_push(rate, h1_frac, push_mult, trailing_mask, rng):
    h1 = rng.poisson(rate * h1_frac)
    h2_rate = np.where(trailing_mask, rate * (1.0 - h1_frac) * push_mult, rate * (1.0 - h1_frac))
    return h1, rng.poisson(h2_rate)


def fmt_ci(p, n):
    se = np.sqrt(p * (1 - p) / n)
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def simulate(cfg):
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["n_sims"]

    lmbda, mu, rho, P = anchor_goal_model(cfg)
    civ_goals, ecu_goals = sample_goals(P, n, rng)   # fav=CIV (rows), dog=ECU (cols)

    k = cfg["dominance_concentration"]
    d = rng.beta(cfg["poss_fav"] * k, (1 - cfg["poss_fav"]) * k, size=n)
    scale_fav = d / cfg["poss_fav"]          # CIV
    scale_dog = (1.0 - d) / cfg["poss_dog"]  # ECU

    g1f = rng.binomial(civ_goals, cfg["split_goals_h1"])
    g1d = rng.binomial(ecu_goals, cfg["split_goals_h1"])
    trailing_dog = g1f > g1d   # ECU behind at HT
    trailing_fav = g1d > g1f   # CIV behind at HT

    def build_sot(sf):
        rf = cfg["sot"]["fav"] * scale_fav * sf
        rd = cfg["sot"]["dog"] * scale_dog * sf
        fh1, fh2 = split_with_push(rf, cfg["split_sot_h1"], cfg["trail_push"], trailing_fav, rng)
        dh1, dh2 = split_with_push(rd, cfg["split_sot_h1"], cfg["trail_push"], trailing_dog, rng)
        return fh1, fh2, dh1, dh2
    sot_scale = 1.0
    sot_f_h1, sot_f_h2, sot_d_h1, sot_d_h2 = build_sot(sot_scale)
    combined = (sot_f_h1 + sot_f_h2 + sot_d_h1 + sot_d_h2).mean()
    if combined > cfg["sot_combined_cap"]:
        sot_scale = cfg["sot_combined_target"] / combined
        sot_f_h1, sot_f_h2, sot_d_h1, sot_d_h2 = build_sot(sot_scale)
    sot_f, sot_d = sot_f_h1 + sot_f_h2, sot_d_h1 + sot_d_h2

    cor_f_h1, cor_f_h2 = split_with_push(cfg["corners"]["fav"] * scale_fav, cfg["split_corners_h1"], cfg["trail_push"], trailing_fav, rng)
    cor_d_h1, cor_d_h2 = split_with_push(cfg["corners"]["dog"] * scale_dog, cfg["split_corners_h1"], cfg["trail_push"], trailing_dog, rng)

    off_f = rng.poisson(cfg["offsides"]["fav"] * scale_fav)   # CIV
    off_d = rng.poisson(cfg["offsides"]["dog"] * scale_dog)   # ECU

    fouls_f = rng.poisson(cfg["fouls"]["fav"] * scale_dog)    # CIV fouls more when ECU dominates
    fouls_d = rng.poisson(cfg["fouls"]["dog"] * scale_fav)    # ECU fouls more when CIV dominates

    cards_f = rng.poisson(cfg["cards"]["fav"], size=n)
    cards_d = rng.poisson(cfg["cards"]["dog"], size=n)

    # Sangare (CIV, benched holder): expected SOT directly modelled, dominance-scaled.
    sangare_sot = rng.poisson(cfg["sangare"]["exp_sot"] * scale_fav)
    # Sarmiento (ECU, benched winger): goal-OR-assist, joint off ECU's scored goals.
    sarmiento_inv = rng.binomial(ecu_goals, cfg["sarmiento_involve_per_goal"]) >= 1

    total_goals = civ_goals + ecu_goals
    h1_total = g1f + g1d
    h2_total = (civ_goals - g1f) + (ecu_goals - g1d)

    # ---- 10 PROPS ----
    q = {}
    q["Q1 Ecuador more fouls than CIV"]        = (fouls_d > fouls_f).mean()
    q["Q2 Ecuador more corners than CIV at HT"]= (cor_d_h1 > cor_f_h1).mean()
    q["Q3 Cote d'Ivoire 2+ offsides"]          = (off_f >= 2).mean()
    q["Q4 Ecuador more SOT than CIV in 2H"]    = (sot_d_h2 > sot_f_h2).mean()
    q["Q5 2H more goals than 1H (strict)"]     = (h2_total > h1_total).mean()
    q["Q6 Cote d'Ivoire win"]                  = np.tril(P, -1).sum()
    q["Q7 Cote d'Ivoire score in 2H"]          = ((civ_goals - g1f) >= 1).mean()
    q["Q8 4+ total cards"]                     = ((cards_f + cards_d) >= 4).mean()
    q["Q9 Sangare 1+ SOT (benched)"]           = (sangare_sot >= 1).mean()
    q["Q10 Sarmiento score or assist (bench)"] = sarmiento_inv.mean()

    diag = {
        "lambda": lmbda, "mu": mu, "rho": rho, "wdl": wdl_from_pmf(P), "sot_scale": sot_scale,
        "ou_line": cfg["ou_line"], "ou_under": (total_goals <= 1).mean(),
        "total_goals_mean": total_goals.mean(),
        "calib": {
            "sot_combined": (sot_f + sot_d).mean(),
            "corners_combined": (cor_f_h1 + cor_f_h2 + cor_d_h1 + cor_d_h2).mean(),
            "fouls_combined": (fouls_f + fouls_d).mean(), "cards_combined": (cards_f + cards_d).mean(),
            "sot_fav": sot_f.mean(), "sot_dog": sot_d.mean(),
            "offsides_fav": off_f.mean(), "offsides_dog": off_d.mean(),
            "goals_fav": civ_goals.mean(), "goals_dog": ecu_goals.mean(),
        },
    }
    return q, diag


def coin_flag(p):
    return "  <-- COIN FLIP (42-58%)" if 0.42 <= p <= 0.58 else ""


def print_report(cfg):
    t0 = time.time()
    q, diag = simulate(cfg)
    elapsed = time.time() - t0
    n = cfg["n_sims"]

    print("=" * 76)
    print(f"  {cfg['team_fav']} vs {cfg['team_dog']} — 2026 World Cup, Philadelphia")
    print(f"  {n:,} vectorized sims in {elapsed:.3f}s")
    print("=" * 76)

    f, dd, g = diag["wdl"]
    m = cfg["market_wdl"]
    print("\nGOAL MODEL (Dixon-Coles, anchored to de-vigged market)")
    print(f"  fitted lambda(CIV)={diag['lambda']:.2f}  lambda(ECU)={diag['mu']:.2f}  rho={diag['rho']:+.3f}")
    print(f"  realized W/D/L : {f:.3f} / {dd:.3f} / {g:.3f}   (CIV / draw / ECU)")
    print(f"  market  W/D/L : {m['fav']:.3f} / {m['draw']:.3f} / {m['dog']:.3f}")
    print(f"  total goals mean = {diag['total_goals_mean']:.2f};  O/U {diag['ou_line']}: "
          f"P(under)={100*diag['ou_under']:.1f}%  P(over)={100*(1-diag['ou_under']):.1f}%")

    c = diag["calib"]
    print("\nCALIBRATION (simulated full-match means vs World Cup norms)")
    print(f"  SOT combined   : {c['sot_combined']:.2f}   (norm ~{WC_NORMS['sot']})"
          f"{'   [SOT rescaled x%.3f]' % diag['sot_scale'] if diag['sot_scale'] != 1.0 else ''}")
    print(f"     CIV / ECU   : {c['sot_fav']:.2f} / {c['sot_dog']:.2f}")
    print(f"  Corners combined: {c['corners_combined']:.2f}  (norm ~{WC_NORMS['corners']})")
    print(f"  Fouls combined : {c['fouls_combined']:.2f}  (norm ~{WC_NORMS['fouls']})")
    print(f"  Cards combined : {c['cards_combined']:.2f}   (norm ~{WC_NORMS['cards']}; card-heavy ref)")
    print(f"  Offsides CIV/ECU: {c['offsides_fav']:.2f} / {c['offsides_dog']:.2f}")
    print(f"  Goals CIV/ECU  : {c['goals_fav']:.2f} / {c['goals_dog']:.2f}")

    print("\nPROP PROBABILITIES (with 95% CI)")
    print("-" * 76)
    for label, p in q.items():
        lo, hi = fmt_ci(p, n)
        print(f"  {label:<40s} {100*p:5.1f}%  [{100*lo:4.1f}, {100*hi:4.1f}]{coin_flag(p)}")
    print("-" * 76)

    print("\nSENSITIVITY — Cote d'Ivoire xG shifted +/- 0.2 (full re-anchor)")
    deltas = [-0.2, 0.0, +0.2]
    runs = {}
    for dx in deltas:
        c2 = dict(cfg)
        c2["xg_fav"] = round(cfg["xg_fav"] + dx, 2)
        runs[dx], _ = simulate(c2)
    print("  {:<40s}".format("prop") + "".join(f"{('xG '+format(cfg['xg_fav']+dx,'.2f')):>10s}" for dx in deltas))
    for label in q:
        print("  {:<40s}".format(label) + "".join(f"{100*runs[dx][label]:9.1f}%" for dx in deltas))
    print("=" * 76)


if __name__ == "__main__":
    print_report(CFG)
