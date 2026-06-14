"""
Monte Carlo match simulator — Jump Trading Probability Cup
Match: Netherlands vs Japan, 2026 World Cup group stage, Dallas.

TIGHT line — Netherlands only modest favourites. Weighted-Brier scoring, so the
deliverable is a calibrated probability per prop; never 0/100, and 42-58% is
flagged a coin flip.

Same anchored Dixon-Coles / shared-dominance / half-split architecture as the
prior matches. New piece: Kubo's goal-OR-assist prop is simulated jointly off
Japan's *scored* goals (each Japan goal has a chance of being a Kubo
involvement), so it correctly can't fire when Japan is shut out — never a
marginal multiply.
"""

import time
import numpy as np

# ======================================================================
# CFG — single source of truth.
# ======================================================================
CFG = {
    "seed": 20260614,
    "n_sims": 200_000,

    "team_fav": "Netherlands",
    "team_dog": "Japan",

    # De-vigged market anchor (trusted over raw xG).
    "market_wdl": {"fav": 0.47, "draw": 0.28, "dog": 0.25},
    "ou_line": 2.5,   # slight under lean expected

    # Input xG (regularizer target). Total 2.5.
    "xg_fav": 1.45,
    "xg_dog": 1.05,

    # Possession -> dominance Beta mean. NED dominate; Japan sit deep & counter.
    "poss_fav": 0.58,
    "poss_dog": 0.42,
    "dominance_concentration": 18.0,

    # Per-90 base rates.
    "sot":     {"fav": 4.1, "dog": 3.1},   # combined ~7.2 base (target ~7-7.5)
    "corners": {"fav": 5.2, "dog": 3.6},
    "fouls":   {"fav": 11.0, "dog": 11.8}, # roughly even, Japan a touch higher
    "cards":   {"fav": 2.0, "dog": 2.0},   # disciplined, near-tie (ref Elfath ~4/g)
    "offsides":{"fav": 2.0, "dog": 1.0},   # NED high line vs deep block

    # Half splits + trailing-team 2H push.
    "split_goals_h1":   0.45,
    "split_cards_h1":   0.35,
    "split_corners_h1": 0.48,
    "split_sot_h1":     0.48,
    "trail_push": 1.20,

    "sot_combined_cap": 7.5,
    "sot_combined_target": 7.0,

    # Goal-model grid search (coarse-to-fine).
    "grid_goals_max": 12,
    "reg_weight": 0.012,
    "lam_grid": np.round(np.arange(1.00, 2.51, 0.10), 2),
    "mu_grid":  np.round(np.arange(0.60, 1.81, 0.10), 2),
    "rho_grid": np.round(np.arange(-0.10, 0.201, 0.05), 3),  # preserve draw w/ slightly +rho
    "rho_cap": 0.20,

    # Player props.
    "gakpo": {"per90": 1.3, "minutes": 90, "side": "fav"},   # high-volume, NOT faded
    # Kubo: per Japan goal, prob that goal is a Kubo involvement (his goal or his
    # assist). Tuned so P(goal or assist) ~ 0.35 given Japan's anchored scoring.
    "kubo_involve_per_goal": 0.43,
    "kubo_target": 0.35,
}

WC_NORMS = {"sot": 7.0, "corners": 8.9, "fouls": "22-26", "cards": 3.3}


# ======================================================================
# Goal model
# ======================================================================
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
    lam_f = np.round(np.arange(lmbda - 0.05, lmbda + 0.051, 0.0125), 4)
    mu_f = np.round(np.arange(max(0.05, mu - 0.05), mu + 0.051, 0.0125), 4)
    rho_f = np.round(np.arange(rho - 0.03, rho + 0.0301, 0.005), 4)
    _, lmbda, mu, rho = _grid_search(lam_f, mu_f, rho_f, cfg, px_cache)
    return lmbda, mu, rho, dc_joint_pmf(lmbda, mu, rho, kmax)


def sample_goals(P, n, rng):
    kmax = P.shape[0] - 1
    idx = rng.choice(P.size, size=n, p=P.ravel())
    return (idx // (kmax + 1)).astype(np.int64), (idx % (kmax + 1)).astype(np.int64)


# ======================================================================
# Helpers
# ======================================================================
def split_with_push(rate, h1_frac, push_mult, trailing_mask, rng):
    h1 = rng.poisson(rate * h1_frac)
    h2_rate = np.where(trailing_mask, rate * (1.0 - h1_frac) * push_mult, rate * (1.0 - h1_frac))
    return h1, rng.poisson(h2_rate)


def fmt_ci(p, n):
    se = np.sqrt(p * (1 - p) / n)
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


# ======================================================================
# Core simulation
# ======================================================================
def simulate(cfg):
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["n_sims"]

    lmbda, mu, rho, P = anchor_goal_model(cfg)
    fav_goals, dog_goals = sample_goals(P, n, rng)

    k = cfg["dominance_concentration"]
    d = rng.beta(cfg["poss_fav"] * k, (1 - cfg["poss_fav"]) * k, size=n)
    scale_fav = d / cfg["poss_fav"]
    scale_dog = (1.0 - d) / cfg["poss_dog"]

    g1f = rng.binomial(fav_goals, cfg["split_goals_h1"])
    g1d = rng.binomial(dog_goals, cfg["split_goals_h1"])
    trailing_dog = g1f > g1d
    trailing_fav = g1d > g1f

    # SOT with realized-total calibration rescale.
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

    off_f = rng.poisson(cfg["offsides"]["fav"] * scale_fav)
    off_d = rng.poisson(cfg["offsides"]["dog"] * scale_dog)

    fouls_f = rng.poisson(cfg["fouls"]["fav"] * scale_dog)
    fouls_d = rng.poisson(cfg["fouls"]["dog"] * scale_fav)

    cards_f = rng.poisson(cfg["cards"]["fav"], size=n)
    cards_d = rng.poisson(cfg["cards"]["dog"], size=n)

    # Gakpo SOT.
    gk = cfg["gakpo"]
    gakpo_sot = rng.poisson(gk["per90"] * (gk["minutes"] / 90.0) * scale_fav)

    # Kubo goal-OR-assist: each Japan scored goal is a Kubo involvement w.p. p.
    # Involved iff >=1 of Japan's goals is his -> joint with Japan actually scoring.
    kubo_involved = rng.binomial(dog_goals, cfg["kubo_involve_per_goal"]) >= 1

    total_goals = fav_goals + dog_goals
    jpn_goals_h2 = dog_goals - g1d

    # ---- 10 PROPS ----
    q = {}
    q["Q1 Both teams 1+ SOT in 2H"]          = ((sot_f_h2 >= 1) & (sot_d_h2 >= 1)).mean()
    q["Q2 Japan more fouls than NED"]        = (fouls_d > fouls_f).mean()
    q["Q3 Netherlands 2+ offsides"]          = (off_f >= 2).mean()
    q["Q4 Both score AND 3+ total goals"]    = ((fav_goals >= 1) & (dog_goals >= 1) & (total_goals >= 3)).mean()
    q["Q5 Japan more cards than NED"]        = (cards_d > cards_f).mean()
    q["Q6 Netherlands win"]                  = np.tril(P, -1).sum()
    q["Q7 Japan score in 2H"]                = (jpn_goals_h2 >= 1).mean()
    q["Q8 8+ total combined SOT"]            = ((sot_f + sot_d) >= 8).mean()
    q["Q9 Gakpo 1+ SOT"]                     = (gakpo_sot >= 1).mean()
    q["Q10 Kubo score or assist"]            = kubo_involved.mean()

    diag = {
        "lambda": lmbda, "mu": mu, "rho": rho, "wdl": wdl_from_pmf(P), "sot_scale": sot_scale,
        "ou_under": (total_goals <= 2).mean(), "ou_line": cfg["ou_line"],
        "calib": {
            "sot_combined": (sot_f + sot_d).mean(),
            "corners_combined": (cor_f_h1 + cor_f_h2 + cor_d_h1 + cor_d_h2).mean(),
            "fouls_combined": (fouls_f + fouls_d).mean(), "cards_combined": (cards_f + cards_d).mean(),
            "sot_fav": sot_f.mean(), "sot_dog": sot_d.mean(),
            "offsides_fav": off_f.mean(), "offsides_dog": off_d.mean(),
            "goals_fav": fav_goals.mean(), "goals_dog": dog_goals.mean(),
        },
    }
    return q, diag


# ======================================================================
# Reporting
# ======================================================================
def coin_flag(p):
    return "  <-- COIN FLIP (42-58%)" if 0.42 <= p <= 0.58 else ""


def print_report(cfg):
    t0 = time.time()
    q, diag = simulate(cfg)
    elapsed = time.time() - t0
    n = cfg["n_sims"]

    print("=" * 74)
    print(f"  {cfg['team_fav']} vs {cfg['team_dog']} — 2026 World Cup, Dallas")
    print(f"  {n:,} vectorized sims in {elapsed:.3f}s")
    print("=" * 74)

    f, dd, g = diag["wdl"]
    m = cfg["market_wdl"]
    print("\nGOAL MODEL (Dixon-Coles, anchored to de-vigged market)")
    print(f"  fitted lambda(NED)={diag['lambda']:.2f}  lambda(JPN)={diag['mu']:.2f}  rho={diag['rho']:+.3f}")
    print(f"  realized W/D/L : {f:.3f} / {dd:.3f} / {g:.3f}")
    print(f"  market  W/D/L : {m['fav']:.3f} / {m['draw']:.3f} / {m['dog']:.3f}")
    print(f"  O/U {diag['ou_line']}: P(under) = {100*diag['ou_under']:.1f}%  (slight under lean expected)")

    c = diag["calib"]
    print("\nCALIBRATION (simulated full-match means vs World Cup norms)")
    print(f"  SOT combined   : {c['sot_combined']:.2f}   (norm ~{WC_NORMS['sot']})"
          f"{'   [SOT rescaled x%.3f]' % diag['sot_scale'] if diag['sot_scale'] != 1.0 else ''}")
    print(f"     NED / JPN   : {c['sot_fav']:.2f} / {c['sot_dog']:.2f}")
    print(f"  Corners combined: {c['corners_combined']:.2f}  (norm ~{WC_NORMS['corners']})")
    print(f"  Fouls combined : {c['fouls_combined']:.2f}  (norm ~{WC_NORMS['fouls']})")
    print(f"  Cards combined : {c['cards_combined']:.2f}   (norm ~{WC_NORMS['cards']})")
    print(f"  Offsides NED/JPN: {c['offsides_fav']:.2f} / {c['offsides_dog']:.2f}")
    print(f"  Goals NED/JPN  : {c['goals_fav']:.2f} / {c['goals_dog']:.2f}")

    print("\nPROP PROBABILITIES (with 95% CI)")
    print("-" * 74)
    for label, p in q.items():
        lo, hi = fmt_ci(p, n)
        print(f"  {label:<38s} {100*p:5.1f}%  [{100*lo:4.1f}, {100*hi:4.1f}]{coin_flag(p)}")
    print("-" * 74)

    print("\nSENSITIVITY — Netherlands xG shifted +/- 0.2 (full re-anchor)")
    deltas = [-0.2, 0.0, +0.2]
    runs = {}
    for dx in deltas:
        c2 = dict(cfg)
        c2["xg_fav"] = round(cfg["xg_fav"] + dx, 2)
        runs[dx], _ = simulate(c2)
    print("  {:<38s}".format("prop") + "".join(f"{('xG '+format(cfg['xg_fav']+dx,'.2f')):>10s}" for dx in deltas))
    for label in q:
        print("  {:<38s}".format(label) + "".join(f"{100*runs[dx][label]:9.1f}%" for dx in deltas))
    print("=" * 74)


if __name__ == "__main__":
    print_report(CFG)
