"""
Monte Carlo match simulator — Jump Trading Probability Cup
Match: Haiti vs Scotland, 2026 World Cup group stage, Gillette Stadium.

Scotland favourite but NOT overwhelming; Haiti a weak underdog WITH a real
scoring floor (Scotland's defence is leaky). Scoring is weighted Brier, so the
deliverable is a calibrated probability per prop, not extreme picks. Anything
in 42-58% is flagged a coin flip.

Architecture (refined over previous matches):
  1. Goal model : Dixon-Coles bivariate Poisson, low-score rho correction,
                  grid-searched to the de-vigged market W/D/L with a small xG
                  regularizer; sims drawn from the joint PMF so realized W/D/L
                  matches the market by construction.
  2. Dominance  : one shared Beta dominance factor per sim (mean = favourite's
                  possession) scaling SOT/corners/offsides inversely per team.
  3. Half splits: goals 45/55, cards 35/65, corners 48/52, SOT 48/52; the team
                  trailing at HT gets a ~20% push on 2nd-half SOT & corners.
                  Goal totals stay fixed (market anchor + HT ties preserved).
  4. Calibration: print simulated full-match means vs WC norms; auto-scale SOT
                  down if combined > ~7.5 to hit ~7.0. Cards kept as configured.
  5. Fouls      : each team's fouls scale with the OPPONENT's dominance (less
                  possession -> more fouls); lean to the underdog fouling more.
  6. Player/OR  : player SOT = dominance/minutes-scaled Poisson, P(1+)=1-e^-l.
                  A chasing sub gets the trailing push. Q4 penalty-OR-red is a
                  true joint OR event per sim, never a marginal multiply.

Player-modelling lessons applied:
  - Fade ONLY deep-lying midfielders. McTominay is a box-crashing shooter
    (3.16 shots/90 at Napoli) -> moderate-high, NOT a fade.
  - Fade weak-team attack GENTLY: even underdogs have a floor, higher vs a
    non-elite defence. Scotland's defence is beatable, so Haiti's floor is real.
  - A benched player: model expected minutes ~20 (may not appear) -> low P(1+).

Runs 200k vectorized sims in well under a second. Re-tune everything in CFG.
"""

import time
import numpy as np

# ======================================================================
# CFG — single source of truth. Re-tune after lineups confirm.
# ======================================================================
CFG = {
    "seed": 20260614,
    "n_sims": 200_000,

    "team_fav": "Scotland",
    "team_dog": "Haiti",

    # De-vigged market — Scotland favourite but not overwhelming. Anchor here.
    "market_wdl": {"fav": 0.62, "draw": 0.22, "dog": 0.16},

    # Match xG (regularizer target for the goal model).
    "xg_fav": 1.75,
    "xg_dog": 0.95,   # real underdog floor vs a leaky Scotland defence

    # Possession share -> dominance Beta mean.
    "poss_fav": 0.58,
    "poss_dog": 0.42,
    "dominance_concentration": 18.0,

    # Per-90 base rates.
    "sot":     {"fav": 4.8, "dog": 2.6},
    "corners": {"fav": 5.1, "dog": 3.2},
    "fouls":   {"fav": 10.5, "dog": 13.0},  # underdog fouls more, chasing
    "cards":   {"fav": 1.5, "dog": 2.0},
    "offsides":{"fav": 1.4, "dog": 1.1},

    # Half-split fractions (first-half share) and trailing-team 2nd-half push.
    "split_goals_h1":   0.45,
    "split_cards_h1":   0.35,
    "split_corners_h1": 0.48,
    "split_sot_h1":     0.48,
    "trail_push": 1.20,

    # Calibration guardrail for SOT.
    "sot_combined_cap": 7.5,
    "sot_combined_target": 7.0,

    # Q4: penalty-OR-red-card. Independent Poisson rates per match, OR'd per sim.
    # Tuned so combined P(1+ pen OR 1+ red) ~ 0.45 (base ~0.40, elevated this WC).
    "pen_lambda": 0.40,   # P(1+ pen)  = 1-e^-0.40 ~ 0.33
    "red_lambda": 0.20,   # P(1+ red)  = 1-e^-0.20 ~ 0.18
    "pen_red_target": 0.45,

    # Goal-model grid search (coarse-to-fine).
    "grid_goals_max": 12,
    "reg_weight": 0.012,
    "lam_grid": np.round(np.arange(1.40, 3.01, 0.10), 2),
    "mu_grid":  np.round(np.arange(0.15, 1.51, 0.10), 2),
    "rho_grid": np.round(np.arange(-0.20, 0.201, 0.05), 3),
    "rho_cap": 0.20,

    # Player SOT props. chase=True applies the trailing-team push (a sub coming
    # on to chase enters a stretched end-to-end phase).
    "nazon": {"per90": 0.75, "minutes": 20, "side": "dog", "chase": True},   # BENCHED late-sub
    "mctominay": {"per90": 1.1, "minutes": 90, "side": "fav", "chase": False},  # box-crashing shooter, STARTS
}

WC_NORMS = {"sot": 7.0, "corners": 8.9, "fouls": "22-26", "cards": 3.3}


# ======================================================================
# 1. GOAL MODEL — Dixon-Coles bivariate Poisson, anchored to market.
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

    # --- Goal model (anchored) ---
    lmbda, mu, rho, P = anchor_goal_model(cfg)
    fav_goals, dog_goals = sample_goals(P, n, rng)

    # --- Dominance ---
    k = cfg["dominance_concentration"]
    d = rng.beta(cfg["poss_fav"] * k, (1 - cfg["poss_fav"]) * k, size=n)
    scale_fav = d / cfg["poss_fav"]
    scale_dog = (1.0 - d) / cfg["poss_dog"]

    # --- Goals into halves (totals fixed; HT ties preserved) ---
    g1f = rng.binomial(fav_goals, cfg["split_goals_h1"])
    g1d = rng.binomial(dog_goals, cfg["split_goals_h1"])
    trailing_dog = g1f > g1d
    trailing_fav = g1d > g1f

    # --- SOT (calibration rescale measured on the REALIZED total, incl. push) ---
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

    # --- Corners ---
    cor_f_h1, cor_f_h2 = split_with_push(cfg["corners"]["fav"] * scale_fav, cfg["split_corners_h1"], cfg["trail_push"], trailing_fav, rng)
    cor_d_h1, cor_d_h2 = split_with_push(cfg["corners"]["dog"] * scale_dog, cfg["split_corners_h1"], cfg["trail_push"], trailing_dog, rng)
    corners_f, corners_d = cor_f_h1 + cor_f_h2, cor_d_h1 + cor_d_h2

    # --- Offsides (attacking VOLUME drives these) ---
    off_f = rng.poisson(cfg["offsides"]["fav"] * scale_fav)
    off_d = rng.poisson(cfg["offsides"]["dog"] * scale_dog)

    # --- Fouls (scale with OPPONENT dominance: less possession -> more fouls) ---
    fouls_f = rng.poisson(cfg["fouls"]["fav"] * scale_dog)
    fouls_d = rng.poisson(cfg["fouls"]["dog"] * scale_fav)

    # --- Cards (kept at configured rates) ---
    cards_f = rng.poisson(cfg["cards"]["fav"], size=n)
    cards_d = rng.poisson(cfg["cards"]["dog"], size=n)

    # --- Q4: penalty OR red card (true joint OR per sim) ---
    pens = rng.poisson(cfg["pen_lambda"], size=n)
    reds = rng.poisson(cfg["red_lambda"], size=n)
    pen_or_red = (pens >= 1) | (reds >= 1)

    # --- Player SOT props ---
    def player_sot(spec):
        scale = scale_fav if spec["side"] == "fav" else scale_dog
        trailing = trailing_fav if spec["side"] == "fav" else trailing_dog
        lam = spec["per90"] * (spec["minutes"] / 90.0) * scale
        if spec["chase"]:
            lam = lam * np.where(trailing, cfg["trail_push"], 1.0)
        return rng.poisson(lam)
    nazon_sot = player_sot(cfg["nazon"])
    mctom_sot = player_sot(cfg["mctominay"])

    # --- Goal-derived helpers ---
    total_goals = fav_goals + dog_goals
    goals_h2 = (fav_goals - g1f) + (dog_goals - g1d)

    # ------------------------------------------------------------------
    # THE 10 PROPS
    # ------------------------------------------------------------------
    q = {}
    q["Q1 Haiti more fouls than Scotland"]        = (fouls_d > fouls_f).mean()
    q["Q2 Haiti more corners than Scot (2nd H)"]  = (cor_d_h2 > cor_f_h2).mean()
    q["Q3 Haiti more SOT than Scot (2nd half)"]   = (sot_d_h2 > sot_f_h2).mean()
    q["Q4 penalty OR red card in match"]          = pen_or_red.mean()
    q["Q5 Haiti score 1+ goal"]                   = (dog_goals >= 1).mean()
    q["Q6 Match tied at halftime"]                = (g1f == g1d).mean()
    q["Q7 2 or fewer total goals"]                = (total_goals <= 2).mean()
    q["Q8 2nd half 2+ total goals"]               = (goals_h2 >= 2).mean()
    q["Q9 Duckens Nazon 1+ SOT (benched)"]        = (nazon_sot >= 1).mean()
    q["Q10 Scott McTominay 1+ SOT (starts)"]      = (mctom_sot >= 1).mean()

    diag = {
        "lambda": lmbda, "mu": mu, "rho": rho, "wdl": wdl_from_pmf(P), "sot_scale": sot_scale,
        "pen_red": {"combined": pen_or_red.mean(), "pen": (pens >= 1).mean(), "red": (reds >= 1).mean()},
        "ties": {"q2": (cor_d_h2 == cor_f_h2).mean(), "q3": (sot_d_h2 == sot_f_h2).mean()},
        "calib": {
            "sot_combined": (sot_f + sot_d).mean(), "corners_combined": (corners_f + corners_d).mean(),
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
    print(f"  {cfg['team_dog']} vs {cfg['team_fav']} — 2026 World Cup, Gillette Stadium")
    print(f"  {n:,} vectorized sims in {elapsed:.3f}s")
    print("=" * 74)

    f, d, g = diag["wdl"]
    m = cfg["market_wdl"]
    print("\nGOAL MODEL (Dixon-Coles, anchored to de-vigged market)")
    print(f"  fitted lambda(fav)={diag['lambda']:.2f}  mu(dog)={diag['mu']:.2f}  rho={diag['rho']:+.3f}")
    print(f"  realized W/D/L : {f:.3f} / {d:.3f} / {g:.3f}")
    print(f"  market  W/D/L : {m['fav']:.3f} / {m['draw']:.3f} / {m['dog']:.3f}")

    c = diag["calib"]
    print("\nCALIBRATION (simulated full-match means vs World Cup norms)")
    print(f"  SOT combined   : {c['sot_combined']:.2f}   (norm ~{WC_NORMS['sot']})"
          f"{'   [SOT rescaled x%.3f]' % diag['sot_scale'] if diag['sot_scale'] != 1.0 else ''}")
    print(f"     fav / dog   : {c['sot_fav']:.2f} / {c['sot_dog']:.2f}")
    print(f"  Corners combined: {c['corners_combined']:.2f}  (norm ~{WC_NORMS['corners']})")
    print(f"  Fouls combined : {c['fouls_combined']:.2f}  (norm ~{WC_NORMS['fouls']})")
    print(f"  Cards combined : {c['cards_combined']:.2f}   (norm ~{WC_NORMS['cards']})")
    print(f"  Offsides fav/dog: {c['offsides_fav']:.2f} / {c['offsides_dog']:.2f}")
    print(f"  Goals fav/dog  : {c['goals_fav']:.2f} / {c['goals_dog']:.2f}")

    pr = diag["pen_red"]
    print("\nQ4 OR-EVENT CHECK (penalty OR red card; modelled jointly)")
    print(f"  P(1+ penalty)={100*pr['pen']:.1f}%   P(1+ red)={100*pr['red']:.1f}%"
          f"   ->  combined OR = {100*pr['combined']:.1f}%  (target ~{100*cfg['pen_red_target']:.0f}%)")

    print("\nPROP PROBABILITIES (with 95% CI)")
    print("-" * 74)
    for label, p in q.items():
        lo, hi = fmt_ci(p, n)
        print(f"  {label:<42s} {100*p:5.1f}%  [{100*lo:4.1f}, {100*hi:4.1f}]{coin_flag(p)}")
    print("-" * 74)
    print(f"  (Q2/Q3 ties excluded from 'more than': 2nd-half corner ties {100*diag['ties']['q2']:.1f}%,"
          f" SOT ties {100*diag['ties']['q3']:.1f}%)")

    print("\nSENSITIVITY — Scotland xG shifted +/- 0.2 (full re-anchor)")
    deltas = [-0.2, 0.0, +0.2]
    runs = {}
    for dx in deltas:
        c2 = dict(cfg)
        c2["xg_fav"] = round(cfg["xg_fav"] + dx, 2)
        runs[dx], _ = simulate(c2)
    print("  {:<42s}".format("prop") + "".join(f"{('xG '+format(cfg['xg_fav']+dx,'.2f')):>10s}" for dx in deltas))
    for label in q:
        print("  {:<42s}".format(label) + "".join(f"{100*runs[dx][label]:9.1f}%" for dx in deltas))
    print("=" * 74)


if __name__ == "__main__":
    print_report(CFG)
