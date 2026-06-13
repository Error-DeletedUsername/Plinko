"""
Monte Carlo match simulator — Jump Trading Probability Cup
Match: Qatar vs Switzerland, 2026 World Cup group stage, Levi's Stadium.

Scoring is weighted Brier: the deliverable is a CALIBRATED probability per
prop, not extreme picks. A confident wrong answer is heavily penalized, so we
anchor hard to the de-vigged market and flag anything in the 42-58% band as a
coin flip.

Architecture (refined over previous matches):
  1. Goal model : Dixon-Coles bivariate Poisson, low-score rho correction,
                  grid-searched to the de-vigged market W/D/L with a small
                  regularizer toward the input xG. Sims are drawn directly
                  from the anchored joint PMF, so realized W/D/L matches the
                  market by construction.
  2. Dominance  : one shared Beta dominance factor per sim (mean = favourite's
                  possession share) scales corners/SOT/offsides for both teams
                  inversely.
  3. Half splits: goals 45/55, cards 35/65, corners 48/52, SOT 48/52; the team
                  trailing at HT gets a ~20% push on 2nd-half SOT & corners.
                  Goal totals stay fixed (market anchor + HT ties preserved).
  4. Calibration: print simulated full-match means vs WC norms; auto-scale SOT
                  down if combined > ~7.5 to hit ~7.0. Cards left as configured.
  5. Fouls      : each team's fouls scale with the OPPONENT's dominance (less
                  possession -> more fouls).
  6. Player props: SOT modelled as a dominance- and minutes-scaled Poisson rate;
                  P(1+) = 1 - e^-lambda. Conjunctive questions simulated jointly.

Runs 200k vectorized sims in well under a second. Re-tune everything in CFG.
"""

import time
import numpy as np

# ======================================================================
# CFG — single source of truth. Re-tune after lineups confirm.
# ======================================================================
CFG = {
    "seed": 20260613,
    "n_sims": 200_000,

    # Team labels (team A = home grid axis, team B = away grid axis)
    "team_fav": "Switzerland",
    "team_dog": "Qatar",

    # De-vigged market — Switzerland heavy favourite. We anchor to this.
    # Indexed as (fav win, draw, dog win).
    "market_wdl": {"fav": 0.80, "draw": 0.14, "dog": 0.06},

    # Match xG (regularizer target for the goal model).
    "xg_fav": 1.85,   # Swiss strong attack + elite defence (~0.3 conceded/g)
    "xg_dog": 0.55,   # Qatar feeble attack (scoreless 283 min, winless in 6)

    # Possession share -> dominance Beta mean.
    "poss_fav": 0.60,
    "poss_dog": 0.40,
    "dominance_concentration": 18.0,   # Beta(alpha+beta); higher = tighter

    # Per-90 base rates.
    "sot":     {"fav": 5.3, "dog": 1.3},   # Qatar very weak (1 effort vs ESA)
    "corners": {"fav": 5.5, "dog": 2.8},
    "fouls":   {"fav": 9.8, "dog": 12.8},  # Qatar fouls more, chasing the ball
    "cards":   {"fav": 1.3, "dog": 1.8},   # deliberately elevated; keep as-is
    "offsides":{"fav": 1.3, "dog": 1.0},

    # Half-split fractions (first-half share) and trailing-team 2nd-half push.
    "split_goals_h1":   0.45,
    "split_cards_h1":   0.35,
    "split_corners_h1": 0.48,
    "split_sot_h1":     0.48,
    "trail_push": 1.20,   # +20% on 2nd-half SOT & corners for HT-trailing side

    # Calibration guardrail for SOT.
    "sot_combined_cap": 7.5,
    "sot_combined_target": 7.0,

    # Goal-model grid search.
    "grid_goals_max": 12,
    "reg_weight": 0.012,  # small pull toward input xG
    "lam_grid": np.round(np.arange(1.40, 3.01, 0.05), 2),
    "mu_grid":  np.round(np.arange(0.15, 1.21, 0.05), 2),
    "rho_grid": np.round(np.arange(-0.20, 0.201, 0.02), 3),

    # Player props: (per-90 SOT rate, expected minutes).
    "afif":  {"per90": 0.55, "minutes": 85, "side": "dog"},  # creator, not shooter
    "xhaka": {"per90": 0.15, "minutes": 90, "side": "fav"},  # deep-lying, rarely shoots
}

WC_NORMS = {  # combined full-match norms for sanity-checking
    "sot": 7.0, "corners": 8.9, "fouls": "22-26", "cards": 3.3,
}


# ======================================================================
# 1. GOAL MODEL — Dixon-Coles bivariate Poisson, anchored to market.
# ======================================================================
def _poisson_pmf_vec(lmbda, kmax):
    k = np.arange(kmax + 1)
    # exp(k*ln(l) - l - ln(k!))
    logp = k * np.log(lmbda) - lmbda - np.cumsum(np.log(np.r_[1, np.arange(1, kmax + 1)]))
    return np.exp(logp)


def _dc_tau(lmbda, mu, rho, kmax):
    """Dixon-Coles low-score correction matrix (rows=fav goals, cols=dog goals)."""
    tau = np.ones((kmax + 1, kmax + 1))
    tau[0, 0] = 1.0 - lmbda * mu * rho
    tau[0, 1] = 1.0 + lmbda * rho
    tau[1, 0] = 1.0 + mu * rho
    tau[1, 1] = 1.0 - rho
    return tau


def dc_joint_pmf(lmbda, mu, rho, kmax):
    px = _poisson_pmf_vec(lmbda, kmax)
    py = _poisson_pmf_vec(mu, kmax)
    P = np.outer(px, py) * _dc_tau(lmbda, mu, rho, kmax)
    P = np.clip(P, 0.0, None)   # tau can push tiny cells slightly negative
    P /= P.sum()
    return P


def wdl_from_pmf(P):
    """Return (fav win, draw, dog win) from joint PMF (rows=fav, cols=dog)."""
    fav = np.tril(P, -1).sum()   # fav goals > dog goals
    draw = np.trace(P)
    dog = np.triu(P, 1).sum()
    return fav, draw, dog


def anchor_goal_model(cfg):
    kmax = cfg["grid_goals_max"]
    tgt = cfg["market_wdl"]
    reg = cfg["reg_weight"]
    best = None
    for lmbda in cfg["lam_grid"]:
        px = _poisson_pmf_vec(lmbda, kmax)
        for mu in cfg["mu_grid"]:
            py = _poisson_pmf_vec(mu, kmax)
            base = np.outer(px, py)
            for rho in cfg["rho_grid"]:
                P = base * _dc_tau(lmbda, mu, rho, kmax)
                s = P.sum()
                if s <= 0:
                    continue
                P = np.clip(P, 0.0, None)
                P /= P.sum()
                f, d, g = wdl_from_pmf(P)
                loss = ((f - tgt["fav"]) ** 2 + (d - tgt["draw"]) ** 2 +
                        (g - tgt["dog"]) ** 2 +
                        reg * ((lmbda - cfg["xg_fav"]) ** 2 + (mu - cfg["xg_dog"]) ** 2))
                if best is None or loss < best[0]:
                    best = (loss, lmbda, mu, rho)
    _, lmbda, mu, rho = best
    P = dc_joint_pmf(lmbda, mu, rho, kmax)
    return lmbda, mu, rho, P


def sample_goals(P, n, rng):
    kmax = P.shape[0] - 1
    flat = P.ravel()
    idx = rng.choice(flat.size, size=n, p=flat)
    fav_goals = idx // (kmax + 1)
    dog_goals = idx % (kmax + 1)
    return fav_goals.astype(np.int64), dog_goals.astype(np.int64)


# ======================================================================
# Helpers
# ======================================================================
def split_with_push(rate, h1_frac, push_mult, trailing_mask, rng):
    """Simulate a count split into two halves; trailing side gets a 2nd-half push.
    `rate` is a per-sim full-match expected count array."""
    h1 = rng.poisson(rate * h1_frac)
    h2_rate = rate * (1.0 - h1_frac)
    h2_rate = np.where(trailing_mask, h2_rate * push_mult, h2_rate)
    h2 = rng.poisson(h2_rate)
    return h1, h2


def fmt_ci(p, n):
    se = np.sqrt(p * (1 - p) / n)
    lo, hi = max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)
    return lo, hi


# ======================================================================
# Core simulation — returns a dict of probabilities for the 10 props.
# ======================================================================
def simulate(cfg, verbose=False):
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["n_sims"]

    # --- Goal model (anchored to market) ---
    lmbda, mu, rho, P = anchor_goal_model(cfg)
    fav_goals, dog_goals = sample_goals(P, n, rng)

    # --- Dominance factor (shared latent per sim) ---
    k = cfg["dominance_concentration"]
    a = cfg["poss_fav"] * k
    b = (1.0 - cfg["poss_fav"]) * k
    d = rng.beta(a, b, size=n)               # fav possession share this sim
    scale_fav = d / cfg["poss_fav"]          # mean ~1.0
    scale_dog = (1.0 - d) / cfg["poss_dog"]  # mean ~1.0, inverse of fav

    # --- Goals into halves (totals fixed; HT ties preserved) ---
    g1f = rng.binomial(fav_goals, cfg["split_goals_h1"])
    g1d = rng.binomial(dog_goals, cfg["split_goals_h1"])
    trailing_dog = g1f > g1d   # dog behind at HT -> chases in 2nd half
    trailing_fav = g1d > g1f

    # --- SOT (dominance-scaled), with optional calibration rescale ---
    def sot_rates(scale_factor):
        rf = cfg["sot"]["fav"] * scale_fav * scale_factor
        rd = cfg["sot"]["dog"] * scale_dog * scale_factor
        return rf, rd

    sot_scale = 1.0
    rf, rd = sot_rates(sot_scale)
    sot_combined_raw = (rf + rd).mean()
    if sot_combined_raw > cfg["sot_combined_cap"]:
        sot_scale = cfg["sot_combined_target"] / sot_combined_raw
        rf, rd = sot_rates(sot_scale)

    sot_f_h1, sot_f_h2 = split_with_push(rf, cfg["split_sot_h1"], cfg["trail_push"], trailing_fav, rng)
    sot_d_h1, sot_d_h2 = split_with_push(rd, cfg["split_sot_h1"], cfg["trail_push"], trailing_dog, rng)
    sot_f = sot_f_h1 + sot_f_h2
    sot_d = sot_d_h1 + sot_d_h2

    # --- Corners ---
    cf = cfg["corners"]["fav"] * scale_fav
    cd = cfg["corners"]["dog"] * scale_dog
    cor_f_h1, cor_f_h2 = split_with_push(cf, cfg["split_corners_h1"], cfg["trail_push"], trailing_fav, rng)
    cor_d_h1, cor_d_h2 = split_with_push(cd, cfg["split_corners_h1"], cfg["trail_push"], trailing_dog, rng)
    corners_f = cor_f_h1 + cor_f_h2
    corners_d = cor_d_h1 + cor_d_h2

    # --- Offsides (attacking VOLUME drives these; dominant side racks them up) ---
    off_f = rng.poisson(cfg["offsides"]["fav"] * scale_fav)
    off_d = rng.poisson(cfg["offsides"]["dog"] * scale_dog)

    # --- Fouls (scale with OPPONENT dominance: less possession -> more fouls) ---
    fouls_f = rng.poisson(cfg["fouls"]["fav"] * scale_dog)  # fav fouls more when dog dominates
    fouls_d = rng.poisson(cfg["fouls"]["dog"] * scale_fav)  # dog fouls more when fav dominates

    # --- Cards (halves 35/65; kept at configured rates, not forced to norm) ---
    card_f_h1 = rng.poisson(cfg["cards"]["fav"] * cfg["split_cards_h1"], size=n)
    card_f_h2 = rng.poisson(cfg["cards"]["fav"] * (1 - cfg["split_cards_h1"]), size=n)
    card_d_h1 = rng.poisson(cfg["cards"]["dog"] * cfg["split_cards_h1"], size=n)
    card_d_h2 = rng.poisson(cfg["cards"]["dog"] * (1 - cfg["split_cards_h1"]), size=n)
    cards_f = card_f_h1 + card_f_h2
    cards_d = card_d_h1 + card_d_h2

    # --- Player SOT props (dominance- and minutes-scaled Poisson) ---
    def player_sot(spec, team_scale):
        lam = spec["per90"] * (spec["minutes"] / 90.0) * team_scale
        return rng.poisson(lam)
    afif_sot = player_sot(cfg["afif"], scale_dog)    # Qatar attacker
    xhaka_sot = player_sot(cfg["xhaka"], scale_fav)  # Swiss deep midfielder

    # ------------------------------------------------------------------
    # THE 10 PROP QUESTIONS (jointly simulated where conjunctive)
    # ------------------------------------------------------------------
    q = {}
    # Q1: at HT, BOTH teams have >= 1 SOT (joint)
    q["Q1 HT: both teams 1+ SOT"] = ((sot_f_h1 >= 1) & (sot_d_h1 >= 1)).mean()
    # Q2: Switzerland (fav) caught offside 2+ times
    q["Q2 Switzerland 2+ offsides"] = (off_f >= 2).mean()
    # Q3: Switzerland 1+ card in the second half
    q["Q3 Switzerland 1+ card in 2nd half"] = (card_f_h2 >= 1).mean()
    # Q4: both teams score AND 3+ total goals (joint)
    q["Q4 both score AND 3+ total goals"] = (
        (fav_goals >= 1) & (dog_goals >= 1) & ((fav_goals + dog_goals) >= 3)).mean()
    # Q5: Qatar 2+ SOT in the second half
    q["Q5 Qatar 2+ SOT in 2nd half"] = (sot_d_h2 >= 2).mean()
    # Q6: Qatar commit more fouls than Switzerland
    q["Q6 Qatar more fouls than Switzerland"] = (fouls_d > fouls_f).mean()
    # Q7: Qatar score 1+ goal
    q["Q7 Qatar score 1+ goal"] = (dog_goals >= 1).mean()
    # Q8: Qatar 2+ SOT full match
    q["Q8 Qatar 2+ SOT full match"] = (sot_d >= 2).mean()
    # Q9: Akram Afif 1+ SOT
    q["Q9 Akram Afif 1+ SOT"] = (afif_sot >= 1).mean()
    # Q10: Granit Xhaka 1+ SOT
    q["Q10 Granit Xhaka 1+ SOT"] = (xhaka_sot >= 1).mean()

    diag = {
        "lambda": lmbda, "mu": mu, "rho": rho,
        "wdl": wdl_from_pmf(P),
        "sot_scale": sot_scale,
        "calib": {
            "sot_combined": (sot_f + sot_d).mean(),
            "corners_combined": (corners_f + corners_d).mean(),
            "fouls_combined": (fouls_f + fouls_d).mean(),
            "cards_combined": (cards_f + cards_d).mean(),
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

    print("=" * 72)
    print(f"  {cfg['team_dog']} vs {cfg['team_fav']} — 2026 World Cup, Levi's Stadium")
    print(f"  {n:,} vectorized sims in {elapsed:.3f}s")
    print("=" * 72)

    # Goal model anchor check
    f, d, g = diag["wdl"]
    m = cfg["market_wdl"]
    print("\nGOAL MODEL (Dixon-Coles, anchored to de-vigged market)")
    print(f"  fitted lambda(fav)={diag['lambda']:.2f}  mu(dog)={diag['mu']:.2f}  rho={diag['rho']:+.3f}")
    print(f"  realized W/D/L : {f:.3f} / {d:.3f} / {g:.3f}")
    print(f"  market  W/D/L : {m['fav']:.3f} / {m['draw']:.3f} / {m['dog']:.3f}")

    # Calibration block
    c = diag["calib"]
    print("\nCALIBRATION (simulated full-match means vs World Cup norms)")
    print(f"  SOT combined   : {c['sot_combined']:.2f}   (norm ~{WC_NORMS['sot']})"
          f"{'   [SOT rescaled x%.3f]' % diag['sot_scale'] if diag['sot_scale'] != 1.0 else ''}")
    print(f"     fav / dog   : {c['sot_fav']:.2f} / {c['sot_dog']:.2f}")
    print(f"  Corners combined: {c['corners_combined']:.2f}  (norm ~{WC_NORMS['corners']})")
    print(f"  Fouls combined : {c['fouls_combined']:.2f}  (norm ~{WC_NORMS['fouls']})")
    print(f"  Cards combined : {c['cards_combined']:.2f}   (norm ~{WC_NORMS['cards']}; kept elevated by design)")
    print(f"  Offsides fav/dog: {c['offsides_fav']:.2f} / {c['offsides_dog']:.2f}")
    print(f"  Goals fav/dog  : {c['goals_fav']:.2f} / {c['goals_dog']:.2f}")

    # Prop questions
    print("\nPROP PROBABILITIES (with 95% CI)")
    print("-" * 72)
    for label, p in q.items():
        lo, hi = fmt_ci(p, n)
        print(f"  {label:<40s} {100*p:5.1f}%  [{100*lo:4.1f}, {100*hi:4.1f}]{coin_flag(p)}")
    print("-" * 72)

    # Sensitivity: Switzerland xG +/- 0.2 (re-anchors goal model each time)
    print("\nSENSITIVITY — Switzerland xG shifted +/- 0.2 (full re-anchor)")
    deltas = [-0.2, 0.0, +0.2]
    runs = {}
    base_xg = cfg["xg_fav"]
    for dx in deltas:
        c2 = dict(cfg)
        c2["xg_fav"] = round(base_xg + dx, 2)
        runs[dx], _ = simulate(c2)
    header = f"  {'prop':<40s} " + " ".join(f"{base_xg+dx:+.2f}->{base_xg+dx:.2f}"[-5:].rjust(8) for dx in deltas)
    cols = "  {:<40s}".format("prop") + "".join(f"{('xG '+format(base_xg+dx,'.2f')):>10s}" for dx in deltas)
    print(cols)
    for label in q:
        row = "  {:<40s}".format(label)
        for dx in deltas:
            row += f"{100*runs[dx][label]:9.1f}%"
        print(row)
    print("=" * 72)


if __name__ == "__main__":
    print_report(CFG)
