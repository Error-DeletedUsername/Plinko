"""
Monte Carlo match simulator — Jump Trading Probability Cup
Match: Türkiye vs Australia, 2026 World Cup group stage.

Türkiye favourite, Australia underdog. Scoring is weighted Brier, so the
deliverable is a calibrated probability per prop; 42-58% is flagged a coin flip.

Same anchored Dixon-Coles / shared-dominance / half-split architecture as the
Qatar-Switzerland and Haiti-Scotland models. See those files for the full
architecture notes.

!!! INPUT PROVENANCE !!!
This match was specced only partially. Values tagged `# INFERRED` were NOT
supplied and are my best-guess defaults from xG + international norms, tuned to
hit the targets you gave (Q3 offsides ~41, Q4 4+ cards ~58, Q5 Kökçü ~40).
Correct any of them and re-run. Values tagged `# GIVEN` came from you.
"""

import time
import numpy as np

CFG = {
    "seed": 20260614,
    "n_sims": 200_000,

    "team_fav": "Türkiye",
    "team_dog": "Australia",

    "market_wdl": {"fav": 0.562, "draw": 0.256, "dog": 0.182},  # GIVEN (de-vigged)
    "xg_fav": 1.55,   # GIVEN
    "xg_dog": 0.95,   # GIVEN

    "poss_fav": 0.55,  # INFERRED (favourite, not overwhelming)
    "poss_dog": 0.45,  # INFERRED
    "dominance_concentration": 18.0,

    # Per-90 base rates. INFERRED unless noted.
    "sot":     {"fav": 4.5, "dog": 3.0},    # INFERRED
    "corners": {"fav": 5.0, "dog": 3.5},    # INFERRED
    "fouls":   {"fav": 10.8, "dog": 12.5},  # INFERRED (underdog fouls more)
    "cards":   {"fav": 2.0, "dog": 2.05},   # INFERRED, tuned to ref Valenzuela ~5/g
                                            #   so P(4+ cards) ~ 0.58 (GIVEN target)
    "offsides":{"fav": 1.3, "dog": 1.0},    # AUS 1.0 GIVEN; TUR INFERRED

    # Half-split fractions. Türkiye loaded slightly toward the 2nd half (GIVEN nudge).
    "split_goals_h1_fav": 0.43,  # GIVEN nudge: TUR scores a bit more in 2H
    "split_goals_h1_dog": 0.45,
    "split_cards_h1":   0.35,
    "split_corners_h1": 0.48,
    "split_sot_h1":     0.48,
    "split_offsides_h1":0.48,
    "trail_push": 1.20,

    "sot_combined_cap": 7.5,
    "sot_combined_target": 7.0,

    "grid_goals_max": 12,
    "reg_weight": 0.012,
    "lam_grid": np.round(np.arange(1.40, 3.01, 0.10), 2),
    "mu_grid":  np.round(np.arange(0.15, 1.51, 0.10), 2),
    "rho_grid": np.round(np.arange(-0.20, 0.201, 0.05), 3),
    "rho_cap": 0.20,

    # Kökçü: GIVEN full-match SOT lambda 0.7 (deep central double-pivot role, down
    # from ~1.05 at club). Q5 asks 1+ SOT in the 2ND HALF only.
    "kokcu": {"per90": 0.7, "minutes": 90, "side": "fav", "half": "2H"},
}

WC_NORMS = {"sot": 7.0, "corners": 8.9, "fouls": "22-26", "cards": 3.3}


# ---- goal model (identical engine to the other matches) ----
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
    fav_goals, dog_goals = sample_goals(P, n, rng)

    k = cfg["dominance_concentration"]
    d = rng.beta(cfg["poss_fav"] * k, (1 - cfg["poss_fav"]) * k, size=n)
    scale_fav = d / cfg["poss_fav"]
    scale_dog = (1.0 - d) / cfg["poss_dog"]

    # Goals into halves with per-team first-half fractions (TUR loaded to 2H).
    g1f = rng.binomial(fav_goals, cfg["split_goals_h1_fav"])
    g1d = rng.binomial(dog_goals, cfg["split_goals_h1_dog"])
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

    # Corners.
    cor_f_h1, cor_f_h2 = split_with_push(cfg["corners"]["fav"] * scale_fav, cfg["split_corners_h1"], cfg["trail_push"], trailing_fav, rng)
    cor_d_h1, cor_d_h2 = split_with_push(cfg["corners"]["dog"] * scale_dog, cfg["split_corners_h1"], cfg["trail_push"], trailing_dog, rng)

    # Offsides (now half-split so we can ask a 2nd-half question).
    off_f_h1, off_f_h2 = split_with_push(cfg["offsides"]["fav"] * scale_fav, cfg["split_offsides_h1"], cfg["trail_push"], trailing_fav, rng)
    off_d_h1, off_d_h2 = split_with_push(cfg["offsides"]["dog"] * scale_dog, cfg["split_offsides_h1"], cfg["trail_push"], trailing_dog, rng)
    off_f, off_d = off_f_h1 + off_f_h2, off_d_h1 + off_d_h2

    # Fouls (opponent-dominance scaled).
    fouls_f = rng.poisson(cfg["fouls"]["fav"] * scale_dog)
    fouls_d = rng.poisson(cfg["fouls"]["dog"] * scale_fav)

    # Cards (elevated by card-heavy ref).
    cards_f = rng.poisson(cfg["cards"]["fav"], size=n)
    cards_d = rng.poisson(cfg["cards"]["dog"], size=n)
    cards_total = cards_f + cards_d

    # Kökçü 2nd-half SOT (deep role -> low full-match rate; only the 2H share here).
    ks = cfg["kokcu"]
    k_2h_rate = ks["per90"] * (ks["minutes"] / 90.0) * (1.0 - cfg["split_sot_h1"]) * scale_fav
    kokcu_sot_2h = rng.poisson(k_2h_rate)

    total_goals = fav_goals + dog_goals
    goals_h2 = (fav_goals - g1f) + (dog_goals - g1d)

    # ---- 9 PROPS (Q3/Q5 wording & several others INFERRED; correct as needed) ----
    q = {}
    q["Q1 Türkiye win"]                          = np.tril(P, -1).sum()  # exact from anchor
    q["Q2 Both teams score"]                     = ((fav_goals >= 1) & (dog_goals >= 1)).mean()
    q["Q3 Australia 1+ offside in 2nd half"]     = (off_d_h2 >= 1).mean()
    q["Q4 4+ total cards in match"]              = (cards_total >= 4).mean()
    q["Q5 Kökçü 1+ SOT in 2nd half"]             = (kokcu_sot_2h >= 1).mean()
    q["Q6 Match 3+ total goals"]                 = (total_goals >= 3).mean()
    q["Q7 Australia 2+ SOT (full match)"]        = (sot_d >= 2).mean()
    q["Q8 Türkiye win to nil"]                   = ((fav_goals > dog_goals) & (dog_goals == 0)).mean()
    q["Q9 2nd half 2+ total goals"]              = (goals_h2 >= 2).mean()

    diag = {
        "lambda": lmbda, "mu": mu, "rho": rho, "wdl": wdl_from_pmf(P), "sot_scale": sot_scale,
        "calib": {
            "sot_combined": (sot_f + sot_d).mean(),
            "corners_combined": (cor_f_h1 + cor_f_h2 + cor_d_h1 + cor_d_h2).mean(),
            "fouls_combined": (fouls_f + fouls_d).mean(), "cards_combined": cards_total.mean(),
            "sot_fav": sot_f.mean(), "sot_dog": sot_d.mean(),
            "offsides_fav": off_f.mean(), "offsides_dog": off_d.mean(),
            "goals_fav": fav_goals.mean(), "goals_dog": dog_goals.mean(),
        },
        # Kökçü cross-checks so you can reconcile against your ~40 target.
        "kokcu": {
            "p_2h": (kokcu_sot_2h >= 1).mean(),
            "p_full": (rng.poisson(ks["per90"] * scale_fav) >= 1).mean(),
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

    print("=" * 74)
    print(f"  {cfg['team_fav']} vs {cfg['team_dog']} — 2026 World Cup")
    print(f"  {n:,} vectorized sims in {elapsed:.3f}s")
    print("  NOTE: many inputs INFERRED (see CFG tags) — correct against your slate")
    print("=" * 74)

    f, dd, g = diag["wdl"]
    m = cfg["market_wdl"]
    print("\nGOAL MODEL (Dixon-Coles, anchored to de-vigged market)")
    print(f"  fitted lambda(fav)={diag['lambda']:.2f}  mu(dog)={diag['mu']:.2f}  rho={diag['rho']:+.3f}")
    print(f"  realized W/D/L : {f:.3f} / {dd:.3f} / {g:.3f}")
    print(f"  market  W/D/L : {m['fav']:.3f} / {m['draw']:.3f} / {m['dog']:.3f}")

    c = diag["calib"]
    print("\nCALIBRATION (simulated full-match means vs World Cup norms)")
    print(f"  SOT combined   : {c['sot_combined']:.2f}   (norm ~{WC_NORMS['sot']})"
          f"{'   [SOT rescaled x%.3f]' % diag['sot_scale'] if diag['sot_scale'] != 1.0 else ''}")
    print(f"     fav / dog   : {c['sot_fav']:.2f} / {c['sot_dog']:.2f}")
    print(f"  Corners combined: {c['corners_combined']:.2f}  (norm ~{WC_NORMS['corners']})")
    print(f"  Fouls combined : {c['fouls_combined']:.2f}  (norm ~{WC_NORMS['fouls']})")
    print(f"  Cards combined : {c['cards_combined']:.2f}   (norm ~{WC_NORMS['cards']}; elevated for ref)")
    print(f"  Offsides fav/dog: {c['offsides_fav']:.2f} / {c['offsides_dog']:.2f}")
    print(f"  Goals fav/dog  : {c['goals_fav']:.2f} / {c['goals_dog']:.2f}")

    kk = diag["kokcu"]
    print(f"\nKÖKÇÜ CHECK : P(1+ SOT 2nd half)={100*kk['p_2h']:.1f}%   "
          f"P(1+ SOT full match)={100*kk['p_full']:.1f}%   (your target ~40)")

    print("\nPROP PROBABILITIES (with 95% CI)")
    print("-" * 74)
    for label, p in q.items():
        lo, hi = fmt_ci(p, n)
        print(f"  {label:<40s} {100*p:5.1f}%  [{100*lo:4.1f}, {100*hi:4.1f}]{coin_flag(p)}")
    print("-" * 74)

    print("\nSENSITIVITY — Türkiye xG shifted +/- 0.2 (full re-anchor)")
    deltas = [-0.2, 0.0, +0.2]
    runs = {}
    for dx in deltas:
        c2 = dict(cfg)
        c2["xg_fav"] = round(cfg["xg_fav"] + dx, 2)
        runs[dx], _ = simulate(c2)
    print("  {:<40s}".format("prop") + "".join(f"{('xG '+format(cfg['xg_fav']+dx,'.2f')):>10s}" for dx in deltas))
    for label in q:
        print("  {:<40s}".format(label) + "".join(f"{100*runs[dx][label]:9.1f}%" for dx in deltas))
    print("=" * 74)


if __name__ == "__main__":
    print_report(CFG)
