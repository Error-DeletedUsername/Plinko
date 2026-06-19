"""
Monte Carlo match simulator — Jump Trading Probability Cup
Match: Mexico vs South Korea, 2026 World Cup, Estadio Akron Guadalajara
(open-air, 1,566m altitude, dry evening). 9-QUESTION CARD.

Mexico favoured (home, Guadalajara crowd) and lightly possession-dominant; Korea
content on the ball, a 2nd-half side. Draw is LIVE (both happy with a point) ->
LOW event profile. No heavy favourite, so game-state EASING is OFF (the trailing
team's chase push stays). Two match-specific knobs:

  REF CARD FACTOR (Tejera, very strict ~5.0 Y/g): a 1.30x multiplier on card
  rates — the single biggest driver of Q8 (4+ cards).

  ALTITUDE LATE FATIGUE: a SMALL symmetric 2nd-half nudge — goals already skew 2H
  via the 0.42/0.58 split (Korea's goals are 2H-heavy), plus a small 2H card bump.
  Kept small: Korea acclimatized in SLC and already played MD1 at Akron.

Q2 = "Mexico scores FIRST and Korea scores in the 2nd half" — ordered joint event
(first-goal identity from exchangeable arrival order), never a marginal product.
200k vectorized sims, <1s.
"""

import time
import copy
import numpy as np

CFG = {
    "seed": 20260615,
    "n_sims": 200_000,

    "team_fav": "Mexico",
    "team_dog": "South Korea",
    "match_profile": "LOW",

    "host_team": None, "host_type": None,
    "host_bump_by_type": {"host": 0.35, "home": 0.18, None: 0.0},
    "anchor_includes_host": True,

    "market_wdl": {"fav": 0.47, "draw": 0.29, "dog": 0.24},
    "ou_line": 2.5,
    "market_p_over": 0.40,
    "over_in_grid": True,
    "x12_weight": 2.5,
    "reconcile_threshold": 0.02,

    "xg_fav": 1.35,   # MEX
    "xg_dog": 0.95,   # KOR
    "dog_lambda_floor": 0.55,

    "poss_fav": 0.55, "poss_dog": 0.45,
    "dominance_concentration": 18.0,

    "heat_pct": 0.0,            # dry evening, no weather dampen
    "heat_affects_goals": False,

    "sot":     {"fav": 4.5, "dog": 3.5},   # LOW profile (no SOT question; calibration only)
    "corners": {"fav": 5.2, "dog": 4.7},   # Korea content on ball -> corners fairly even
    "fouls":   {"fav": 12.0, "dog": 13.0}, # (no foul question; calibration only)
    "fouls_use_dominance": True,
    "cards":   {"fav": 1.65, "dog": 1.70}, # PRE-ref base; x REF_CARD_FACTOR below
    "ref_card_factor": 1.30,               # Tejera very strict — biggest Q8 driver
    "offsides":{"fav": 1.55, "dog": 1.00},

    "split_goals_h1":   0.42,   # Korea's goals skew 2H
    "split_cards_h1":   0.35,
    "split_corners_h1": 0.48,
    "split_sot_h1":     0.48,
    "split_offsides_h1":0.50,
    "trail_push": 1.20,
    "card_2h_fatigue_bump": 1.10,   # SMALL altitude/late-game 2H card nudge

    "h2_ease_enabled": False,   # tight game, no heavy favourite to ease
    "h2_ease_factor": 0.825,
    "h2_ease_lead": 2,

    # LOW profile: trim combined SOT a touch below the ~7 norm.
    "sot_combined_cap": 7.2,
    "sot_combined_target": 6.8,

    "pen_lambda": 0.30, "red_lambda": 0.12,   # calibration only (no pen/red question)

    "grid_goals_max": 12,
    "reg_weight": 0.006,
    "lam_grid": np.round(np.arange(0.80, 2.21, 0.10), 2),
    "mu_grid":  np.round(np.arange(0.50, 1.61, 0.10), 2),
    "rho_grid": np.round(np.arange(-0.14, 0.121, 0.02), 3),
    "rho_cap": 0.12,

    # Q9 Son Heung-min (KOR) score-or-assist, two scenarios.
    "son": {"start_involve": 0.47, "bench_involve": 0.135},   # P ~0.36 / ~0.12
}

PRIORS = {
    "Q1 Mexico 2+ offsides": 0.45,
    "Q2 Mexico scores first AND Korea scores 2H": 0.18,
    "Q3 Korea more corners than Mexico": 0.40,
    "Q4 Mexico win": 0.47,
    "Q5 Mexico lead at halftime": 0.32,
    "Q6 3+ total goals": 0.40,
    "Q7 Korea score in 2nd half": 0.45,
    "Q8 4+ total cards": 0.68,
    "Q9a Son score or assist — STARTS": 0.36,
    "Q9b Son score or assist — BENCHED": 0.12,
}

WC_NORMS = {"sot": 7.0, "corners": 8.9, "fouls": "22-26", "cards": 3.3}


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


def p_over_from_pmf(P):
    le2 = P[0, 0] + P[0, 1] + P[1, 0] + P[2, 0] + P[0, 2] + P[1, 1]
    return 1.0 - le2


def _grid_search(lam_arr, mu_arr, rho_arr, cfg, xg_f, xg_d, px_cache):
    kmax, tgt, reg = cfg["grid_goals_max"], cfg["market_wdl"], cfg["reg_weight"]
    rho_cap, floor = cfg["rho_cap"], cfg["dog_lambda_floor"]
    w12, w_over = cfg["x12_weight"], (1.0 if cfg["over_in_grid"] else 0.0)
    over_tgt = cfg["market_p_over"]
    best = None
    for lmbda in lam_arr:
        px = px_cache.get(lmbda)
        if px is None:
            px = px_cache[lmbda] = _poisson_pmf_vec(lmbda, kmax)
        reg_lam = reg * (lmbda - xg_f) ** 2
        for mu in mu_arr:
            if mu < floor:
                continue
            py = px_cache.get(-mu)
            if py is None:
                py = px_cache[-mu] = _poisson_pmf_vec(mu, kmax)
            base = np.outer(px, py)
            reg_mu = reg * (mu - xg_d) ** 2
            for rho in rho_arr:
                if rho > rho_cap:
                    continue
                P = np.clip(base * _dc_tau(lmbda, mu, rho, kmax), 0.0, None)
                P /= P.sum()
                f, dd, g = wdl_from_pmf(P)
                loss = (w12 * ((f - tgt["fav"]) ** 2 + (dd - tgt["draw"]) ** 2 + (g - tgt["dog"]) ** 2) +
                        w_over * (p_over_from_pmf(P) - over_tgt) ** 2 + reg_lam + reg_mu)
                if best is None or loss < best[0]:
                    best = (loss, lmbda, mu, rho)
    return best


def anchor_goal_model(cfg):
    kmax, floor = cfg["grid_goals_max"], cfg["dog_lambda_floor"]
    xg_f, xg_d = cfg["xg_fav"], cfg["xg_dog"]
    px_cache = {}
    _, lmbda, mu, rho = _grid_search(cfg["lam_grid"], cfg["mu_grid"], cfg["rho_grid"], cfg, xg_f, xg_d, px_cache)
    lam_f = np.round(np.arange(max(0.05, lmbda - 0.06), lmbda + 0.061, 0.0125), 4)
    mu_f = np.round(np.arange(max(floor, mu - 0.06), mu + 0.061, 0.0125), 4)
    rho_f = np.round(np.arange(rho - 0.02, rho + 0.0201, 0.005), 4)
    _, lmbda, mu, rho = _grid_search(lam_f, mu_f, rho_f, cfg, xg_f, xg_d, px_cache)
    return lmbda, mu, rho, dc_joint_pmf(lmbda, mu, rho, kmax)


def sample_goals(P, n, rng):
    kmax = P.shape[0] - 1
    idx = rng.choice(P.size, size=n, p=P.ravel())
    return (idx // (kmax + 1)).astype(np.int64), (idx % (kmax + 1)).astype(np.int64)


def split_mult(rate, h1_frac, h2_mult, rng):
    h1 = rng.poisson(rate * h1_frac)
    h2 = rng.poisson(rate * (1.0 - h1_frac) * h2_mult)
    return h1, h2


def fmt_ci(p, n):
    se = np.sqrt(p * (1 - p) / n)
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def simulate(cfg):
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["n_sims"]

    lmbda, mu, rho, P = anchor_goal_model(cfg)
    mex_goals, kor_goals = sample_goals(P, n, rng)   # fav=MEX, dog=KOR

    k = cfg["dominance_concentration"]
    d = rng.beta(cfg["poss_fav"] * k, (1 - cfg["poss_fav"]) * k, size=n)
    scale_fav = d / cfg["poss_fav"]          # MEX
    scale_dog = (1.0 - d) / cfg["poss_dog"]  # KOR

    g1f = rng.binomial(mex_goals, cfg["split_goals_h1"])
    g1d = rng.binomial(kor_goals, cfg["split_goals_h1"])
    g2f, g2d = mex_goals - g1f, kor_goals - g1d
    trailing_dog = g1f > g1d
    trailing_fav = g1d > g1f

    # First goal of the game (Q2): earliest-scoring-half exchangeable order.
    mex_first = np.zeros(n, dtype=bool)
    u = rng.random(n)
    h1g, h2g = g1f + g1d, g2f + g2d
    m1 = h1g > 0
    mex_first[m1] = u[m1] < (g1f[m1] / h1g[m1])
    m2 = (~m1) & (h2g > 0)
    mex_first[m2] = u[m2] < (g2f[m2] / h2g[m2])

    mex_leads_2plus = (g1f - g1d) >= cfg["h2_ease_lead"]
    ease = cfg["h2_ease_factor"] if cfg["h2_ease_enabled"] else 1.0
    mult_fav_2h = np.where(trailing_fav, cfg["trail_push"], np.where(mex_leads_2plus, ease, 1.0))
    mult_dog_2h = np.where(trailing_dog, cfg["trail_push"], 1.0)

    def build_sot(sf):
        fh1, fh2 = split_mult(cfg["sot"]["fav"] * scale_fav * sf, cfg["split_sot_h1"], mult_fav_2h, rng)
        dh1, dh2 = split_mult(cfg["sot"]["dog"] * scale_dog * sf, cfg["split_sot_h1"], mult_dog_2h, rng)
        return fh1, fh2, dh1, dh2
    sot_scale = 1.0
    sot_f_h1, sot_f_h2, sot_d_h1, sot_d_h2 = build_sot(sot_scale)
    combined = (sot_f_h1 + sot_f_h2 + sot_d_h1 + sot_d_h2).mean()
    if combined > cfg["sot_combined_cap"]:
        sot_scale = cfg["sot_combined_target"] / combined
        sot_f_h1, sot_f_h2, sot_d_h1, sot_d_h2 = build_sot(sot_scale)
    sot_f, sot_d = sot_f_h1 + sot_f_h2, sot_d_h1 + sot_d_h2

    cor_f_h1, cor_f_h2 = split_mult(cfg["corners"]["fav"] * scale_fav, cfg["split_corners_h1"], mult_fav_2h, rng)
    cor_d_h1, cor_d_h2 = split_mult(cfg["corners"]["dog"] * scale_dog, cfg["split_corners_h1"], mult_dog_2h, rng)
    corners_f, corners_d = cor_f_h1 + cor_f_h2, cor_d_h1 + cor_d_h2

    off_f = rng.poisson(cfg["offsides"]["fav"] * scale_fav)
    off_d = rng.poisson(cfg["offsides"]["dog"] * scale_dog)

    fouls_f = rng.poisson(cfg["fouls"]["fav"] * scale_dog)
    fouls_d = rng.poisson(cfg["fouls"]["dog"] * scale_fav)

    # Cards: x ref factor, with a small 2H altitude/fatigue bump.
    rcf, bump = cfg["ref_card_factor"], cfg["card_2h_fatigue_bump"]
    cards_f = (rng.poisson(cfg["cards"]["fav"] * rcf * cfg["split_cards_h1"], size=n) +
               rng.poisson(cfg["cards"]["fav"] * rcf * (1 - cfg["split_cards_h1"]) * bump, size=n))
    cards_d = (rng.poisson(cfg["cards"]["dog"] * rcf * cfg["split_cards_h1"], size=n) +
               rng.poisson(cfg["cards"]["dog"] * rcf * (1 - cfg["split_cards_h1"]) * bump, size=n))
    cards_total = cards_f + cards_d

    pens = rng.poisson(cfg["pen_lambda"], size=n)
    reds = rng.poisson(cfg["red_lambda"], size=n)

    son_start = (rng.binomial(kor_goals, cfg["son"]["start_involve"]) >= 1).mean()
    son_bench = (rng.binomial(kor_goals, cfg["son"]["bench_involve"]) >= 1).mean()

    total_goals = mex_goals + kor_goals

    q = {}
    q["Q1 Mexico 2+ offsides"]                  = (off_f >= 2).mean()
    q["Q2 Mexico scores first AND Korea scores 2H"] = (mex_first & (g2d >= 1)).mean()
    q["Q3 Korea more corners than Mexico"]      = (corners_d > corners_f).mean()
    q["Q4 Mexico win"]                          = np.tril(P, -1).sum()
    q["Q5 Mexico lead at halftime"]            = (g1f > g1d).mean()
    q["Q6 3+ total goals"]                      = (total_goals >= 3).mean()
    q["Q7 Korea score in 2nd half"]            = (g2d >= 1).mean()
    q["Q8 4+ total cards"]                      = (cards_total >= 4).mean()
    q["Q9a Son score or assist — STARTS"]       = son_start
    q["Q9b Son score or assist — BENCHED"]      = son_bench

    diag = {
        "lambda": lmbda, "mu": mu, "rho": rho, "wdl": wdl_from_pmf(P), "sot_scale": sot_scale,
        "p_over_model": p_over_from_pmf(P), "total_goals_mean": total_goals.mean(),
        "p_kor_score": (kor_goals >= 1).mean(), "p_mex_score": (mex_goals >= 1).mean(),
        "calib": {
            "sot_combined": (sot_f + sot_d).mean(),
            "corners_combined": (corners_f + corners_d).mean(),
            "fouls_combined": (fouls_f + fouls_d).mean(), "cards_combined": cards_total.mean(),
            "offsides_fav": off_f.mean(), "offsides_dog": off_d.mean(),
            "goals_fav": mex_goals.mean(), "goals_dog": kor_goals.mean(),
            "pen_red_or": ((pens >= 1) | (reds >= 1)).mean(),
        },
    }
    return q, diag


def flag(p):
    return "COIN" if 0.42 <= p <= 0.58 else "    "


def print_report(cfg):
    t0 = time.time()
    q, diag = simulate(cfg)
    elapsed = time.time() - t0
    n = cfg["n_sims"]

    print("=" * 88)
    print(f"  {cfg['match_profile']}  |  {cfg['team_fav']} (FAV, home) vs {cfg['team_dog']} — Estadio Akron, Guadalajara (1566m)")
    print(f"  {n:,} sims in {elapsed:.3f}s   |   host OFF, easing OFF, ref card factor {cfg['ref_card_factor']}x")
    print("=" * 88)

    f, dd, g = diag["wdl"]
    m = cfg["market_wdl"]
    print("\nGOAL MODEL — DUAL ANCHOR (Dixon-Coles)")
    print(f"  fitted lambda(MEX)={diag['lambda']:.2f}  lambda(KOR)={diag['mu']:.2f}  rho={diag['rho']:+.3f}"
          f"   (KOR floored at {cfg['dog_lambda_floor']:.2f})")
    print(f"  realized W/D/L : {f:.3f} / {dd:.3f} / {g:.3f}   vs market {m['fav']:.3f}/{m['draw']:.3f}/{m['dog']:.3f}"
          f"   (max err {max(abs(f-m['fav']),abs(dd-m['draw']),abs(g-m['dog']))*100:.1f}pp)")
    over_err = diag["p_over_model"] - cfg["market_p_over"]
    print(f"  Over {cfg['ou_line']}: model {100*diag['p_over_model']:.1f}% vs market {100*cfg['market_p_over']:.1f}%"
          f"  -> residual {100*over_err:+.1f}pp", end="")
    print("   ** >2pp: deferred to 1X2 **" if abs(over_err) > cfg["reconcile_threshold"] else "   (both anchors within threshold)")
    print(f"  total goals mean = {diag['total_goals_mean']:.2f}   P(MEX score)={100*diag['p_mex_score']:.1f}%  P(KOR score)={100*diag['p_kor_score']:.1f}%")

    c = diag["calib"]
    print("\nCALIBRATION (full-match means)")
    print(f"  SOT {c['sot_combined']:.2f}{'  [rescaled x%.3f]' % diag['sot_scale'] if diag['sot_scale'] != 1.0 else ''}"
          f"   Corners {c['corners_combined']:.2f}   Fouls {c['fouls_combined']:.2f}"
          f"   Cards {c['cards_combined']:.2f} (ref {cfg['ref_card_factor']}x)   Off MEX/KOR {c['offsides_fav']:.2f}/{c['offsides_dog']:.2f}")

    print("\n  Q#  question                                       sim P   target  flag")
    print("  " + "-" * 74)
    recon = []
    for label, p in q.items():
        prior = PRIORS[label]
        print(f"  {label:<48s}  {100*p:5.1f}%  {100*prior:5.1f}%   {flag(p)}")
        if abs(p - prior) > 0.05:
            recon.append((label, p, prior))

    if recon:
        print("\n  RECONCILIATION (sim diverges >5pp from target):")
        for label, p, prior in recon:
            print(f"   - {label}: {100*p:.1f}% vs {100*prior:.0f}% ({100*(p-prior):+.1f}pp) — {_why(label, diag)}")

    print("\nSENSITIVITY — Mexico (fav) xG +/-0.2 (easing already OFF; chase push ON)")
    cols = []
    for dx in (-0.2, 0.0, +0.2):
        r, _ = simulate({**copy.deepcopy(cfg), "xg_fav": round(cfg["xg_fav"] + dx, 2)})
        cols.append((f"xG{cfg['xg_fav']+dx:+.2f}", r))
    print("  {:<48s}".format("prop") + "".join(f"{name:>10s}" for name, _ in cols))
    for label in ("Q4 Mexico win", "Q5 Mexico lead at halftime", "Q6 3+ total goals",
                  "Q2 Mexico scores first AND Korea scores 2H", "Q7 Korea score in 2nd half"):
        print("  {:<48s}".format(label) + "".join(f"{100*r[label]:9.1f}%" for _, r in cols))
    print("=" * 88)


def _why(label, diag):
    if "Q2" in label or "Q7" in label:
        return f"Korea's real floor (P(KOR score)={100*diag['p_kor_score']:.0f}%, 2H-weighted) lifts this above the gut"
    if "Q6" in label or "3+" in label:
        return f"tied to dual-anchor total (mean {diag['total_goals_mean']:.2f}); 1X2 won"
    if "Q8" in label or "cards" in label:
        return "ref card factor 1.30x drives total cards; tune cards base or factor"
    return "given input-rate driven; see CFG"


if __name__ == "__main__":
    print_report(CFG)
