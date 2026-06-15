"""
Monte Carlo match simulator — Jump Trading Probability Cup
Match: Sweden vs Tunisia, 2026 World Cup group stage, Estadio Monterrey.

Sweden moderate favourite; LOW-SCORING expected (Tunisia defends deep with a
threadbare attack; Sweden's defence is leaky but the Gyokeres/Isak attack is
strong). Weighted-Brier scoring: calibration over confidence; never 0/100;
42-58% flagged a coin flip.

Same anchored Dixon-Coles / shared-dominance / half-split engine as prior
matches. Notable pieces this match:
  - Q4 is a conjunction needing goal ORDER: "Sweden scored the first goal AND
    Tunisia scored in the 2nd half". Conditional on the half-by-half counts,
    arrival order is exchangeable, so P(first goal is Sweden) = SWE goals /
    total goals in the earliest half that has any goals. Simulated per sim.
  - Q6 penalty-OR-red is a true joint OR event (strict ref, VAR -> ~40%).
  - Gyokeres goal-OR-assist simulated jointly off Sweden's scored goals.
  - Skhiri SOT hammered as a fade but kept off zero.
"""

import time
import numpy as np

CFG = {
    "seed": 20260615,
    "n_sims": 200_000,

    "team_fav": "Sweden",
    "team_dog": "Tunisia",

    "market_wdl": {"fav": 0.50, "draw": 0.28, "dog": 0.22},
    "ou_line": 2.5,   # under favoured (~43% for 3+)

    "xg_fav": 1.28,   # SWE
    "xg_dog": 1.05,   # TUN  (total ~2.33, slightly under the 2.5 line)

    "poss_fav": 0.55,  # SWE attacks more
    "poss_dog": 0.45,  # TUN sits deep, counters
    "dominance_concentration": 18.0,

    "sot":     {"fav": 4.0, "dog": 2.8},    # GIVEN, combined ~6.8 (TUN weak attack)
    "corners": {"fav": 5.0, "dog": 3.9},    # INFERRED (SWE attacks more)
    "fouls":   {"fav": 10.8, "dog": 11.6},  # TUN deeper -> a touch more fouls
    "cards":   {"fav": 2.5, "dog": 2.9},    # strict ref ~5.4 combined yellows
    "offsides":{"fav": 1.6, "dog": 1.0},    # INFERRED (no offsides question)

    "split_goals_h1":   0.42,
    "split_cards_h1":   0.35,
    "split_corners_h1": 0.48,
    "split_sot_h1":     0.48,
    "trail_push": 1.20,

    "sot_combined_cap": 7.5,
    "sot_combined_target": 7.0,

    # Q6: penalty OR red. Independent Poisson per match, OR'd per sim, tuned so
    # combined ~0.40 (base ~0.35, elevated by strict ref Falcon Perez + VAR).
    "pen_lambda": 0.28,   # P(1+ pen) ~ 24%
    "red_lambda": 0.23,   # P(1+ red) ~ 20%
    "pen_red_target": 0.40,

    "grid_goals_max": 12,
    "reg_weight": 0.012,
    "lam_grid": np.round(np.arange(0.70, 2.01, 0.05), 2),
    "mu_grid":  np.round(np.arange(0.60, 1.81, 0.05), 2),
    "rho_grid": np.round(np.arange(-0.12, 0.121, 0.02), 3),  # let it land near 0
    "rho_cap": 0.12,

    # Player props.
    "gyokeres_involve_per_goal": 0.42,   # G+A ~0.45 (goal-dominated; minimal assists)
    "gyokeres_target": 0.45,
    "skhiri": {"exp_sot": 0.13, "side": "dog"},   # 1 SOT all season -> ~12%
    "skhiri_target": 0.12,
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
    swe_goals, tun_goals = sample_goals(P, n, rng)

    k = cfg["dominance_concentration"]
    d = rng.beta(cfg["poss_fav"] * k, (1 - cfg["poss_fav"]) * k, size=n)
    scale_fav = d / cfg["poss_fav"]          # SWE
    scale_dog = (1.0 - d) / cfg["poss_dog"]  # TUN

    g1f = rng.binomial(swe_goals, cfg["split_goals_h1"])
    g1d = rng.binomial(tun_goals, cfg["split_goals_h1"])
    g2f, g2d = swe_goals - g1f, tun_goals - g1d
    trailing_dog = g1f > g1d
    trailing_fav = g1d > g1f

    # First goal of the game (exchangeable order within the earliest scoring half).
    swe_first = np.zeros(n, dtype=bool)
    u = rng.random(n)
    h1g, h2g = g1f + g1d, g2f + g2d
    m1 = h1g > 0
    swe_first[m1] = u[m1] < (g1f[m1] / h1g[m1])
    m2 = (~m1) & (h2g > 0)
    swe_first[m2] = u[m2] < (g2f[m2] / h2g[m2])

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

    fouls_f = rng.poisson(cfg["fouls"]["fav"] * scale_dog)   # SWE
    fouls_d = rng.poisson(cfg["fouls"]["dog"] * scale_fav)   # TUN

    cards_f = rng.poisson(cfg["cards"]["fav"], size=n)
    cards_d = rng.poisson(cfg["cards"]["dog"], size=n)

    pens = rng.poisson(cfg["pen_lambda"], size=n)
    reds = rng.poisson(cfg["red_lambda"], size=n)
    pen_or_red = (pens >= 1) | (reds >= 1)

    gyokeres_inv = rng.binomial(swe_goals, cfg["gyokeres_involve_per_goal"]) >= 1
    skhiri_sot = rng.poisson(cfg["skhiri"]["exp_sot"] * scale_dog)

    total_goals = swe_goals + tun_goals

    # ---- 10 PROPS ----
    q = {}
    q["Q1 Tunisia more corners than SWE at HT"] = (cor_d_h1 > cor_f_h1).mean()
    q["Q2 Sweden more fouls than Tunisia"]      = (fouls_f > fouls_d).mean()
    q["Q3 2H more goals than 1H (strict)"]      = (h2g > h1g).mean()
    q["Q4 SWE scores first AND TUN scores 2H"]  = (swe_first & (g2d >= 1)).mean()
    q["Q5 Tunisia more SOT than SWE in 2H"]     = (sot_d_h2 > sot_f_h2).mean()
    q["Q6 Penalty OR red card in match"]        = pen_or_red.mean()
    q["Q7 Sweden win"]                          = np.tril(P, -1).sum()
    q["Q8 3+ total goals"]                      = (total_goals >= 3).mean()
    q["Q9 Gyokeres score or assist"]            = gyokeres_inv.mean()
    q["Q10 Skhiri 1+ SOT (fade)"]               = (skhiri_sot >= 1).mean()

    diag = {
        "lambda": lmbda, "mu": mu, "rho": rho, "wdl": wdl_from_pmf(P), "sot_scale": sot_scale,
        "ou_line": cfg["ou_line"], "total_goals_mean": total_goals.mean(),
        "p_3plus": (total_goals >= 3).mean(),
        "pen_red": {"combined": pen_or_red.mean(), "pen": (pens >= 1).mean(), "red": (reds >= 1).mean()},
        "calib": {
            "sot_combined": (sot_f + sot_d).mean(),
            "corners_combined": (cor_f_h1 + cor_f_h2 + cor_d_h1 + cor_d_h2).mean(),
            "fouls_combined": (fouls_f + fouls_d).mean(), "cards_combined": (cards_f + cards_d).mean(),
            "sot_fav": sot_f.mean(), "sot_dog": sot_d.mean(),
            "offsides_fav": off_f.mean(), "offsides_dog": off_d.mean(),
            "goals_fav": swe_goals.mean(), "goals_dog": tun_goals.mean(),
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
    print(f"  {cfg['team_fav']} vs {cfg['team_dog']} — 2026 World Cup, Estadio Monterrey")
    print(f"  {n:,} vectorized sims in {elapsed:.3f}s")
    print("=" * 76)

    f, dd, g = diag["wdl"]
    m = cfg["market_wdl"]
    print("\nGOAL MODEL (Dixon-Coles, anchored to de-vigged market)")
    print(f"  fitted lambda(SWE)={diag['lambda']:.2f}  lambda(TUN)={diag['mu']:.2f}  rho={diag['rho']:+.3f}")
    print(f"  realized W/D/L : {f:.3f} / {dd:.3f} / {g:.3f}   (SWE / draw / TUN)")
    print(f"  market  W/D/L : {m['fav']:.3f} / {m['draw']:.3f} / {m['dog']:.3f}")
    print(f"  total goals mean = {diag['total_goals_mean']:.2f};  O/U {diag['ou_line']}: "
          f"P(3+)={100*diag['p_3plus']:.1f}%  (market ~43%)")

    c = diag["calib"]
    print("\nCALIBRATION (simulated full-match means vs World Cup norms)")
    print(f"  SOT combined   : {c['sot_combined']:.2f}   (norm ~{WC_NORMS['sot']})"
          f"{'   [SOT rescaled x%.3f]' % diag['sot_scale'] if diag['sot_scale'] != 1.0 else ''}")
    print(f"     SWE / TUN   : {c['sot_fav']:.2f} / {c['sot_dog']:.2f}")
    print(f"  Corners combined: {c['corners_combined']:.2f}  (norm ~{WC_NORMS['corners']})")
    print(f"  Fouls combined : {c['fouls_combined']:.2f}  (norm ~{WC_NORMS['fouls']})")
    print(f"  Cards combined : {c['cards_combined']:.2f}   (norm ~{WC_NORMS['cards']}; strict ref)")
    print(f"  Offsides SWE/TUN: {c['offsides_fav']:.2f} / {c['offsides_dog']:.2f}")
    print(f"  Goals SWE/TUN  : {c['goals_fav']:.2f} / {c['goals_dog']:.2f}")

    pr = diag["pen_red"]
    print(f"\nQ6 OR-EVENT CHECK : P(1+ pen)={100*pr['pen']:.1f}%  P(1+ red)={100*pr['red']:.1f}%"
          f"  ->  combined OR = {100*pr['combined']:.1f}%  (target ~{100*cfg['pen_red_target']:.0f}%)")

    print("\nPROP PROBABILITIES (with 95% CI)")
    print("-" * 76)
    for label, p in q.items():
        lo, hi = fmt_ci(p, n)
        print(f"  {label:<40s} {100*p:5.1f}%  [{100*lo:4.1f}, {100*hi:4.1f}]{coin_flag(p)}")
    print("-" * 76)

    print("\nSENSITIVITY — Sweden xG shifted +/- 0.2 (full re-anchor)")
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
