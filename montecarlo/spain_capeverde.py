"""
Monte Carlo match simulator — Jump Trading Probability Cup
Match: Spain vs Cape Verde, 2026 World Cup group stage, Mercedes-Benz Atlanta
(domed, weather not a factor). Spain a MASSIVE favourite (Euro 2024 champs,
unbeaten in 30); Cape Verde a debutant minnow that defends deep and counters.
Neither side is a host -> host term OFF.

Weighted-Brier scoring: calibration over confidence; never 0/100; 42-58% flagged.
Same anchored Dixon-Coles / dominance / half-split engine as prior matches, with
two match-specific pieces:

  1. 2H GAME-STATE EASING: if Spain leads by 2+ at HT (very likely), Spain eases
     off -> a ~17.5% REDUCTION to its 2nd-half SOT and corner intensity. This is
     the main risk to Q2/Q4/Q9 and is a toggle (h2_ease_enabled) so the
     sensitivity table can show how fragile those props are to game state.
  2. UNDERDOG SCORING FLOOR: the grid search floors Cape Verde's lambda at 0.40
     so the extreme-favourite anchor cannot zero out their real ~33-38% chance to
     score.

Player props output TWO lineup scenarios each (Olmo starts/sub, Mendes starts/
benched); goal-or-assist is simulated jointly off Spain's scored goals.
"""

import time
import copy
import numpy as np

CFG = {
    "seed": 20260615,
    "n_sims": 200_000,

    "team_fav": "Spain",
    "team_dog": "Cape Verde",

    # Host term OFF (neither side is a host).
    "host_team": None,
    "host_type": None,
    "host_bump_by_type": {"host": 0.35, "home": 0.18, None: 0.0},
    "anchor_includes_host": True,

    "market_wdl": {"fav": 0.89, "draw": 0.075, "dog": 0.035},
    "ou_line": 3.5,

    "xg_fav": 2.70,   # ESP
    "xg_dog": 0.45,   # CPV (real scoring floor; do NOT zero out)
    "dog_lambda_floor": 0.40,

    "poss_fav": 0.68,  # Spain dominates the ball massively
    "poss_dog": 0.32,
    "dominance_concentration": 18.0,

    "sot":     {"fav": 5.5, "dog": 1.7},   # CPV suppressed by lack of possession
    "corners": {"fav": 6.5, "dog": 2.0},
    "fouls":   {"fav": 8.0, "dog": 12.0},  # CPV chases/defends -> more fouls
    "cards":   {"fav": 1.7, "dog": 1.6},   # below-median ref ~3.3 combined
    "offsides":{"fav": 1.5, "dog": 1.0},

    "split_goals_h1":   0.45,
    "split_cards_h1":   0.35,
    "split_corners_h1": 0.48,
    "split_sot_h1":     0.48,
    "trail_push": 1.20,

    # 2H game-state easing for the leading favourite.
    "h2_ease_enabled": True,
    "h2_ease_factor": 0.825,    # -17.5% on Spain 2H SOT & corners when cruising
    "h2_ease_lead": 2,          # leads by >= this at HT -> ease

    "sot_combined_cap": 7.5,
    "sot_combined_target": 7.0,

    # Q5 penalty-OR-red. Tuned to ~44% (lopsided game lifts pens; below-median ref
    # keeps reds modest -> P(red) ~15% matches the ref).
    "pen_lambda": 0.42,
    "red_lambda": 0.16,
    "pen_red_target": 0.44,

    "grid_goals_max": 14,
    "reg_weight": 0.004,   # light: at this extreme line let the 1X2 anchor drive lambda high
    "lam_grid": np.round(np.arange(1.60, 4.01, 0.05), 2),
    "mu_grid":  np.round(np.arange(0.40, 1.21, 0.05), 2),
    "rho_grid": np.round(np.arange(-0.10, 0.161, 0.02), 3),
    "rho_cap": 0.16,

    # Player props (two scenarios each). Goal-or-assist involve = per-Spain-goal prob.
    "olmo": {"start_involve": 0.24, "sub_involve": 0.077},   # P ~0.50 starts / ~0.20 sub
    "mendes": {"start_sot": 0.25, "bench_sot": 0.083},       # P ~0.22 starts / ~0.08 benched
}

WC_NORMS = {"sot": 7.0, "corners": 8.9, "fouls": "22-26", "cards": 3.3}


# ---- goal model engine (with underdog floor in the grids) ----
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
    floor = cfg["dog_lambda_floor"]
    best = None
    for lmbda in lam_arr:
        px = px_cache.get(lmbda)
        if px is None:
            px = px_cache[lmbda] = _poisson_pmf_vec(lmbda, kmax)
        reg_lam = reg * (lmbda - cfg["xg_fav"]) ** 2
        for mu in mu_arr:
            if mu < floor:
                continue
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
    floor = cfg["dog_lambda_floor"]
    px_cache = {}
    _, lmbda, mu, rho = _grid_search(cfg["lam_grid"], cfg["mu_grid"], cfg["rho_grid"], cfg, px_cache)
    lam_f = np.round(np.arange(max(0.05, lmbda - 0.03), lmbda + 0.031, 0.0125), 4)
    mu_f = np.round(np.arange(max(floor, mu - 0.03), mu + 0.031, 0.0125), 4)
    rho_f = np.round(np.arange(rho - 0.02, rho + 0.0201, 0.005), 4)
    _, lmbda, mu, rho = _grid_search(lam_f, mu_f, rho_f, cfg, px_cache)
    return lmbda, mu, rho, dc_joint_pmf(lmbda, mu, rho, kmax)


def sample_goals(P, n, rng):
    kmax = P.shape[0] - 1
    idx = rng.choice(P.size, size=n, p=P.ravel())
    return (idx // (kmax + 1)).astype(np.int64), (idx % (kmax + 1)).astype(np.int64)


def split_mult(rate, h1_frac, h2_mult, rng):
    """Half split with an explicit per-sim 2nd-half intensity multiplier."""
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
    esp_goals, cpv_goals = sample_goals(P, n, rng)

    k = cfg["dominance_concentration"]
    d = rng.beta(cfg["poss_fav"] * k, (1 - cfg["poss_fav"]) * k, size=n)
    scale_fav = d / cfg["poss_fav"]
    scale_dog = (1.0 - d) / cfg["poss_dog"]

    g1f = rng.binomial(esp_goals, cfg["split_goals_h1"])
    g1d = rng.binomial(cpv_goals, cfg["split_goals_h1"])
    g2f, g2d = esp_goals - g1f, cpv_goals - g1d
    trailing_dog = g1f > g1d
    trailing_fav = g1d > g1f

    # 2nd-half intensity multipliers (game state).
    esp_leads_2plus = (g1f - g1d) >= cfg["h2_ease_lead"]
    ease = cfg["h2_ease_factor"] if cfg["h2_ease_enabled"] else 1.0
    mult_fav_2h = np.where(trailing_fav, cfg["trail_push"],
                           np.where(esp_leads_2plus, ease, 1.0))
    mult_dog_2h = np.where(trailing_dog, cfg["trail_push"], 1.0)

    # SOT (game-state aware) with realized-total calibration rescale.
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

    cards_f = rng.poisson(cfg["cards"]["fav"], size=n)
    cards_d = rng.poisson(cfg["cards"]["dog"], size=n)

    pens = rng.poisson(cfg["pen_lambda"], size=n)
    reds = rng.poisson(cfg["red_lambda"], size=n)
    pen_or_red = (pens >= 1) | (reds >= 1)

    # Player props (two scenarios each), joint off Spain's scored goals.
    olmo_start = (rng.binomial(esp_goals, cfg["olmo"]["start_involve"]) >= 1).mean()
    olmo_sub = (rng.binomial(esp_goals, cfg["olmo"]["sub_involve"]) >= 1).mean()
    mendes_start = (rng.poisson(cfg["mendes"]["start_sot"] * scale_dog) >= 1).mean()
    mendes_bench = (rng.poisson(cfg["mendes"]["bench_sot"] * scale_dog) >= 1).mean()

    total_goals = esp_goals + cpv_goals
    both_score = (esp_goals >= 1) & (cpv_goals >= 1)

    q = {}
    q["Q1 Cape Verde 2+ offsides"]            = (off_d >= 2).mean()
    q["Q2 Spain more corners than CPV (2H)"]  = (cor_f_h2 > cor_d_h2).mean()
    q["Q3 Cape Verde more fouls than Spain"]  = (fouls_d > fouls_f).mean()
    q["Q4 Spain more SOT than CPV (2H)"]      = (sot_f_h2 > sot_d_h2).mean()
    q["Q5 Penalty OR red card"]               = pen_or_red.mean()
    q["Q6 Both score AND 3+ total goals"]     = (both_score & (total_goals >= 3)).mean()
    q["Q7a Olmo G+A — STARTS"]                = olmo_start
    q["Q7b Olmo G+A — SUB"]                   = olmo_sub
    q["Q8 Spain win"]                         = np.tril(P, -1).sum()
    q["Q9 8+ total combined SOT"]             = ((sot_f + sot_d) >= 8).mean()
    q["Q10a Ryan Mendes 1+ SOT — STARTS"]     = mendes_start
    q["Q10b Ryan Mendes 1+ SOT — BENCHED"]    = mendes_bench

    diag = {
        "lambda": lmbda, "mu": mu, "rho": rho, "wdl": wdl_from_pmf(P), "sot_scale": sot_scale,
        "total_goals_mean": total_goals.mean(), "p_4plus": (total_goals >= 4).mean(),
        "p_cpv_score": (cpv_goals >= 1).mean(), "p_esp_lead2_ht": esp_leads_2plus.mean(),
        "pen_red": {"combined": pen_or_red.mean(), "pen": (pens >= 1).mean(), "red": (reds >= 1).mean()},
        "calib": {
            "sot_combined": (sot_f + sot_d).mean(), "corners_combined": (corners_f + corners_d).mean(),
            "fouls_combined": (fouls_f + fouls_d).mean(), "cards_combined": (cards_f + cards_d).mean(),
            "sot_fav": sot_f.mean(), "sot_dog": sot_d.mean(),
            "offsides_fav": off_f.mean(), "offsides_dog": off_d.mean(),
            "goals_fav": esp_goals.mean(), "goals_dog": cpv_goals.mean(),
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

    print("=" * 80)
    print(f"  {cfg['team_fav']} vs {cfg['team_dog']} — 2026 World Cup, Mercedes-Benz Atlanta (domed)")
    print(f"  {n:,} vectorized sims in {elapsed:.3f}s   |   host term OFF")
    print("=" * 80)

    f, dd, g = diag["wdl"]
    m = cfg["market_wdl"]
    print("\nGOAL MODEL (Dixon-Coles, anchored to de-vigged market)")
    print(f"  fitted lambda(ESP)={diag['lambda']:.2f}  lambda(CPV)={diag['mu']:.2f}  rho={diag['rho']:+.3f}"
          f"   (CPV floored at {cfg['dog_lambda_floor']:.2f})")
    print(f"  realized W/D/L : {f:.3f} / {dd:.3f} / {g:.3f}   (ESP / draw / CPV)")
    print(f"  market  W/D/L : {m['fav']:.3f} / {m['draw']:.3f} / {m['dog']:.3f}")
    print(f"  total goals mean = {diag['total_goals_mean']:.2f};  O/U {cfg['ou_line']}: "
          f"P(4+)={100*diag['p_4plus']:.1f}%   |   P(CPV score)={100*diag['p_cpv_score']:.1f}%   "
          f"P(ESP lead 2+ @HT)={100*diag['p_esp_lead2_ht']:.1f}%")

    c = diag["calib"]
    print("\nCALIBRATION (simulated full-match means vs World Cup norms)")
    print(f"  SOT combined   : {c['sot_combined']:.2f}   (norm ~{WC_NORMS['sot']})"
          f"{'   [rescaled x%.3f]' % diag['sot_scale'] if diag['sot_scale'] != 1.0 else ''}")
    print(f"     ESP / CPV   : {c['sot_fav']:.2f} / {c['sot_dog']:.2f}")
    print(f"  Corners combined: {c['corners_combined']:.2f}  (norm ~{WC_NORMS['corners']})")
    print(f"  Fouls combined : {c['fouls_combined']:.2f}  (norm ~{WC_NORMS['fouls']})")
    print(f"  Cards combined : {c['cards_combined']:.2f}   (norm ~{WC_NORMS['cards']}; below-median ref)")
    print(f"  Offsides ESP/CPV: {c['offsides_fav']:.2f} / {c['offsides_dog']:.2f}")
    print(f"  Goals ESP/CPV  : {c['goals_fav']:.2f} / {c['goals_dog']:.2f}")

    pr = diag["pen_red"]
    print(f"\nQ5 OR-EVENT CHECK : P(1+ pen)={100*pr['pen']:.1f}%  P(1+ red)={100*pr['red']:.1f}%"
          f"  ->  combined OR = {100*pr['combined']:.1f}%  (target ~{100*cfg['pen_red_target']:.0f}%)")

    print("\nPROP PROBABILITIES (with 95% CI)  — Q7/Q10 show BOTH lineup scenarios")
    print("-" * 80)
    for label, p in q.items():
        lo, hi = fmt_ci(p, n)
        print(f"  {label:<40s} {100*p:5.1f}%  [{100*lo:4.1f}, {100*hi:4.1f}]{coin_flag(p)}")
    print("-" * 80)

    # Sensitivity: Spain xG +/-0.3 (ease ON) plus an ease-OFF column at base xG.
    print("\nSENSITIVITY — Spain xG +/-0.3 (ease ON) and 2H-easing toggled OFF (base xG)")
    cols = []
    for dx in (-0.3, 0.0, +0.3):
        c2 = {**copy.deepcopy(cfg), "xg_fav": round(cfg["xg_fav"] + dx, 2)}
        r, _ = simulate(c2)
        cols.append((f"xG{cfg['xg_fav']+dx:+.1f}", r))
    c_off = {**copy.deepcopy(cfg), "h2_ease_enabled": False}
    r_off, _ = simulate(c_off)
    cols.append(("easeOFF", r_off))
    print("  {:<40s}".format("prop") + "".join(f"{name:>10s}" for name, _ in cols))
    for label in q:
        print("  {:<40s}".format(label) + "".join(f"{100*r[label]:9.1f}%" for _, r in cols))
    print("  (compare last two columns on Q2/Q4/Q9 to read game-state fragility)")
    print("=" * 80)


if __name__ == "__main__":
    print_report(CFG)
