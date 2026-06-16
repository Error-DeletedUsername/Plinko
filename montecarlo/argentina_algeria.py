"""
Monte Carlo match simulator — Jump Trading Probability Cup
Match: Argentina vs Algeria, 2026 World Cup Group J, Kansas City (hot/humid).

Argentina favoured & possession-dominant; Algeria a REAL AFCON attacker (Amoura)
that counters and chases — EVENT_PROFILE = MODERATE, so the unders are NOT
defaulted down and Algeria's attack keeps a genuine floor.

Engine ported from prior matches (anchored Dixon-Coles + shared dominance +
half-splits + game-state easing + heat-events-only). Two pieces specific here:

  DUAL ANCHOR: the grid search fits BOTH the de-vigged 1X2 AND P(total>2.5) to
  the market, each targeted within ~1pp. If they conflict by >2pp the 1X2 wins
  (it is weighted heavier) and the residual on the total is reported.

  SEQUENCED CONJUNCTION (Q4): "Argentina score the FIRST goal AND Algeria score
  in the 2nd half" is simulated as an ordered joint event — first-goal identity
  comes from exchangeable arrival order within the earliest scoring half — never
  a product of marginals.

200k vectorized sims, < ~1s. All inputs in CFG.
"""

import time
import copy
import numpy as np

CFG = {
    "seed": 20260615,
    "n_sims": 200_000,

    "team_fav": "Argentina",
    "team_dog": "Algeria",
    "match_profile": "MODERATE",

    "host_team": None, "host_type": None,
    "host_bump_by_type": {"host": 0.35, "home": 0.18, None: 0.0},
    "anchor_includes_host": True,

    # market_wdl["fav"] = Argentina win; ["dog"] = Algeria win.
    "market_wdl": {"fav": 0.67, "draw": 0.20, "dog": 0.13},
    "ou_line": 2.5,
    "market_p_over": 0.49,           # de-vigged Over 2.5 (coin flip) — a HARD anchor here
    "over_in_grid": True,            # fit the total jointly with the 1X2
    "x12_weight": 2.5,               # 1X2 weighted heavier -> defer to 1X2 on conflict
    "reconcile_threshold": 0.02,

    "xg_fav": 1.70,   # ARG seed
    "xg_dog": 0.88,   # ALG seed (real attacker)
    "dog_lambda_floor": 0.50,

    "poss_fav": 0.61, "poss_dog": 0.39,
    "dominance_concentration": 18.0,

    "heat_pct": 0.05,            # mild KC heat dampener on event rates only
    "heat_affects_goals": False,

    "sot":     {"fav": 5.4, "dog": 2.7},   # ARG 5-6, ALG 2-3 (real floor)
    "corners": {"fav": 6.0, "dog": 3.5},
    "fouls":   {"fav": 11.0, "dog": 13.0}, # ALG lower possession -> fouls more
    "cards":   {"fav": 1.9, "dog": 2.1},   # ref Marciniak ~4.1/g combined
    "offsides":{"fav": 2.0, "dog": 1.0},   # ARG high line -> offside-prone

    "split_goals_h1":   0.45,
    "split_cards_h1":   0.35,
    "split_corners_h1": 0.48,
    "split_sot_h1":     0.48,
    "split_offsides_h1":0.50,
    "trail_push": 1.20,
    "card_2h_heat_bump": 1.10,

    "h2_ease_enabled": True,
    "h2_ease_factor": 0.825,
    "h2_ease_lead": 2,

    # MODERATE game: let combined SOT sit a touch above the ~7 norm (do not bury).
    "sot_combined_cap": 8.2,
    "sot_combined_target": 7.6,

    # Q6 pen-OR-red, ref Marciniak: P(pen)=.278, P(red)=.113 -> OR ~ .36.
    "pen_lambda": 0.326,   # 1-e^-0.326 = .278
    "red_lambda": 0.120,   # 1-e^-0.120 = .113

    "grid_goals_max": 12,
    "reg_weight": 0.006,
    "lam_grid": np.round(np.arange(1.00, 2.51, 0.10), 2),
    "mu_grid":  np.round(np.arange(0.50, 1.51, 0.10), 2),
    "rho_grid": np.round(np.arange(-0.14, 0.121, 0.02), 3),
    "rho_cap": 0.12,

    # Mahrez (ALG) 1+ SOT, two scenarios.
    "mahrez": {"start_sot": 0.60, "bench_sot": 0.357},   # P ~0.45 / ~0.30
}

PRIORS = {
    "Q1 Argentina 2+ offsides": 0.55,
    "Q2 Algeria more fouls than Argentina": 0.58,
    "Q3 4+ total SOT in 2nd half": 0.57,
    "Q4 ARG score first AND ALG score in 2H": 0.16,
    "Q5 Algeria 2+ SOT in 2nd half": 0.38,
    "Q6 Penalty OR red card": 0.36,
    "Q7 Argentina win": 0.67,
    "Q8 3+ total goals": 0.49,
    "Q9 Argentina score in 1st half": 0.57,
    "Q10a Mahrez 1+ SOT — STARTS": 0.45,
    "Q10b Mahrez 1+ SOT — BENCHED": 0.30,
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


def p_over_from_pmf(P):
    """P(total goals >= 3) = Over 2.5."""
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
                f, d, g = wdl_from_pmf(P)
                loss = (w12 * ((f - tgt["fav"]) ** 2 + (d - tgt["draw"]) ** 2 + (g - tgt["dog"]) ** 2) +
                        w_over * (p_over_from_pmf(P) - over_tgt) ** 2 + reg_lam + reg_mu)
                if best is None or loss < best[0]:
                    best = (loss, lmbda, mu, rho)
    return best


def anchor_goal_model(cfg):
    kmax, floor = cfg["grid_goals_max"], cfg["dog_lambda_floor"]
    hf = (1.0 - cfg["heat_pct"]) if cfg["heat_affects_goals"] else 1.0
    xg_f, xg_d = cfg["xg_fav"] * hf, cfg["xg_dog"] * hf
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
    heat = 1.0 - cfg["heat_pct"]

    lmbda, mu, rho, P = anchor_goal_model(cfg)
    arg_goals, alg_goals = sample_goals(P, n, rng)   # fav=ARG, dog=ALG

    k = cfg["dominance_concentration"]
    d = rng.beta(cfg["poss_fav"] * k, (1 - cfg["poss_fav"]) * k, size=n)
    scale_fav = d / cfg["poss_fav"]          # ARG
    scale_dog = (1.0 - d) / cfg["poss_dog"]  # ALG

    g1f = rng.binomial(arg_goals, cfg["split_goals_h1"])
    g1d = rng.binomial(alg_goals, cfg["split_goals_h1"])
    g2f, g2d = arg_goals - g1f, alg_goals - g1d
    trailing_dog = g1f > g1d
    trailing_fav = g1d > g1f

    # First goal of the game (exchangeable order within the earliest scoring half).
    arg_first = np.zeros(n, dtype=bool)
    u = rng.random(n)
    h1g, h2g = g1f + g1d, g2f + g2d
    m1 = h1g > 0
    arg_first[m1] = u[m1] < (g1f[m1] / h1g[m1])
    m2 = (~m1) & (h2g > 0)
    arg_first[m2] = u[m2] < (g2f[m2] / h2g[m2])

    arg_leads_2plus = (g1f - g1d) >= cfg["h2_ease_lead"]
    ease = cfg["h2_ease_factor"] if cfg["h2_ease_enabled"] else 1.0
    mult_fav_2h = np.where(trailing_fav, cfg["trail_push"], np.where(arg_leads_2plus, ease, 1.0))
    mult_dog_2h = np.where(trailing_dog, cfg["trail_push"], 1.0)

    def build_sot(sf):
        fh1, fh2 = split_mult(cfg["sot"]["fav"] * heat * scale_fav * sf, cfg["split_sot_h1"], mult_fav_2h, rng)
        dh1, dh2 = split_mult(cfg["sot"]["dog"] * heat * scale_dog * sf, cfg["split_sot_h1"], mult_dog_2h, rng)
        return fh1, fh2, dh1, dh2
    sot_scale = 1.0
    sot_f_h1, sot_f_h2, sot_d_h1, sot_d_h2 = build_sot(sot_scale)
    combined = (sot_f_h1 + sot_f_h2 + sot_d_h1 + sot_d_h2).mean()
    if combined > cfg["sot_combined_cap"]:
        sot_scale = cfg["sot_combined_target"] / combined
        sot_f_h1, sot_f_h2, sot_d_h1, sot_d_h2 = build_sot(sot_scale)
    sot_f, sot_d = sot_f_h1 + sot_f_h2, sot_d_h1 + sot_d_h2

    cor_f_h1, cor_f_h2 = split_mult(cfg["corners"]["fav"] * heat * scale_fav, cfg["split_corners_h1"], mult_fav_2h, rng)
    cor_d_h1, cor_d_h2 = split_mult(cfg["corners"]["dog"] * heat * scale_dog, cfg["split_corners_h1"], mult_dog_2h, rng)

    off_f = rng.poisson(cfg["offsides"]["fav"] * heat * scale_fav)
    off_d = rng.poisson(cfg["offsides"]["dog"] * heat * scale_dog)

    fouls_f = rng.poisson(cfg["fouls"]["fav"] * scale_dog)   # ARG
    fouls_d = rng.poisson(cfg["fouls"]["dog"] * scale_fav)   # ALG

    bump = cfg["card_2h_heat_bump"]
    cards_f = (rng.poisson(cfg["cards"]["fav"] * cfg["split_cards_h1"], size=n) +
               rng.poisson(cfg["cards"]["fav"] * (1 - cfg["split_cards_h1"]) * bump, size=n))
    cards_d = (rng.poisson(cfg["cards"]["dog"] * cfg["split_cards_h1"], size=n) +
               rng.poisson(cfg["cards"]["dog"] * (1 - cfg["split_cards_h1"]) * bump, size=n))

    pens = rng.poisson(cfg["pen_lambda"], size=n)
    reds = rng.poisson(cfg["red_lambda"], size=n)
    pen_or_red = (pens >= 1) | (reds >= 1)

    mahrez_start = (rng.poisson(cfg["mahrez"]["start_sot"] * scale_dog) >= 1).mean()
    mahrez_bench = (rng.poisson(cfg["mahrez"]["bench_sot"] * scale_dog) >= 1).mean()

    total_goals = arg_goals + alg_goals

    q = {}
    q["Q1 Argentina 2+ offsides"]              = (off_f >= 2).mean()
    q["Q2 Algeria more fouls than Argentina"]  = (fouls_d > fouls_f).mean()
    q["Q3 4+ total SOT in 2nd half"]           = ((sot_f_h2 + sot_d_h2) >= 4).mean()
    q["Q4 ARG score first AND ALG score in 2H"]= (arg_first & (g2d >= 1)).mean()
    q["Q5 Algeria 2+ SOT in 2nd half"]         = (sot_d_h2 >= 2).mean()
    q["Q6 Penalty OR red card"]                = pen_or_red.mean()
    q["Q7 Argentina win"]                      = np.tril(P, -1).sum()
    q["Q8 3+ total goals"]                     = (total_goals >= 3).mean()
    q["Q9 Argentina score in 1st half"]        = (g1f >= 1).mean()
    q["Q10a Mahrez 1+ SOT — STARTS"]           = mahrez_start
    q["Q10b Mahrez 1+ SOT — BENCHED"]          = mahrez_bench

    diag = {
        "lambda": lmbda, "mu": mu, "rho": rho, "wdl": wdl_from_pmf(P), "sot_scale": sot_scale,
        "p_over_model": p_over_from_pmf(P), "total_goals_mean": total_goals.mean(),
        "p_alg_score": (alg_goals >= 1).mean(),
        "pen_red": {"combined": pen_or_red.mean(), "pen": (pens >= 1).mean(), "red": (reds >= 1).mean()},
        "calib": {
            "sot_combined": (sot_f + sot_d).mean(),
            "corners_combined": (cor_f_h1 + cor_f_h2 + cor_d_h1 + cor_d_h2).mean(),
            "fouls_combined": (fouls_f + fouls_d).mean(), "cards_combined": (cards_f + cards_d).mean(),
            "sot_fav": sot_f.mean(), "sot_dog": sot_d.mean(),
            "offsides_fav": off_f.mean(), "offsides_dog": off_d.mean(),
            "goals_fav": arg_goals.mean(), "goals_dog": alg_goals.mean(),
        },
    }
    return q, diag


def flag(p):
    return "🪙" if 0.42 <= p <= 0.58 else " "


def print_report(cfg):
    t0 = time.time()
    q, diag = simulate(cfg)
    elapsed = time.time() - t0
    n = cfg["n_sims"]

    print("=" * 84)
    print(f"  {cfg['match_profile']}  |  {cfg['team_fav']} vs {cfg['team_dog']} — 2026 WC Group J, Kansas City")
    print(f"  {n:,} sims in {elapsed:.3f}s   |   host OFF, heat {int(100*cfg['heat_pct'])}% on events only")
    print("=" * 84)

    f, dd, g = diag["wdl"]
    m = cfg["market_wdl"]
    print("\nGOAL MODEL — DUAL ANCHOR (Dixon-Coles)")
    print(f"  fitted lambda(ARG)={diag['lambda']:.2f}  lambda(ALG)={diag['mu']:.2f}  rho={diag['rho']:+.3f}"
          f"   (ALG floored at {cfg['dog_lambda_floor']:.2f})")
    print(f"  realized W/D/L : {f:.3f} / {dd:.3f} / {g:.3f}   vs market {m['fav']:.3f}/{m['draw']:.3f}/{m['dog']:.3f}"
          f"   (max err {max(abs(f-m['fav']),abs(dd-m['draw']),abs(g-m['dog']))*100:.1f}pp)")
    over_err = diag["p_over_model"] - cfg["market_p_over"]
    print(f"  Over {cfg['ou_line']}: model {100*diag['p_over_model']:.1f}% vs market {100*cfg['market_p_over']:.1f}%"
          f"  -> residual {100*over_err:+.1f}pp", end="")
    if abs(over_err) > cfg["reconcile_threshold"]:
        print("   ** >2pp: deferred to 1X2, total carries the residual **")
    else:
        print("   (both anchors hit within threshold)")
    print(f"  total goals mean = {diag['total_goals_mean']:.2f}   P(ALG score) = {100*diag['p_alg_score']:.1f}%")

    c = diag["calib"]
    print("\nCALIBRATION (full-match means; heat dampens events, not goals)")
    print(f"  SOT {c['sot_combined']:.2f} (norm ~7; ARG {c['sot_fav']:.2f}/ALG {c['sot_dog']:.2f})"
          f"{'  [rescaled x%.3f]' % diag['sot_scale'] if diag['sot_scale'] != 1.0 else ''}"
          f"   Corners {c['corners_combined']:.2f}   Fouls {c['fouls_combined']:.2f}"
          f"   Cards {c['cards_combined']:.2f}   Off ARG/ALG {c['offsides_fav']:.2f}/{c['offsides_dog']:.2f}")
    pr = diag["pen_red"]
    print(f"  Q6 check: P(pen)={100*pr['pen']:.1f}% P(red)={100*pr['red']:.1f}% -> OR {100*pr['combined']:.1f}%")

    print("\n  Q#  question                                   sim P    prior   flag")
    print("  " + "-" * 70)
    recon = []
    for i, (label, p) in enumerate(q.items(), 0):
        prior = PRIORS[label]
        print(f"  {label:<44s}  {100*p:5.1f}%  {100*prior:5.1f}%   {flag(p)}")
        if abs(p - prior) > 0.05:
            recon.append((label, p, prior))

    if recon:
        print("\n  RECONCILIATION (sim diverges >5pp from prior):")
        for label, p, prior in recon:
            why = _why(label, p, prior, diag, cfg)
            print(f"   - {label}: {100*p:.1f}% vs {100*prior:.0f}%  ({100*(p-prior):+.1f}pp) — {why}")

    print("\nSENSITIVITY — ARG xG +/-0.2 and 2H-easing OFF (focus Q7/Q8/Q9)")
    cols = []
    for dx in (-0.2, 0.0, +0.2):
        r, _ = simulate({**copy.deepcopy(cfg), "xg_fav": round(cfg["xg_fav"] + dx, 2)})
        cols.append((f"xG{cfg['xg_fav']+dx:+.2f}", r))
    cols.append(("easeOFF", simulate({**copy.deepcopy(cfg), "h2_ease_enabled": False})[0]))
    print("  {:<44s}".format("prop") + "".join(f"{name:>10s}" for name, _ in cols))
    for label in ("Q7 Argentina win", "Q8 3+ total goals", "Q9 Argentina score in 1st half",
                  "Q3 4+ total SOT in 2nd half", "Q5 Algeria 2+ SOT in 2nd half"):
        print("  {:<44s}".format(label) + "".join(f"{100*r[label]:9.1f}%" for _, r in cols))
    print("=" * 84)


def _why(label, p, prior, diag, cfg):
    if "Q8" in label or "3+ total" in label:
        return f"tied to dual-anchor total (mean {diag['total_goals_mean']:.2f}); 1X2 won, total residual reported above"
    if "fouls" in label:
        return "given foul gap (ALG 13 vs ARG 11) + possession coupling widen it beyond the flat prior"
    if "SOT" in label:
        return "driven by given SOT rates + 2H chase push; check rate confidence"
    return "given input-rate driven; see CFG"


if __name__ == "__main__":
    print_report(CFG)
