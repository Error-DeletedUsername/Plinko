"""
Monte Carlo match simulator — Jump Trading Probability Cup
Match: USA vs Paraguay, 2026 World Cup group stage (HOST-TERM validation case).

Adds a HOST / HOME-ADVANTAGE term to the goal model. The bug it fixes: feeding
neutral-venue xG for a host nation under-prices the game's tempo, poisoning every
"something happens" prop, even when the 1X2 is anchored to the market.

KEY IDEA (why this works without double-counting):
A given de-vigged W/D/L does NOT uniquely pin the goal TOTAL once the Dixon-Coles
rho is free — there is a family of (lambda, mu, rho) all giving the same W/D/L at
different totals. The regularizer toward the (bumped) xG selects WHERE on that
family we land. So when ANCHOR_INCLUDES_HOST is True, bumping the host's xG:
  - leaves the 1X2 split alone (the anchor wins; rho absorbs the change), and
  - raises the goal TOTAL + the host's per-team event rates (SOT/corners/offsides)
    that the 1X2 alone never pinned down.
That is the real fix: the market gets the winner right, but a naive neutral-xG
model under-prices the tempo.

Everything else (DC rho fit, dominance factor, half splits, SOT guardrail) is the
same engine as the other matches.
"""

import time
import copy
import numpy as np

# ======================================================================
# CFG — single source of truth.
# ======================================================================
CFG = {
    "seed": 20260615,
    "n_sims": 200_000,

    "team_fav": "USA",        # "home" maps here (team A axis)
    "team_dog": "Paraguay",   # "away" maps here (team B axis)

    # ---------- HOST / HOME-ADVANTAGE BLOCK (toggle per match) ----------
    "host_team": "home",                 # "home" | "away" | None — which side gets the bump
    "host_xg_bump": 0.35,                # reference bump (the "host" value)
    "host_type": "host",                 # "host" (co-host, full) | "home" (nominal, ~0.5x) | None
    "host_bump_by_type": {"host": 0.35, "home": 0.18, None: 0.0},
    # True if the de-vigged 1X2 already reflects host advantage (almost always True
    # for liquid markets). True -> bump moves tempo only; False -> bump moves 1X2 too.
    "anchor_includes_host": True,
    # -------------------------------------------------------------------

    # De-vigged 1X2 anchor (USA host, market already prices the home edge).
    "market_wdl": {"fav": 0.55, "draw": 0.26, "dog": 0.19},
    "ou_line": 2.5,

    # RAW NEUTRAL-VENUE xG (pre-bump). The host bump is applied on top.
    "xg_fav": 1.30,   # USA neutral
    "xg_dog": 0.95,   # Paraguay

    "poss_fav": 0.57,  # INFERRED (host dominates ball)
    "poss_dog": 0.43,
    "dominance_concentration": 18.0,

    # Per-90 base rates (INFERRED for this case study).
    "sot":     {"fav": 4.3, "dog": 3.0},
    "corners": {"fav": 5.3, "dog": 3.8},
    "fouls":   {"fav": 10.5, "dog": 12.0},
    "cards":   {"fav": 1.9, "dog": 2.2},
    "offsides":{"fav": 1.5, "dog": 1.0},

    "split_goals_h1":   0.45,
    "split_cards_h1":   0.35,
    "split_corners_h1": 0.48,
    "split_sot_h1":     0.48,
    "trail_push": 1.20,

    "sot_combined_cap": 7.5,
    "sot_combined_target": 7.0,

    "grid_goals_max": 12,
    "reg_weight": 0.012,
    "lam_grid": np.round(np.arange(0.80, 2.51, 0.05), 2),
    "mu_grid":  np.round(np.arange(0.50, 1.81, 0.05), 2),
    "rho_grid": np.round(np.arange(-0.20, 0.121, 0.02), 3),
    "rho_cap": 0.12,
}

WC_NORMS = {"sot": 7.0, "corners": 8.9, "fouls": "22-26", "cards": 3.3}


# ======================================================================
# Goal model engine (unchanged)
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
    lam_f = np.round(np.arange(max(0.05, lmbda - 0.03), lmbda + 0.031, 0.0125), 4)
    mu_f = np.round(np.arange(max(0.05, mu - 0.03), mu + 0.031, 0.0125), 4)
    rho_f = np.round(np.arange(rho - 0.02, rho + 0.0201, 0.005), 4)
    _, lmbda, mu, rho = _grid_search(lam_f, mu_f, rho_f, cfg, px_cache)
    return lmbda, mu, rho, dc_joint_pmf(lmbda, mu, rho, kmax)


def sample_goals(P, n, rng):
    kmax = P.shape[0] - 1
    idx = rng.choice(P.size, size=n, p=P.ravel())
    return (idx // (kmax + 1)).astype(np.int64), (idx % (kmax + 1)).astype(np.int64)


# ======================================================================
# HOST-TERM application — builds an "effective" CFG from the host block.
# ======================================================================
def apply_host(cfg):
    """Return a copy of cfg with the host/home-advantage term folded in."""
    c = copy.deepcopy(cfg)
    bump = cfg["host_bump_by_type"][cfg["host_type"]]
    c["_host"] = {"bump": bump, "team": cfg["host_team"], "ratio": 1.0,
                  "raw_xg": None, "bumped_xg": None,
                  "anchor_includes_host": cfg["anchor_includes_host"]}
    if cfg["host_team"] is None or bump == 0.0:
        return c

    key = "fav" if cfg["host_team"] == "home" else "dog"
    raw = cfg["xg_fav"] if key == "fav" else cfg["xg_dog"]
    bumped = raw + bump
    ratio = bumped / raw
    c["_host"].update(ratio=ratio, raw_xg=raw, bumped_xg=bumped, key=key)

    # 1. Bump the regularizer-target xG for the host side (flows into the goal TOTAL
    #    by selecting the higher-total point on the equi-W/D/L family).
    c["xg_fav" if key == "fav" else "xg_dog"] = bumped

    # 2. Anchor handling (no double-count).
    if not cfg["anchor_includes_host"]:
        # Illiquid/no market: let the bump move the 1X2 too -> recompute the target
        # W/D/L from the bumped xG (rho seed 0).
        P = dc_joint_pmf(c["xg_fav"], c["xg_dog"], 0.0, cfg["grid_goals_max"])
        f, d, g = wdl_from_pmf(P)
        c["market_wdl"] = {"fav": f, "draw": d, "dog": g}
    # else: market_wdl untouched -> the anchor wins the split.

    # 3. Per-team event rates the 1X2 never pinned: scale the host's SOT/corners/offsides.
    for stat in ("sot", "corners", "offsides"):
        c[stat] = dict(cfg[stat])
        c[stat][key] = cfg[stat][key] * ratio

    # 4. Lift the SOT guardrail by the induced tempo so the bump survives the rescale
    #    (a host game legitimately sits above the neutral ~7.0 norm).
    uplift = (c["sot"]["fav"] + c["sot"]["dog"]) / (cfg["sot"]["fav"] + cfg["sot"]["dog"])
    c["sot_combined_target"] = cfg["sot_combined_target"] * uplift
    c["sot_combined_cap"] = cfg["sot_combined_cap"] * uplift
    return c


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
# Core simulation (takes an already host-applied effective cfg)
# ======================================================================
def simulate(cfg):
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["n_sims"]

    lmbda, mu, rho, P = anchor_goal_model(cfg)
    usa_goals, par_goals = sample_goals(P, n, rng)

    k = cfg["dominance_concentration"]
    d = rng.beta(cfg["poss_fav"] * k, (1 - cfg["poss_fav"]) * k, size=n)
    scale_fav = d / cfg["poss_fav"]
    scale_dog = (1.0 - d) / cfg["poss_dog"]

    g1f = rng.binomial(usa_goals, cfg["split_goals_h1"])
    g1d = rng.binomial(par_goals, cfg["split_goals_h1"])
    g2f, g2d = usa_goals - g1f, par_goals - g1d
    trailing_dog = g1f > g1d
    trailing_fav = g1d > g1f

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
    corners_f, corners_d = cor_f_h1 + cor_f_h2, cor_d_h1 + cor_d_h2

    off_f = rng.poisson(cfg["offsides"]["fav"] * scale_fav)
    off_d = rng.poisson(cfg["offsides"]["dog"] * scale_dog)

    fouls_f = rng.poisson(cfg["fouls"]["fav"] * scale_dog)
    fouls_d = rng.poisson(cfg["fouls"]["dog"] * scale_fav)

    cards_f = rng.poisson(cfg["cards"]["fav"], size=n)
    cards_d = rng.poisson(cfg["cards"]["dog"], size=n)

    total_goals = usa_goals + par_goals
    h1g, h2g = g1f + g1d, g2f + g2d
    both_score = (usa_goals >= 1) & (par_goals >= 1)

    # ---- 10 PROPS (tempo-focused for the host-term validation) ----
    q = {}
    q["Q1 USA win"]                       = np.tril(P, -1).sum()
    q["Q2 3+ total goals"]                = (total_goals >= 3).mean()
    q["Q3 Both teams score"]              = both_score.mean()
    q["Q4 2nd half 2+ goals"]            = (h2g >= 2).mean()
    q["Q5 USA 2+ goals"]                  = (usa_goals >= 2).mean()
    q["Q6 8+ combined SOT"]               = ((sot_f + sot_d) >= 8).mean()
    q["Q7 10+ combined corners"]          = ((corners_f + corners_d) >= 10).mean()
    q["Q8 Paraguay score 1+"]             = (par_goals >= 1).mean()
    q["Q9 4+ total cards"]                = ((cards_f + cards_d) >= 4).mean()
    q["Q10 Both score AND 3+ goals"]      = (both_score & (total_goals >= 3)).mean()

    diag = {
        "lambda": lmbda, "mu": mu, "rho": rho, "wdl": wdl_from_pmf(P), "sot_scale": sot_scale,
        "host": cfg["_host"],
        "total_goals_mean": total_goals.mean(),
        "p_3plus": (total_goals >= 3).mean(), "p_bts": both_score.mean(),
        "calib": {
            "sot_combined": (sot_f + sot_d).mean(), "corners_combined": (corners_f + corners_d).mean(),
            "fouls_combined": (fouls_f + fouls_d).mean(), "cards_combined": (cards_f + cards_d).mean(),
            "sot_fav": sot_f.mean(), "sot_dog": sot_d.mean(),
            "offsides_fav": off_f.mean(), "offsides_dog": off_d.mean(),
            "goals_fav": usa_goals.mean(), "goals_dog": par_goals.mean(),
        },
    }
    return q, diag


def run(cfg):
    return simulate(apply_host(cfg))


# ======================================================================
# Reporting
# ======================================================================
def coin_flag(p):
    return "  <-- COIN FLIP (42-58%)" if 0.42 <= p <= 0.58 else ""


def print_report(cfg):
    t0 = time.time()
    q, diag = run(cfg)
    elapsed = time.time() - t0
    n = cfg["n_sims"]
    h = diag["host"]

    print("=" * 78)
    print(f"  {cfg['team_fav']} vs {cfg['team_dog']} — 2026 World Cup  (HOST-TERM build)")
    print(f"  {n:,} vectorized sims in {elapsed:.3f}s")
    print("=" * 78)

    # ---- CFG echo (host block) ----
    print("\nCFG  HOST BLOCK")
    print(f"  host_team={cfg['host_team']}  host_type={cfg['host_type']}  "
          f"effective_bump={h['bump']:.2f}  anchor_includes_host={cfg['anchor_includes_host']}")
    print(f"  host_bump_by_type={cfg['host_bump_by_type']}")

    f, dd, g = diag["wdl"]
    m = cfg["market_wdl"]
    print("\nGOAL MODEL (Dixon-Coles, anchored to de-vigged market)")
    print(f"  fitted lambda(USA)={diag['lambda']:.2f}  lambda(PAR)={diag['mu']:.2f}  rho={diag['rho']:+.3f}")
    print(f"  realized W/D/L : {f:.3f} / {dd:.3f} / {g:.3f}   (USA / draw / PAR)")
    print(f"  market  W/D/L : {m['fav']:.3f} / {m['draw']:.3f} / {m['dog']:.3f}")
    print(f"  total goals mean = {diag['total_goals_mean']:.2f};  O/U {cfg['ou_line']}: "
          f"P(3+)={100*diag['p_3plus']:.1f}%")

    c = diag["calib"]
    print("\nCALIBRATION (simulated full-match means vs World Cup norms)")
    print(f"  SOT combined   : {c['sot_combined']:.2f}   (norm ~{WC_NORMS['sot']}; host-lifted)"
          f"{'   [rescaled x%.3f]' % diag['sot_scale'] if diag['sot_scale'] != 1.0 else ''}")
    print(f"     USA / PAR   : {c['sot_fav']:.2f} / {c['sot_dog']:.2f}")
    print(f"  Corners combined: {c['corners_combined']:.2f}  (norm ~{WC_NORMS['corners']})")
    print(f"  Fouls combined : {c['fouls_combined']:.2f}  (norm ~{WC_NORMS['fouls']})")
    print(f"  Cards combined : {c['cards_combined']:.2f}   (norm ~{WC_NORMS['cards']})")
    print(f"  Offsides USA/PAR: {c['offsides_fav']:.2f} / {c['offsides_dog']:.2f}")
    print(f"  Goals USA/PAR  : {c['goals_fav']:.2f} / {c['goals_dog']:.2f}")

    # ---- HOST DIAGNOSTIC: before (neutral) vs after (host) ----
    neutral_cfg = {**copy.deepcopy(cfg), "host_type": None}
    qn, dn = run(neutral_cfg)
    print("\n" + "-" * 78)
    print("HOST DIAGNOSTIC  (neutral-xG  ->  host-bumped)")
    if h["bump"] > 0:
        print(f"  pre-bump xG  : USA {h['raw_xg']:.2f} / PAR {cfg['xg_dog']:.2f}")
        print(f"  post-bump xG : USA {h['bumped_xg']:.2f} / PAR {cfg['xg_dog']:.2f}   (x{h['ratio']:.3f})")
    cn, ca = dn["calib"], diag["calib"]
    rows = [
        ("total goals (mean)", dn["total_goals_mean"], diag["total_goals_mean"], "{:.2f}"),
        ("SOT combined",       cn["sot_combined"],     ca["sot_combined"],       "{:.2f}"),
        ("corners combined",   cn["corners_combined"], ca["corners_combined"],   "{:.2f}"),
        ("P(3+ goals) %",      100*dn["p_3plus"],      100*diag["p_3plus"],      "{:.1f}"),
        ("P(both score) %",    100*dn["p_bts"],        100*diag["p_bts"],        "{:.1f}"),
    ]
    print(f"  {'metric':<22s}{'neutral':>10s}{'host':>10s}{'delta':>10s}")
    for name, a, b, fm in rows:
        print(f"  {name:<22s}{fm.format(a):>10s}{fm.format(b):>10s}{('+'+fm.format(b-a)) if b>=a else fm.format(b-a):>10s}")

    # ---- USA-PARAGUAY BEFORE/AFTER PROP TABLE ----
    print("\n" + "-" * 78)
    print("USA vs PARAGUAY — BEFORE (neutral xG) vs AFTER (host bump), 95% CI on AFTER")
    print(f"  {'prop':<34s}{'neutral':>9s}{'host':>9s}{'delta':>9s}   95% CI (host)")
    for label in q:
        a, b = qn[label], q[label]
        lo, hi = fmt_ci(b, n)
        flag = coin_flag(b)
        delta = b - a
        print(f"  {label:<34s}{100*a:8.1f}%{100*b:8.1f}%{('+' if delta>=0 else '')+format(100*delta,'.1f'):>9s}"
              f"   [{100*lo:4.1f},{100*hi:4.1f}]{flag}")

    # ---- sensitivity on NED... USA xG +/-0.2 (on top of the host bump) ----
    print("\n" + "-" * 78)
    print("SENSITIVITY — USA raw xG +/- 0.2 (host bump still applied, full re-anchor)")
    deltas = [-0.2, 0.0, +0.2]
    runs = {}
    for dx in deltas:
        c2 = {**copy.deepcopy(cfg), "xg_fav": round(cfg["xg_fav"] + dx, 2)}
        runs[dx], _ = run(c2)
    print("  {:<34s}".format("prop") + "".join(f"{('xG '+format(cfg['xg_fav']+dx,'.2f')):>10s}" for dx in deltas))
    for label in q:
        print("  {:<34s}".format(label) + "".join(f"{100*runs[dx][label]:9.1f}%" for dx in deltas))
    print("=" * 78)


if __name__ == "__main__":
    print_report(CFG)
