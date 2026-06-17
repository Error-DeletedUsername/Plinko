"""
Monte Carlo match simulator — Jump Trading Probability Cup
Match: Ghana vs Panama, 2026 World Cup Group L, BMO Field Toronto (cool, ~16-18C).

UNUSUAL SHAPE: Ghana is the FAVOURITE but does NOT dominate the ball — Panama
controls possession (~57%) from an organized 3-4-2-1 while Ghana counters in
transition. So "favourite" and "possession-dominant" are DIFFERENT teams here:
    fav-axis = Ghana   (favourite; LOW possession 0.43; direct/transition SOT)
    dog-axis = Panama  (underdog by line; HIGH possession; deep block, more fouls)
LOW_EVENT cagey opener. Cool open-air venue -> NO heat dampening.

Engine ported from prior matches (dual-anchor Dixon-Coles + shared dominance +
half-splits + game-state easing). Two match-specific deviations, both flagged:

  FOULS DECOUPLED FROM DOMINANCE: the usual "less-possession side fouls more" rule
  is INVERTED here — Panama has MORE possession AND fouls more (deep block). So
  fouls use plain base-rate Poisson (no possession coupling) for this match.

  LOW_EVENT SOT NORM: a cagey low-block-vs-low-block game sits BELOW the ~7 SOT
  norm; the guardrail trims combined SOT to ~6.5, so Ghana's realized SOT (~3.7)
  reflects the matchup, not its open-play season rate. Ghana's floor is preserved
  (it stays the higher-SOT side), not crushed to zero.

Q4 = "Panama scores the FIRST goal of the 2nd half" — ordered within-2H event.
200k vectorized sims, <1s.
"""

import time
import copy
import numpy as np

CFG = {
    "seed": 20260615,
    "n_sims": 200_000,

    "team_fav": "Ghana",      # favourite (lower possession)
    "team_dog": "Panama",     # underdog by line (higher possession, deep block)
    "match_profile": "LOW_EVENT",

    "host_team": None, "host_type": None,
    "host_bump_by_type": {"host": 0.35, "home": 0.18, None: 0.0},
    "anchor_includes_host": True,

    # market_wdl["fav"] = Ghana win; ["dog"] = Panama win.
    "market_wdl": {"fav": 0.41, "draw": 0.30, "dog": 0.29},
    "ou_line": 2.5,
    "market_p_over": 0.40,
    "over_in_grid": True,
    "x12_weight": 2.5,
    "reconcile_threshold": 0.02,

    "xg_fav": 1.25,   # Ghana
    "xg_dog": 1.05,   # Panama (real attacking side)
    "dog_lambda_floor": 0.70,

    # Dominance mean = Ghana possession 0.43 (Panama controls the ball).
    "poss_fav": 0.43, "poss_dog": 0.57,
    "dominance_concentration": 18.0,

    "heat_pct": 0.0,            # cool venue -> no heat dampening
    "heat_affects_goals": False,

    "sot":     {"fav": 5.0, "dog": 3.8},   # Ghana direct (5 SOT vs Wales); Panama possession
    "corners": {"fav": 4.5, "dog": 5.5},   # Panama more corners (possession + set-pieces)
    "fouls":   {"fav": 11.5, "dog": 12.2}, # Panama deep block -> a touch more fouls
    "fouls_use_dominance": False,          # INVERTED rule here -> decouple fouls
    "cards":   {"fav": 1.5, "dog": 1.7},   # lenient ref Nyberg; Panama slight edge
    "offsides":{"fav": 1.3, "dog": 1.1},   # Ghana direct runs; Panama organized

    "split_goals_h1":   0.45,
    "split_cards_h1":   0.35,
    "split_corners_h1": 0.48,
    "split_sot_h1":     0.48,
    "split_offsides_h1":0.50,
    "trail_push": 1.20,
    "card_2h_heat_bump": 1.05,   # mild late-fatigue card bump (no heat)

    "h2_ease_enabled": True,
    "h2_ease_factor": 0.825,
    "h2_ease_lead": 2,

    # LOW_EVENT: combined SOT sits below the ~7 norm in a cagey game.
    "sot_combined_cap": 7.0,
    "sot_combined_target": 6.5,

    # Ref Nyberg (not a question; calibration only): pen 22%, red 5% -> OR ~26%.
    "pen_lambda": 0.248, "red_lambda": 0.051,

    "grid_goals_max": 12,
    "reg_weight": 0.006,
    "lam_grid": np.round(np.arange(0.70, 2.01, 0.10), 2),
    "mu_grid":  np.round(np.arange(0.60, 1.81, 0.10), 2),
    "rho_grid": np.round(np.arange(-0.14, 0.121, 0.02), 3),
    "rho_cap": 0.12,

    # Player props (two scenarios each).
    "semenyo": {"start_sot": 0.95, "bench_sot": 0.43},   # Ghana; P ~0.61 / ~0.35
    "fajardo": {"start_sot": 0.357, "bench_sot": 0.128}, # Panama; P ~0.30 / ~0.12
}

PRIORS = {
    "Q1 Ghana more 2H SOT than Panama": 0.52,
    "Q2 Ghana more fouls than Panama": 0.42,
    "Q3 Panama more cards than Ghana": 0.45,
    "Q4 Panama scores first goal of 2nd half": 0.23,
    "Q5 Panama 2+ offsides": 0.32,
    "Q6 Ghana win": 0.41,
    "Q7 2nd half 2+ total goals": 0.37,
    "Q8 Ghana 3+ SOT full match": 0.68,
    "Q9a Semenyo 1+ SOT — STARTS": 0.60,
    "Q9b Semenyo 1+ SOT — BENCHED": 0.35,
    "Q10a Fajardo 1+ SOT — STARTS": 0.30,
    "Q10b Fajardo 1+ SOT — BENCHED": 0.12,
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
                f, d, g = wdl_from_pmf(P)
                loss = (w12 * ((f - tgt["fav"]) ** 2 + (d - tgt["draw"]) ** 2 + (g - tgt["dog"]) ** 2) +
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
    heat = 1.0 - cfg["heat_pct"]

    lmbda, mu, rho, P = anchor_goal_model(cfg)
    gha_goals, pan_goals = sample_goals(P, n, rng)   # fav=Ghana, dog=Panama

    k = cfg["dominance_concentration"]
    d = rng.beta(cfg["poss_fav"] * k, (1 - cfg["poss_fav"]) * k, size=n)
    scale_fav = d / cfg["poss_fav"]          # Ghana (mean 1.0; usually < 1 since Panama on ball)
    scale_dog = (1.0 - d) / cfg["poss_dog"]  # Panama

    g1f = rng.binomial(gha_goals, cfg["split_goals_h1"])
    g1d = rng.binomial(pan_goals, cfg["split_goals_h1"])
    g2f, g2d = gha_goals - g1f, pan_goals - g1d
    trailing_dog = g1f > g1d
    trailing_fav = g1d > g1f

    # Panama scores the FIRST goal of the 2nd half (ordered within 2H).
    g2g = g2f + g2d
    pan_first_2h = np.zeros(n, dtype=bool)
    u = rng.random(n)
    m2 = g2g > 0
    pan_first_2h[m2] = u[m2] < (g2d[m2] / g2g[m2])

    gha_leads_2plus = (g1f - g1d) >= cfg["h2_ease_lead"]
    ease = cfg["h2_ease_factor"] if cfg["h2_ease_enabled"] else 1.0
    mult_fav_2h = np.where(trailing_fav, cfg["trail_push"], np.where(gha_leads_2plus, ease, 1.0))
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

    # Fouls: decoupled from dominance for THIS match (inverted possession/foul rule).
    if cfg["fouls_use_dominance"]:
        fouls_f = rng.poisson(cfg["fouls"]["fav"] * scale_dog)
        fouls_d = rng.poisson(cfg["fouls"]["dog"] * scale_fav)
    else:
        fouls_f = rng.poisson(cfg["fouls"]["fav"], size=n)
        fouls_d = rng.poisson(cfg["fouls"]["dog"], size=n)

    bump = cfg["card_2h_heat_bump"]
    cards_f = (rng.poisson(cfg["cards"]["fav"] * cfg["split_cards_h1"], size=n) +
               rng.poisson(cfg["cards"]["fav"] * (1 - cfg["split_cards_h1"]) * bump, size=n))
    cards_d = (rng.poisson(cfg["cards"]["dog"] * cfg["split_cards_h1"], size=n) +
               rng.poisson(cfg["cards"]["dog"] * (1 - cfg["split_cards_h1"]) * bump, size=n))

    pens = rng.poisson(cfg["pen_lambda"], size=n)
    reds = rng.poisson(cfg["red_lambda"], size=n)

    semenyo_start = (rng.poisson(cfg["semenyo"]["start_sot"] * scale_fav) >= 1).mean()
    semenyo_bench = (rng.poisson(cfg["semenyo"]["bench_sot"] * scale_fav) >= 1).mean()
    fajardo_start = (rng.poisson(cfg["fajardo"]["start_sot"] * scale_dog) >= 1).mean()
    fajardo_bench = (rng.poisson(cfg["fajardo"]["bench_sot"] * scale_dog) >= 1).mean()

    total_goals = gha_goals + pan_goals

    q = {}
    q["Q1 Ghana more 2H SOT than Panama"]      = (sot_f_h2 > sot_d_h2).mean()
    q["Q2 Ghana more fouls than Panama"]       = (fouls_f > fouls_d).mean()
    q["Q3 Panama more cards than Ghana"]       = (cards_d > cards_f).mean()
    q["Q4 Panama scores first goal of 2nd half"]= pan_first_2h.mean()
    q["Q5 Panama 2+ offsides"]                 = (off_d >= 2).mean()
    q["Q6 Ghana win"]                          = np.tril(P, -1).sum()
    q["Q7 2nd half 2+ total goals"]            = (g2g >= 2).mean()
    q["Q8 Ghana 3+ SOT full match"]            = (sot_f >= 3).mean()
    q["Q9a Semenyo 1+ SOT — STARTS"]           = semenyo_start
    q["Q9b Semenyo 1+ SOT — BENCHED"]          = semenyo_bench
    q["Q10a Fajardo 1+ SOT — STARTS"]          = fajardo_start
    q["Q10b Fajardo 1+ SOT — BENCHED"]         = fajardo_bench

    diag = {
        "lambda": lmbda, "mu": mu, "rho": rho, "wdl": wdl_from_pmf(P), "sot_scale": sot_scale,
        "p_over_model": p_over_from_pmf(P), "total_goals_mean": total_goals.mean(),
        "p_pan_score": (pan_goals >= 1).mean(), "p_gha_score": (gha_goals >= 1).mean(),
        "pen_red_or": ((pens >= 1) | (reds >= 1)).mean(),
        "calib": {
            "sot_combined": (sot_f + sot_d).mean(),
            "corners_combined": (cor_f_h1 + cor_f_h2 + cor_d_h1 + cor_d_h2).mean(),
            "fouls_combined": (fouls_f + fouls_d).mean(), "cards_combined": (cards_f + cards_d).mean(),
            "sot_fav": sot_f.mean(), "sot_dog": sot_d.mean(),
            "offsides_fav": off_f.mean(), "offsides_dog": off_d.mean(),
            "goals_fav": gha_goals.mean(), "goals_dog": pan_goals.mean(),
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

    print("=" * 84)
    print(f"  {cfg['match_profile']}  |  {cfg['team_fav']} (FAV) vs {cfg['team_dog']} — 2026 WC Group L, BMO Field Toronto")
    print(f"  {n:,} sims in {elapsed:.3f}s   |   host OFF, no heat (cool venue), fouls decoupled from possession")
    print("=" * 84)

    f, dd, g = diag["wdl"]
    m = cfg["market_wdl"]
    print("\nGOAL MODEL — DUAL ANCHOR (Dixon-Coles)")
    print(f"  fitted lambda(GHA)={diag['lambda']:.2f}  lambda(PAN)={diag['mu']:.2f}  rho={diag['rho']:+.3f}"
          f"   (PAN floored at {cfg['dog_lambda_floor']:.2f})")
    print(f"  realized W/D/L : {f:.3f} / {dd:.3f} / {g:.3f}   vs market {m['fav']:.3f}/{m['draw']:.3f}/{m['dog']:.3f}"
          f"   (max err {max(abs(f-m['fav']),abs(dd-m['draw']),abs(g-m['dog']))*100:.1f}pp)")
    over_err = diag["p_over_model"] - cfg["market_p_over"]
    print(f"  Over {cfg['ou_line']}: model {100*diag['p_over_model']:.1f}% vs market {100*cfg['market_p_over']:.1f}%"
          f"  -> residual {100*over_err:+.1f}pp", end="")
    print("   ** >2pp: deferred to 1X2 **" if abs(over_err) > cfg["reconcile_threshold"] else "   (both anchors within threshold)")
    print(f"  total goals mean = {diag['total_goals_mean']:.2f}   P(GHA score)={100*diag['p_gha_score']:.1f}%  P(PAN score)={100*diag['p_pan_score']:.1f}%")

    c = diag["calib"]
    print("\nCALIBRATION (full-match means; LOW_EVENT SOT trimmed below ~7 norm)")
    print(f"  SOT {c['sot_combined']:.2f} (GHA {c['sot_fav']:.2f}/PAN {c['sot_dog']:.2f})"
          f"{'  [rescaled x%.3f]' % diag['sot_scale'] if diag['sot_scale'] != 1.0 else ''}"
          f"   Corners {c['corners_combined']:.2f}   Fouls {c['fouls_combined']:.2f}"
          f"   Cards {c['cards_combined']:.2f}   Off GHA/PAN {c['offsides_fav']:.2f}/{c['offsides_dog']:.2f}")
    print(f"  (ref Nyberg pen-OR-red ~ {100*diag['pen_red_or']:.0f}% — not a question this match)")

    print("\n  Q#  question                                   sim P    prior   flag")
    print("  " + "-" * 70)
    recon = []
    for label, p in q.items():
        prior = PRIORS[label]
        print(f"  {label:<44s}  {100*p:5.1f}%  {100*prior:5.1f}%   {flag(p)}")
        if abs(p - prior) > 0.05:
            recon.append((label, p, prior))

    if recon:
        print("\n  RECONCILIATION (sim diverges >5pp from prior):")
        for label, p, prior in recon:
            print(f"   - {label}: {100*p:.1f}% vs {100*prior:.0f}% ({100*(p-prior):+.1f}pp) — {_why(label, diag)}")

    print("\nSENSITIVITY — Ghana (fav) xG +/-0.2 and 2H-easing OFF")
    cols = []
    for dx in (-0.2, 0.0, +0.2):
        r, _ = simulate({**copy.deepcopy(cfg), "xg_fav": round(cfg["xg_fav"] + dx, 2)})
        cols.append((f"xG{cfg['xg_fav']+dx:+.2f}", r))
    cols.append(("easeOFF", simulate({**copy.deepcopy(cfg), "h2_ease_enabled": False})[0]))
    print("  {:<44s}".format("prop") + "".join(f"{name:>10s}" for name, _ in cols))
    for label in ("Q6 Ghana win", "Q7 2nd half 2+ total goals", "Q8 Ghana 3+ SOT full match",
                  "Q1 Ghana more 2H SOT than Panama", "Q4 Panama scores first goal of 2nd half"):
        print("  {:<44s}".format(label) + "".join(f"{100*r[label]:9.1f}%" for _, r in cols))
    print("=" * 84)


def _why(label, diag):
    if "Q8" in label:
        return f"Ghana SOT mean {diag['calib']['sot_fav']:.1f}; even LOW_EVENT-trimmed, a high-SOT side clears 3 often"
    if "Q2" in label:
        return "decoupled foul gap (PAN 12.2 vs GHA 11.5) puts Panama ahead more often than the flat prior"
    if "Q4" in label:
        return f"market gives Panama a real share (P(PAN score)={100*diag['p_pan_score']:.0f}%); their 2H-first share > the gut prior"
    if "Q7" in label or "Q8" in label or "total" in label:
        return f"tied to dual-anchor total (mean {diag['total_goals_mean']:.2f}); 1X2 won"
    return "given input-rate driven; see CFG"


if __name__ == "__main__":
    print_report(CFG)
