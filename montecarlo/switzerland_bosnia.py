"""
Monte Carlo match simulator — Jump Trading Probability Cup
Match: Switzerland vs Bosnia-Herzegovina, 2026 World Cup Group B, SoFi LA (mild).

Standard favourite shape: Switzerland is favoured AND possession-dominant (~64%)
vs a deep, physical Bosnia 4-4-2 that keeps a REAL counter/set-piece attack floor
(Demirovic/Lukic, Katic aerial). EVENT_PROFILE = MODERATE (leans low; Swiss
dominate but are wasteful). Mild venue -> NO heat dampening.

Engine ported from prior matches (dual-anchor Dixon-Coles + shared dominance +
half-splits + game-state easing). Standard couplings ON, including the
inverse-possession foul rule (Bosnia = lower possession + deep block -> fouls more,
the rule holds here, unlike Ghana-Panama).

Q4 = BTTS AND 3+ total goals (joint). Player props (Xhaka 2H SOT, Dzeko full SOT)
output both lineup scenarios. 200k vectorized sims, <1s.
"""

import time
import copy
import numpy as np

CFG = {
    "seed": 20260615,
    "n_sims": 200_000,

    "team_fav": "Switzerland",
    "team_dog": "Bosnia",
    "match_profile": "MODERATE",

    "host_team": None, "host_type": None,
    "host_bump_by_type": {"host": 0.35, "home": 0.18, None: 0.0},
    "anchor_includes_host": True,

    "market_wdl": {"fav": 0.62, "draw": 0.23, "dog": 0.15},
    "ou_line": 2.5,
    "market_p_over": 0.47,
    "over_in_grid": True,
    "x12_weight": 2.5,
    "reconcile_threshold": 0.02,

    "xg_fav": 1.55,   # SUI
    "xg_dog": 0.85,   # BIH (real attack)
    "dog_lambda_floor": 0.60,

    "poss_fav": 0.64, "poss_dog": 0.36,
    "dominance_concentration": 18.0,

    "heat_pct": 0.0,
    "heat_affects_goals": False,

    "sot":     {"fav": 6.5, "dog": 5.0},   # Swiss high-volume/wasteful; Bosnia real threat
    "corners": {"fav": 6.5, "dog": 3.5},
    "fouls":   {"fav": 11.5, "dog": 14.5}, # Bosnia deep block -> more fouls
    "fouls_use_dominance": True,           # inverse-possession rule HOLDS here
    "cards":   {"fav": 1.4, "dog": 1.7},   # lenient ref Pinheiro; Bosnia physical
    "offsides":{"fav": 1.45, "dog": 1.0},  # Swiss offside tempered (Bosnia sits deep)

    "split_goals_h1":   0.45,
    "split_cards_h1":   0.35,
    "split_corners_h1": 0.48,
    "split_sot_h1":     0.48,
    "split_offsides_h1":0.50,
    "trail_push": 1.20,
    "card_2h_heat_bump": 1.15,   # late fatigue + Bosnia physicality lift 2H cards

    "h2_ease_enabled": True,
    "h2_ease_factor": 0.825,
    "h2_ease_lead": 2,

    # MODERATE: trim the very high given SOT toward a moderate ~7.6 combined.
    "sot_combined_cap": 8.2,
    "sot_combined_target": 7.6,

    "pen_lambda": 0.315,   # Q6: P(1+ pen) = 1-e^-0.315 ~ 0.27 (ref Pinheiro ~25%)
    "red_lambda": 0.130,

    "grid_goals_max": 12,
    "reg_weight": 0.006,
    "lam_grid": np.round(np.arange(0.90, 2.41, 0.10), 2),
    "mu_grid":  np.round(np.arange(0.50, 1.61, 0.10), 2),
    "rho_grid": np.round(np.arange(-0.14, 0.121, 0.02), 3),
    "rho_cap": 0.12,

    # Player props (two scenarios each).
    "xhaka": {"start_2h": 0.116, "bench_2h": 0.052},   # deep-lying; P(1+ SOT 2H) ~0.11 / ~0.05
    "dzeko": {"start_sot": 0.545, "bench_sot": 0.248}, # P(1+ SOT full) ~0.42 / ~0.22
}

PRIORS = {
    "Q1 Both teams 1+ SOT at halftime": 0.52,
    "Q2 Switzerland 2+ offsides": 0.42,
    "Q3 2+ total cards in 2nd half": 0.67,
    "Q4 BTTS AND 3+ total goals": 0.27,
    "Q5 Bosnia more fouls than Switzerland": 0.63,
    "Q6 Penalty awarded in match": 0.27,
    "Q7 Switzerland more SOT than Bosnia (2H)": 0.60,
    "Q8a Xhaka 1+ SOT in 2H — STARTS": 0.11,
    "Q8b Xhaka 1+ SOT in 2H — BENCHED": 0.05,
    "Q9a Dzeko 1+ SOT — STARTS": 0.42,
    "Q9b Dzeko 1+ SOT — BENCHED": 0.22,
    "Q10 Switzerland win": 0.62,
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
    heat = 1.0 - cfg["heat_pct"]

    lmbda, mu, rho, P = anchor_goal_model(cfg)
    sui_goals, bih_goals = sample_goals(P, n, rng)   # fav=SUI, dog=BIH

    k = cfg["dominance_concentration"]
    d = rng.beta(cfg["poss_fav"] * k, (1 - cfg["poss_fav"]) * k, size=n)
    scale_fav = d / cfg["poss_fav"]          # SUI
    scale_dog = (1.0 - d) / cfg["poss_dog"]  # BIH

    g1f = rng.binomial(sui_goals, cfg["split_goals_h1"])
    g1d = rng.binomial(bih_goals, cfg["split_goals_h1"])
    g2f, g2d = sui_goals - g1f, bih_goals - g1d
    trailing_dog = g1f > g1d
    trailing_fav = g1d > g1f

    sui_leads_2plus = (g1f - g1d) >= cfg["h2_ease_lead"]
    ease = cfg["h2_ease_factor"] if cfg["h2_ease_enabled"] else 1.0
    mult_fav_2h = np.where(trailing_fav, cfg["trail_push"], np.where(sui_leads_2plus, ease, 1.0))
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

    if cfg["fouls_use_dominance"]:
        fouls_f = rng.poisson(cfg["fouls"]["fav"] * scale_dog)
        fouls_d = rng.poisson(cfg["fouls"]["dog"] * scale_fav)
    else:
        fouls_f = rng.poisson(cfg["fouls"]["fav"], size=n)
        fouls_d = rng.poisson(cfg["fouls"]["dog"], size=n)

    bump = cfg["card_2h_heat_bump"]
    card_f_h2 = rng.poisson(cfg["cards"]["fav"] * (1 - cfg["split_cards_h1"]) * bump, size=n)
    card_d_h2 = rng.poisson(cfg["cards"]["dog"] * (1 - cfg["split_cards_h1"]) * bump, size=n)
    cards_2h = card_f_h2 + card_d_h2
    cards_f = rng.poisson(cfg["cards"]["fav"] * cfg["split_cards_h1"], size=n) + card_f_h2
    cards_d = rng.poisson(cfg["cards"]["dog"] * cfg["split_cards_h1"], size=n) + card_d_h2

    pens = rng.poisson(cfg["pen_lambda"], size=n)
    reds = rng.poisson(cfg["red_lambda"], size=n)

    # Xhaka 2H SOT (deep-lying mid, strong under); Dzeko full-match SOT.
    xhaka_start = (rng.poisson(cfg["xhaka"]["start_2h"] * scale_fav) >= 1).mean()
    xhaka_bench = (rng.poisson(cfg["xhaka"]["bench_2h"] * scale_fav) >= 1).mean()
    dzeko_start = (rng.poisson(cfg["dzeko"]["start_sot"] * scale_dog) >= 1).mean()
    dzeko_bench = (rng.poisson(cfg["dzeko"]["bench_sot"] * scale_dog) >= 1).mean()

    total_goals = sui_goals + bih_goals
    both_score = (sui_goals >= 1) & (bih_goals >= 1)

    q = {}
    q["Q1 Both teams 1+ SOT at halftime"]      = ((sot_f_h1 >= 1) & (sot_d_h1 >= 1)).mean()
    q["Q2 Switzerland 2+ offsides"]            = (off_f >= 2).mean()
    q["Q3 2+ total cards in 2nd half"]         = (cards_2h >= 2).mean()
    q["Q4 BTTS AND 3+ total goals"]            = (both_score & (total_goals >= 3)).mean()
    q["Q5 Bosnia more fouls than Switzerland"] = (fouls_d > fouls_f).mean()
    q["Q6 Penalty awarded in match"]           = (pens >= 1).mean()
    q["Q7 Switzerland more SOT than Bosnia (2H)"] = (sot_f_h2 > sot_d_h2).mean()
    q["Q8a Xhaka 1+ SOT in 2H — STARTS"]       = xhaka_start
    q["Q8b Xhaka 1+ SOT in 2H — BENCHED"]      = xhaka_bench
    q["Q9a Dzeko 1+ SOT — STARTS"]             = dzeko_start
    q["Q9b Dzeko 1+ SOT — BENCHED"]            = dzeko_bench
    q["Q10 Switzerland win"]                   = np.tril(P, -1).sum()

    diag = {
        "lambda": lmbda, "mu": mu, "rho": rho, "wdl": wdl_from_pmf(P), "sot_scale": sot_scale,
        "p_over_model": p_over_from_pmf(P), "total_goals_mean": total_goals.mean(),
        "p_bih_score": (bih_goals >= 1).mean(), "p_sui_score": (sui_goals >= 1).mean(),
        "calib": {
            "sot_combined": (sot_f + sot_d).mean(),
            "corners_combined": (cor_f_h1 + cor_f_h2 + cor_d_h1 + cor_d_h2).mean(),
            "fouls_combined": (fouls_f + fouls_d).mean(), "cards_combined": (cards_f + cards_d).mean(),
            "cards_2h_mean": cards_2h.mean(),
            "sot_fav": sot_f.mean(), "sot_dog": sot_d.mean(),
            "sot_dog_h1": sot_d_h1.mean(), "sot_fav_h1": sot_f_h1.mean(),
            "offsides_fav": off_f.mean(), "offsides_dog": off_d.mean(),
            "goals_fav": sui_goals.mean(), "goals_dog": bih_goals.mean(),
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

    print("=" * 86)
    print(f"  {cfg['match_profile']}  |  {cfg['team_fav']} (FAV) vs {cfg['team_dog']} — 2026 WC Group B, SoFi LA")
    print(f"  {n:,} sims in {elapsed:.3f}s   |   host OFF, no heat (mild venue)")
    print("=" * 86)

    f, dd, g = diag["wdl"]
    m = cfg["market_wdl"]
    print("\nGOAL MODEL — DUAL ANCHOR (Dixon-Coles)")
    print(f"  fitted lambda(SUI)={diag['lambda']:.2f}  lambda(BIH)={diag['mu']:.2f}  rho={diag['rho']:+.3f}"
          f"   (BIH floored at {cfg['dog_lambda_floor']:.2f})")
    print(f"  realized W/D/L : {f:.3f} / {dd:.3f} / {g:.3f}   vs market {m['fav']:.3f}/{m['draw']:.3f}/{m['dog']:.3f}"
          f"   (max err {max(abs(f-m['fav']),abs(dd-m['draw']),abs(g-m['dog']))*100:.1f}pp)")
    over_err = diag["p_over_model"] - cfg["market_p_over"]
    print(f"  Over {cfg['ou_line']}: model {100*diag['p_over_model']:.1f}% vs market {100*cfg['market_p_over']:.1f}%"
          f"  -> residual {100*over_err:+.1f}pp", end="")
    print("   ** >2pp: deferred to 1X2 **" if abs(over_err) > cfg["reconcile_threshold"] else "   (both anchors within threshold)")
    print(f"  total goals mean = {diag['total_goals_mean']:.2f}   P(SUI score)={100*diag['p_sui_score']:.1f}%  P(BIH score)={100*diag['p_bih_score']:.1f}%")

    c = diag["calib"]
    print("\nCALIBRATION (full-match means; MODERATE — SOT trimmed from given 11.5)")
    print(f"  SOT {c['sot_combined']:.2f} (SUI {c['sot_fav']:.2f}/BIH {c['sot_dog']:.2f}; BIH H1 {c['sot_dog_h1']:.2f})"
          f"{'  [rescaled x%.3f]' % diag['sot_scale'] if diag['sot_scale'] != 1.0 else ''}"
          f"   Corners {c['corners_combined']:.2f}   Fouls {c['fouls_combined']:.2f}"
          f"   Cards {c['cards_combined']:.2f} (2H {c['cards_2h_mean']:.2f})   Off SUI/BIH {c['offsides_fav']:.2f}/{c['offsides_dog']:.2f}")

    print("\n  Q#  question                                   sim P    prior   flag")
    print("  " + "-" * 72)
    recon = []
    for label, p in q.items():
        prior = PRIORS[label]
        print(f"  {label:<46s}  {100*p:5.1f}%  {100*prior:5.1f}%   {flag(p)}")
        if abs(p - prior) > 0.05:
            recon.append((label, p, prior))

    if recon:
        print("\n  RECONCILIATION (sim diverges >5pp from prior):")
        for label, p, prior in recon:
            print(f"   - {label}: {100*p:.1f}% vs {100*prior:.0f}% ({100*(p-prior):+.1f}pp) — {_why(label, diag)}")

    print("\nSENSITIVITY — Switzerland (fav) xG +/-0.2 and 2H-easing OFF")
    cols = []
    for dx in (-0.2, 0.0, +0.2):
        r, _ = simulate({**copy.deepcopy(cfg), "xg_fav": round(cfg["xg_fav"] + dx, 2)})
        cols.append((f"xG{cfg['xg_fav']+dx:+.2f}", r))
    cols.append(("easeOFF", simulate({**copy.deepcopy(cfg), "h2_ease_enabled": False})[0]))
    print("  {:<46s}".format("prop") + "".join(f"{name:>10s}" for name, _ in cols))
    for label in ("Q10 Switzerland win", "Q4 BTTS AND 3+ total goals", "Q7 Switzerland more SOT than Bosnia (2H)",
                  "Q1 Both teams 1+ SOT at halftime", "Q3 2+ total cards in 2nd half"):
        print("  {:<46s}".format(label) + "".join(f"{100*r[label]:9.1f}%" for _, r in cols))
    print("=" * 86)


def _why(label, diag):
    c = diag["calib"]
    if "Q1" in label:
        return (f"given SOT rates make a HT shot near-automatic for both (BIH H1 mean {c['sot_dog_h1']:.1f} "
                f"-> P~{100*(1-np.exp(-c['sot_dog_h1'])):.0f}%); model does not suppress a deep block's early shots")
    if "Q5" in label:
        return "given foul gap (BIH 14 vs SUI 12) + possession coupling push Bosnia ahead more than the flat prior"
    if "Q7" in label:
        return ("Bosnia's 2H chase push (+20% when trailing) + protected attack floor close the 2H SOT gap; "
                "trimmed counts add ties -> Swiss clear them only ~half the time, below the 0.60 gut")
    if "Q4" in label or "BTTS" in label:
        return (f"tied to dual-anchor total (mean {diag['total_goals_mean']:.2f}); Bosnia's real floor "
                f"(P(BIH score)={100*diag['p_bih_score']:.0f}%) lifts BTTS above the gut prior")
    return "given input-rate driven; see CFG"


if __name__ == "__main__":
    print_report(CFG)
