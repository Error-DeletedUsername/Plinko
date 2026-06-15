"""
Monte Carlo match simulator — Jump Trading Probability Cup
Match: Saudi Arabia vs Uruguay, 2026 WC Group H, Hard Rock Stadium Miami
(OPEN-AIR, hot+humid ~29-33C, ~50% storm risk — heat is a live tempo dampener).

IMPORTANT AXIS NOTE: the FAVOURITE is the SECOND-named team (Uruguay, possession-
dominant high-press under Bielsa). The deep-block UNDERDOG is the FIRST-named team
(Saudi Arabia). So in this engine:
    fav-axis  = Uruguay     (price-up / high-volume logic)
    dog-axis  = Saudi Arabia (fade / low-volume / high-foul logic)
Every question is wired to the team it explicitly names — read labels carefully.

Weighted-Brier scoring: calibration over confidence; never 0/100; 42-58% flagged.
Same engine as belgium_egypt (anchored DC + dominance + half-splits + game-state
easing + heat-events-only + O/U-vs-1X2 reconciliation + underdog floor).
"""

import time
import copy
import numpy as np

CFG = {
    "seed": 20260615,
    "n_sims": 200_000,

    "team_fav": "Uruguay",        # favourite = SECOND-named team
    "team_dog": "Saudi Arabia",   # underdog  = FIRST-named team
    "match_profile": "LOW_EVENT",

    "host_team": None, "host_type": None,
    "host_bump_by_type": {"host": 0.35, "home": 0.18, None: 0.0},
    "anchor_includes_host": True,

    # market_wdl["fav"] = Uruguay win; ["dog"] = Saudi win.
    "market_wdl": {"fav": 0.65, "draw": 0.21, "dog": 0.135},
    "ou_line": 2.5,
    "market_p_over": 0.43,        # market ~43% for 3+ (under lean)
    "reconcile_threshold": 0.02,

    "xg_fav": 1.45,   # URU (anchor already prices URU's finishing drought; do NOT double-subtract)
    "xg_dog": 0.55,   # KSA
    "dog_lambda_floor": 0.40,

    "poss_fav": 0.60, "poss_dog": 0.40,
    "dominance_concentration": 18.0,

    "heat_pct": 0.065,
    "heat_affects_goals": False,

    # Per-90 base rates (PRE-heat). fav=URU, dog=KSA.
    "sot":     {"fav": 5.0, "dog": 2.8},   # combined ~7 (WC norm — do NOT inflate)
    "corners": {"fav": 5.5, "dog": 3.0},
    "fouls":   {"fav": 11.0, "dog": 13.5}, # KSA deep/defending -> more fouls
    "cards":   {"fav": 1.8, "dog": 2.2},   # slight KSA edge, high tie risk
    "offsides":{"fav": 1.8, "dog": 1.5},   # URU more attacking volume; KSA counters

    "split_goals_h1":   0.45,
    "split_cards_h1":   0.35,
    "split_corners_h1": 0.48,
    "split_sot_h1":     0.48,
    "trail_push": 1.20,
    "card_2h_heat_bump": 1.12,

    "h2_ease_enabled": True,
    "h2_ease_factor": 0.825,
    "h2_ease_lead": 2,

    "sot_combined_cap": 7.5,
    "sot_combined_target": 7.0,

    "pen_lambda": 0.40,   # Q5: P(1+ pen) = 1-e^-0.40 ~ 33%

    "grid_goals_max": 12,
    "reg_weight": 0.008,
    "lam_grid": np.round(np.arange(0.90, 2.41, 0.05), 2),
    "mu_grid":  np.round(np.arange(0.40, 1.31, 0.05), 2),
    "rho_grid": np.round(np.arange(-0.12, 0.121, 0.02), 3),
    "rho_cap": 0.12,

    # Player props (two scenarios each).
    "nunez":     {"start_involve": 0.285, "sub_involve": 0.080},    # URU goals; P ~0.37 / ~0.12
    "aldawsari": {"start_involve": 0.317, "bench_involve": 0.113},  # KSA goals; P ~0.16 / ~0.06
}

CATS = {
    "Q1 Saudi Arabia 2+ offsides":             "EVENT_COUNT",
    "Q2 Both teams 1+ SOT in 2nd half":        "EVENT_COUNT",
    "Q3 Saudi more cards than Uruguay":        "EVENT_COUNT",
    "Q4 Uruguay more goals than Saudi (2H)":   "MATCH_OUTCOME",
    "Q5 Penalty awarded in match":             "EVENT_COUNT",
    "Q6 Saudi more fouls than Uruguay":        "EVENT_COUNT",
    "Q7 Uruguay more SOT than Saudi (2H)":     "TEAM_ATTACK",
    "Q8 Saudi Arabia win":                     "MATCH_OUTCOME",
    "Q9a Al-Dawsari score or assist — STARTS": "PLAYER_PROP",
    "Q9b Al-Dawsari score or assist — BENCHED":"PLAYER_PROP",
    "Q10a Darwin Nunez score — STARTS":        "PLAYER_PROP",
    "Q10b Darwin Nunez score — SUB":           "PLAYER_PROP",
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


def _grid_search(lam_arr, mu_arr, rho_arr, cfg, xg_f, xg_d, px_cache):
    kmax, tgt, reg, rho_cap = cfg["grid_goals_max"], cfg["market_wdl"], cfg["reg_weight"], cfg["rho_cap"]
    floor = cfg["dog_lambda_floor"]
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
                loss = ((f - tgt["fav"]) ** 2 + (d - tgt["draw"]) ** 2 +
                        (g - tgt["dog"]) ** 2 + reg_lam + reg_mu)
                if best is None or loss < best[0]:
                    best = (loss, lmbda, mu, rho)
    return best


def anchor_goal_model(cfg):
    kmax, floor = cfg["grid_goals_max"], cfg["dog_lambda_floor"]
    hf = (1.0 - cfg["heat_pct"]) if cfg["heat_affects_goals"] else 1.0
    xg_f, xg_d = cfg["xg_fav"] * hf, cfg["xg_dog"] * hf
    px_cache = {}
    _, lmbda, mu, rho = _grid_search(cfg["lam_grid"], cfg["mu_grid"], cfg["rho_grid"], cfg, xg_f, xg_d, px_cache)
    lam_f = np.round(np.arange(max(0.05, lmbda - 0.03), lmbda + 0.031, 0.0125), 4)
    mu_f = np.round(np.arange(max(floor, mu - 0.03), mu + 0.031, 0.0125), 4)
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
    uru_goals, ksa_goals = sample_goals(P, n, rng)   # fav=URU, dog=KSA

    k = cfg["dominance_concentration"]
    d = rng.beta(cfg["poss_fav"] * k, (1 - cfg["poss_fav"]) * k, size=n)
    scale_fav = d / cfg["poss_fav"]          # URU
    scale_dog = (1.0 - d) / cfg["poss_dog"]  # KSA

    g1f = rng.binomial(uru_goals, cfg["split_goals_h1"])
    g1d = rng.binomial(ksa_goals, cfg["split_goals_h1"])
    g2f, g2d = uru_goals - g1f, ksa_goals - g1d
    trailing_dog = g1f > g1d
    trailing_fav = g1d > g1f

    uru_leads_2plus = (g1f - g1d) >= cfg["h2_ease_lead"]
    ease = cfg["h2_ease_factor"] if cfg["h2_ease_enabled"] else 1.0
    mult_fav_2h = np.where(trailing_fav, cfg["trail_push"], np.where(uru_leads_2plus, ease, 1.0))
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

    off_f = rng.poisson(cfg["offsides"]["fav"] * heat * scale_fav)   # URU
    off_d = rng.poisson(cfg["offsides"]["dog"] * heat * scale_dog)   # KSA

    fouls_f = rng.poisson(cfg["fouls"]["fav"] * scale_dog)   # URU
    fouls_d = rng.poisson(cfg["fouls"]["dog"] * scale_fav)   # KSA

    bump = cfg["card_2h_heat_bump"]
    card_f_h1 = rng.poisson(cfg["cards"]["fav"] * cfg["split_cards_h1"], size=n)
    card_d_h1 = rng.poisson(cfg["cards"]["dog"] * cfg["split_cards_h1"], size=n)
    card_f_h2 = rng.poisson(cfg["cards"]["fav"] * (1 - cfg["split_cards_h1"]) * bump, size=n)
    card_d_h2 = rng.poisson(cfg["cards"]["dog"] * (1 - cfg["split_cards_h1"]) * bump, size=n)
    cards_f, cards_d = card_f_h1 + card_f_h2, card_d_h1 + card_d_h2

    pens = rng.poisson(cfg["pen_lambda"], size=n)

    # Players: Nunez "score" off URU goals; Al-Dawsari "score or assist" off KSA goals.
    nunez_start = (rng.binomial(uru_goals, cfg["nunez"]["start_involve"]) >= 1).mean()
    nunez_sub = (rng.binomial(uru_goals, cfg["nunez"]["sub_involve"]) >= 1).mean()
    ald_start = (rng.binomial(ksa_goals, cfg["aldawsari"]["start_involve"]) >= 1).mean()
    ald_bench = (rng.binomial(ksa_goals, cfg["aldawsari"]["bench_involve"]) >= 1).mean()

    total_goals = uru_goals + ksa_goals

    q = {}
    q["Q1 Saudi Arabia 2+ offsides"]          = (off_d >= 2).mean()
    q["Q2 Both teams 1+ SOT in 2nd half"]     = ((sot_f_h2 >= 1) & (sot_d_h2 >= 1)).mean()
    q["Q3 Saudi more cards than Uruguay"]     = (cards_d > cards_f).mean()
    q["Q4 Uruguay more goals than Saudi (2H)"]= (g2f > g2d).mean()
    q["Q5 Penalty awarded in match"]          = (pens >= 1).mean()
    q["Q6 Saudi more fouls than Uruguay"]     = (fouls_d > fouls_f).mean()
    q["Q7 Uruguay more SOT than Saudi (2H)"]  = (sot_f_h2 > sot_d_h2).mean()
    q["Q8 Saudi Arabia win"]                  = np.triu(P, 1).sum()
    q["Q9a Al-Dawsari score or assist — STARTS"]  = ald_start
    q["Q9b Al-Dawsari score or assist — BENCHED"] = ald_bench
    q["Q10a Darwin Nunez score — STARTS"]     = nunez_start
    q["Q10b Darwin Nunez score — SUB"]        = nunez_sub

    diag = {
        "lambda": lmbda, "mu": mu, "rho": rho, "wdl": wdl_from_pmf(P), "sot_scale": sot_scale,
        "total_goals_mean": total_goals.mean(), "p_3plus": (total_goals >= 3).mean(),
        "p_ksa_score": (ksa_goals >= 1).mean(),
        "calib": {
            "sot_combined": (sot_f + sot_d).mean(),
            "corners_combined": (cor_f_h1 + cor_f_h2 + cor_d_h1 + cor_d_h2).mean(),
            "fouls_combined": (fouls_f + fouls_d).mean(), "cards_combined": (cards_f + cards_d).mean(),
            "cards_2h_mean": (card_f_h2 + card_d_h2).mean(),
            "sot_fav": sot_f.mean(), "sot_dog": sot_d.mean(),
            "offsides_fav": off_f.mean(), "offsides_dog": off_d.mean(),
            "goals_fav": uru_goals.mean(), "goals_dog": ksa_goals.mean(),
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

    print("=" * 86)
    print(f"  {cfg['team_dog']} vs {cfg['team_fav']} — 2026 WC Group H, Hard Rock Miami (open-air, hot)")
    print(f"  {n:,} sims in {elapsed:.3f}s  |  profile={cfg['match_profile']}  host OFF  |  FAV = {cfg['team_fav']} (2nd-named)")
    print("=" * 86)

    f, dd, g = diag["wdl"]
    m = cfg["market_wdl"]
    print("\nGOAL MODEL (Dixon-Coles, anchored to de-vigged market)")
    print(f"  fitted lambda(URU)={diag['lambda']:.2f}  lambda(KSA)={diag['mu']:.2f}  rho={diag['rho']:+.3f}"
          f"   (KSA floored at {cfg['dog_lambda_floor']:.2f})")
    print(f"  realized W/D/L : {f:.3f} / {dd:.3f} / {g:.3f}   (URU / draw / KSA)")
    print(f"  market  W/D/L : {m['fav']:.3f} / {m['draw']:.3f} / {m['dog']:.3f}")
    print(f"  P(KSA score) = {100*diag['p_ksa_score']:.1f}%")

    model_p3, mkt_p3 = diag["p_3plus"], cfg["market_p_over"]
    gap = model_p3 - mkt_p3
    print("\nO/U-vs-1X2 RECONCILIATION  (line {:.1f}, over = 3+)".format(cfg["ou_line"]))
    print(f"  1X2-implied total = {diag['total_goals_mean']:.2f};  1X2-implied P(3+) = {100*model_p3:.1f}%")
    print(f"  market O/U implies P(3+) ~ {100*mkt_p3:.0f}%   ->  gap = {100*gap:+.1f}pp", end="")
    if abs(gap) > cfg["reconcile_threshold"]:
        print(f"   ** DISAGREE >{100*cfg['reconcile_threshold']:.0f}pp -> DEFAULT TO 1X2 ({100*model_p3:.1f}%) **")
    else:
        print("   (agree within threshold)")

    c = diag["calib"]
    print("\nCALIBRATION (full-match means; heat dampens events, NOT goals)")
    print(f"  SOT combined   : {c['sot_combined']:.2f}   (norm ~{WC_NORMS['sot']})"
          f"{'  [rescaled x%.3f]' % diag['sot_scale'] if diag['sot_scale'] != 1.0 else ''}")
    print(f"     URU / KSA   : {c['sot_fav']:.2f} / {c['sot_dog']:.2f}")
    print(f"  Corners combined: {c['corners_combined']:.2f}  (norm ~{WC_NORMS['corners']})")
    print(f"  Fouls combined : {c['fouls_combined']:.2f}  (norm ~{WC_NORMS['fouls']})")
    print(f"  Cards combined : {c['cards_combined']:.2f}  (2H mean {c['cards_2h_mean']:.2f}; card-heavy ref)")
    print(f"  Offsides URU/KSA: {c['offsides_fav']:.2f} / {c['offsides_dog']:.2f}")
    print(f"  Goals URU/KSA  : {c['goals_fav']:.2f} / {c['goals_dog']:.2f}")

    no_heat = simulate({**copy.deepcopy(cfg), "heat_pct": 0.0})[1]["calib"]
    print("\nHEAT DIAGNOSTIC (event rates, no-heat -> heat)")
    print(f"  SOT combined : {no_heat['sot_combined']:.2f} -> {c['sot_combined']:.2f}"
          f"   corners : {no_heat['corners_combined']:.2f} -> {c['corners_combined']:.2f}"
          f"   (goals unchanged: heat_affects_goals={cfg['heat_affects_goals']})")

    print("\nPROP PROBABILITIES (95% CI, category)  — Q9/Q10 show BOTH lineup scenarios")
    print("-" * 86)
    for label, p in q.items():
        lo, hi = fmt_ci(p, n)
        print(f"  {label:<44s} {100*p:5.1f}%  [{100*lo:4.1f},{100*hi:4.1f}]  {CATS[label]:<13s}{coin_flag(p)}")
    print("-" * 86)

    print("\nSENSITIVITY — Uruguay xG +/-0.25 (ease ON) and 2H-easing OFF (base xG)")
    cols = []
    for dx in (-0.25, 0.0, +0.25):
        r, _ = simulate({**copy.deepcopy(cfg), "xg_fav": round(cfg["xg_fav"] + dx, 2)})
        cols.append((f"xG{cfg['xg_fav']+dx:+.2f}", r))
    cols.append(("easeOFF", simulate({**copy.deepcopy(cfg), "h2_ease_enabled": False})[0]))
    print("  {:<44s}".format("prop") + "".join(f"{name:>10s}" for name, _ in cols))
    for label in q:
        print("  {:<44s}".format(label) + "".join(f"{100*r[label]:9.1f}%" for _, r in cols))
    print("  (compare last two columns on Q2/Q4/Q7 to read game-state fragility)")

    print("\nRBP_TRACKER TAGS (append post-match with outcomes + RBP):")
    print(f"  match profile = {cfg['match_profile']}")
    for label in q:
        print(f"    {label:<44s} -> {CATS[label]}")
    print("=" * 86)


if __name__ == "__main__":
    print_report(CFG)
