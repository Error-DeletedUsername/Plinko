"""
Monte Carlo match simulator — Jump Trading Probability Cup
Match: Belgium vs Egypt, 2026 World Cup Group G, Lumen Field Seattle.
OPEN-AIR, NWS Heat Advisory ~90F at noon PT kickoff — heat is a live tempo
variable. Belgium favoured & possession-dominant (4-2-3-1); Egypt sit deep and
counter via Salah/Marmoush. LOW_EVENT match. Neither side is a host.

Weighted-Brier scoring: calibration over confidence; never 0/100; 42-58% flagged.
Same anchored Dixon-Coles / dominance / half-split / game-state-easing engine as
prior matches. Match-specific pieces:

  HEAT TEMPO DAMPENER (~6.5%): applied to the SOT/corners/offsides RESEARCH rates
  (neutral-condition samples), NOT to the goal total. Rationale = the same
  anti-double-count rule as the host term: the de-vigged 1X2/O/U already price the
  90F conditions, so dampening the goal total too would double-count. Toggle
  `heat_affects_goals` flips this. Cards are DECOUPLED from heat here and instead
  get a 2H heat-fatigue BUMP in the card block (heat -> more late cards).

  O/U-vs-1X2 RECONCILIATION: if the 1X2-implied P(3+) and the 2.5 O/U disagree by
  >2pp, flag it and default to the 1X2 (we do not shade totals to the O/U).

  UNDERDOG FLOOR: Egypt lambda floored at 0.40 (scoreless in 4 of last 6) — kept
  off zero but not inflated.
"""

import time
import copy
import numpy as np

CFG = {
    "seed": 20260615,
    "n_sims": 200_000,

    "team_fav": "Belgium",
    "team_dog": "Egypt",
    "match_profile": "LOW_EVENT",   # rbp_tracker tag

    # Host term (ported, OFF this match).
    "host_team": None, "host_type": None,
    "host_bump_by_type": {"host": 0.35, "home": 0.18, None: 0.0},
    "anchor_includes_host": True,

    "market_wdl": {"fav": 0.60, "draw": 0.23, "dog": 0.17},
    "ou_line": 2.5,
    "market_p_over": 0.47,        # market ~47% for 3+ (slight under lean)
    "reconcile_threshold": 0.02,  # flag if 1X2-implied P(3+) differs by > this

    "xg_fav": 1.65,   # BEL
    "xg_dog": 0.70,   # EGY
    "dog_lambda_floor": 0.40,

    "poss_fav": 0.62, "poss_dog": 0.38,
    "dominance_concentration": 18.0,

    # HEAT
    "heat_pct": 0.065,            # ~6.5% downward tempo on event rates
    "heat_affects_goals": False,  # keep goals market-anchored (no double-count)

    # Per-90 base rates (PRE-heat) with research-confidence flags.
    "sot":     {"fav": 6.0, "dog": 3.0},   # BEL MEDIUM (friendly-inflated?), EGY LOW
    "corners": {"fav": 6.5, "dog": 3.0},   # MEDIUM / LOW
    "fouls":   {"fav": 10.5, "dog": 13.0}, # MEDIUM (EGY deeper -> more fouls)
    "cards":   {"fav": 1.3, "dog": 1.3},   # ref strict (~4.77/g) but discounted hard
                                           #   for WC opener + disciplined sides
    "offsides":{"fav": 1.5, "dog": 1.0},   # LOW

    "split_goals_h1":   0.45,
    "split_cards_h1":   0.35,
    "split_corners_h1": 0.48,
    "split_sot_h1":     0.48,
    "trail_push": 1.20,
    "card_2h_heat_bump": 1.12,   # heat fatigue -> more 2nd-half cards

    # Game-state easing (heavy favourite).
    "h2_ease_enabled": True,
    "h2_ease_factor": 0.825,
    "h2_ease_lead": 2,

    # Standard SOT guardrail: trims combined SOT to the ~7 WC norm. This is what
    # makes Q5 a coin flip (~51). NOTE: your "combined ~9" descriptor conflicts with
    # your "Q5 ~51" target — they cannot both hold. Trusting the Q5 target + the WC
    # norm + your own MEDIUM-confidence flag on BEL's inflated 6.0. To run the hot
    # version instead (combined ~8.5, Q5 ~66), set cap=9.5 / target=8.6.
    "sot_combined_cap": 7.5,
    "sot_combined_target": 7.0,

    "pen_lambda": 0.30, "red_lambda": 0.18,   # (not a question here; for completeness)

    "grid_goals_max": 12,
    "reg_weight": 0.008,
    "lam_grid": np.round(np.arange(1.00, 2.51, 0.05), 2),
    "mu_grid":  np.round(np.arange(0.40, 1.41, 0.05), 2),
    "rho_grid": np.round(np.arange(-0.12, 0.121, 0.02), 3),
    "rho_cap": 0.12,

    # Player props.
    "tielemans": {"exp_sot": 0.36, "side": "fav"},                 # deep pivot, fade-ish
    "trezeguet": {"start_involve": 0.40, "bench_involve": 0.12},   # suppressed underdog winger
}

# Question -> category tag (for rbp_tracker).
CATS = {
    "Q1 Egypt more fouls than Belgium":        "EVENT_COUNT",
    "Q2 2+ total cards in 2nd half":           "EVENT_COUNT",
    "Q3 Both score AND 3+ total goals":        "CONJUNCTIVE",
    "Q4 Egypt more SOT than Belgium (2H)":     "TEAM_ATTACK",
    "Q5 4+ total SOT in 2nd half":             "EVENT_COUNT",
    "Q6 Belgium 2+ offsides":                  "EVENT_COUNT",
    "Q7 Belgium win":                          "MATCH_OUTCOME",
    "Q8 Match tied at halftime":               "MATCH_OUTCOME",
    "Q9 Tielemans 1+ SOT":                     "PLAYER_PROP",
    "Q10a Trezeguet score or assist — STARTS": "PLAYER_PROP",
    "Q10b Trezeguet score or assist — BENCHED":"PLAYER_PROP",
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
    heat = 1.0 - cfg["heat_pct"]   # event-rate dampener

    lmbda, mu, rho, P = anchor_goal_model(cfg)
    bel_goals, egy_goals = sample_goals(P, n, rng)

    k = cfg["dominance_concentration"]
    d = rng.beta(cfg["poss_fav"] * k, (1 - cfg["poss_fav"]) * k, size=n)
    scale_fav = d / cfg["poss_fav"]
    scale_dog = (1.0 - d) / cfg["poss_dog"]

    g1f = rng.binomial(bel_goals, cfg["split_goals_h1"])
    g1d = rng.binomial(egy_goals, cfg["split_goals_h1"])
    trailing_dog = g1f > g1d
    trailing_fav = g1d > g1f

    bel_leads_2plus = (g1f - g1d) >= cfg["h2_ease_lead"]
    ease = cfg["h2_ease_factor"] if cfg["h2_ease_enabled"] else 1.0
    mult_fav_2h = np.where(trailing_fav, cfg["trail_push"], np.where(bel_leads_2plus, ease, 1.0))
    mult_dog_2h = np.where(trailing_dog, cfg["trail_push"], 1.0)

    # SOT (heat-dampened rates) with guardrail.
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

    # Fouls NOT heat-dampened (defensive-intensity stat; Q1 is a ratio anyway).
    fouls_f = rng.poisson(cfg["fouls"]["fav"] * scale_dog)
    fouls_d = rng.poisson(cfg["fouls"]["dog"] * scale_fav)

    # Cards: heat-decoupled from tempo, but 2H gets a heat-fatigue bump.
    bump = cfg["card_2h_heat_bump"]
    card_f_h2 = rng.poisson(cfg["cards"]["fav"] * (1 - cfg["split_cards_h1"]) * bump, size=n)
    card_d_h2 = rng.poisson(cfg["cards"]["dog"] * (1 - cfg["split_cards_h1"]) * bump, size=n)
    cards_2h = card_f_h2 + card_d_h2
    cards_full = (rng.poisson(cfg["cards"]["fav"] * cfg["split_cards_h1"], size=n) + card_f_h2 +
                  rng.poisson(cfg["cards"]["dog"] * cfg["split_cards_h1"], size=n) + card_d_h2)

    # Players.
    tiele_sot = rng.poisson(cfg["tielemans"]["exp_sot"] * scale_fav)
    trez_start = (rng.binomial(egy_goals, cfg["trezeguet"]["start_involve"]) >= 1).mean()
    trez_bench = (rng.binomial(egy_goals, cfg["trezeguet"]["bench_involve"]) >= 1).mean()

    total_goals = bel_goals + egy_goals
    both_score = (bel_goals >= 1) & (egy_goals >= 1)

    q = {}
    q["Q1 Egypt more fouls than Belgium"]      = (fouls_d > fouls_f).mean()
    q["Q2 2+ total cards in 2nd half"]         = (cards_2h >= 2).mean()
    q["Q3 Both score AND 3+ total goals"]      = (both_score & (total_goals >= 3)).mean()
    q["Q4 Egypt more SOT than Belgium (2H)"]   = (sot_d_h2 > sot_f_h2).mean()
    q["Q5 4+ total SOT in 2nd half"]           = ((sot_f_h2 + sot_d_h2) >= 4).mean()
    q["Q6 Belgium 2+ offsides"]                = (off_f >= 2).mean()
    q["Q7 Belgium win"]                        = np.tril(P, -1).sum()
    q["Q8 Match tied at halftime"]             = (g1f == g1d).mean()
    q["Q9 Tielemans 1+ SOT"]                   = (tiele_sot >= 1).mean()
    q["Q10a Trezeguet score or assist — STARTS"]  = trez_start
    q["Q10b Trezeguet score or assist — BENCHED"] = trez_bench

    diag = {
        "lambda": lmbda, "mu": mu, "rho": rho, "wdl": wdl_from_pmf(P), "sot_scale": sot_scale,
        "total_goals_mean": total_goals.mean(), "p_3plus": (total_goals >= 3).mean(),
        "p_egy_score": (egy_goals >= 1).mean(),
        "calib": {
            "sot_combined": (sot_f + sot_d).mean(),
            "corners_combined": (cor_f_h1 + cor_f_h2 + cor_d_h1 + cor_d_h2).mean(),
            "fouls_combined": (fouls_f + fouls_d).mean(), "cards_combined": cards_full.mean(),
            "cards_2h_mean": cards_2h.mean(),
            "sot_fav": sot_f.mean(), "sot_dog": sot_d.mean(),
            "offsides_fav": off_f.mean(), "offsides_dog": off_d.mean(),
            "goals_fav": bel_goals.mean(), "goals_dog": egy_goals.mean(),
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

    print("=" * 84)
    print(f"  {cfg['team_fav']} vs {cfg['team_dog']} — 2026 WC Group G, Lumen Field (open-air, ~90F)")
    print(f"  {n:,} vectorized sims in {elapsed:.3f}s   |   profile={cfg['match_profile']}  host OFF")
    print("=" * 84)

    f, dd, g = diag["wdl"]
    m = cfg["market_wdl"]
    print("\nGOAL MODEL (Dixon-Coles, anchored to de-vigged market)")
    print(f"  fitted lambda(BEL)={diag['lambda']:.2f}  lambda(EGY)={diag['mu']:.2f}  rho={diag['rho']:+.3f}"
          f"   (EGY floored at {cfg['dog_lambda_floor']:.2f})")
    print(f"  realized W/D/L : {f:.3f} / {dd:.3f} / {g:.3f}   (BEL / draw / EGY)")
    print(f"  market  W/D/L : {m['fav']:.3f} / {m['draw']:.3f} / {m['dog']:.3f}")
    print(f"  P(EGY score) = {100*diag['p_egy_score']:.1f}%")

    # O/U-vs-1X2 reconciliation.
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
    print("\nCALIBRATION (full-match means vs WC norms; heat dampens event rates, NOT goals)")
    print(f"  SOT combined   : {c['sot_combined']:.2f}   (norm ~{WC_NORMS['sot']}; trimmed to norm — BEL 6.0 was MEDIUM-confidence/inflated)"
          f"{'  [rescaled x%.3f]' % diag['sot_scale'] if diag['sot_scale'] != 1.0 else ''}")
    print(f"     BEL / EGY   : {c['sot_fav']:.2f} / {c['sot_dog']:.2f}")
    print(f"  Corners combined: {c['corners_combined']:.2f}  (norm ~{WC_NORMS['corners']})")
    print(f"  Fouls combined : {c['fouls_combined']:.2f}  (norm ~{WC_NORMS['fouls']})")
    print(f"  Cards combined : {c['cards_combined']:.2f}  (2H mean {c['cards_2h_mean']:.2f}; ref discounted hard for WC opener)")
    print(f"  Offsides BEL/EGY: {c['offsides_fav']:.2f} / {c['offsides_dog']:.2f}")
    print(f"  Goals BEL/EGY  : {c['goals_fav']:.2f} / {c['goals_dog']:.2f}")

    # Heat diagnostic (events only).
    no_heat = simulate({**copy.deepcopy(cfg), "heat_pct": 0.0})[1]["calib"]
    print("\nHEAT DIAGNOSTIC (event rates, no-heat -> heat)")
    print(f"  SOT combined : {no_heat['sot_combined']:.2f} -> {c['sot_combined']:.2f}"
          f"   corners : {no_heat['corners_combined']:.2f} -> {c['corners_combined']:.2f}"
          f"   (goals unchanged: heat_affects_goals={cfg['heat_affects_goals']})")

    print("\nPROP PROBABILITIES (with 95% CI, category)  — Q10 shows BOTH lineup scenarios")
    print("-" * 84)
    for label, p in q.items():
        lo, hi = fmt_ci(p, n)
        print(f"  {label:<42s} {100*p:5.1f}%  [{100*lo:4.1f},{100*hi:4.1f}]  {CATS[label]:<13s}{coin_flag(p)}")
    print("-" * 84)

    print("\nSENSITIVITY — Belgium xG +/-0.25 (ease ON) and 2H-easing toggled OFF (base xG)")
    cols = []
    for dx in (-0.25, 0.0, +0.25):
        r, _ = simulate({**copy.deepcopy(cfg), "xg_fav": round(cfg["xg_fav"] + dx, 2)})
        cols.append((f"xG{cfg['xg_fav']+dx:+.2f}", r))
    cols.append(("easeOFF", simulate({**copy.deepcopy(cfg), "h2_ease_enabled": False})[0]))
    print("  {:<42s}".format("prop") + "".join(f"{name:>10s}" for name, _ in cols))
    for label in q:
        print("  {:<42s}".format(label) + "".join(f"{100*r[label]:9.1f}%" for _, r in cols))
    print("  (compare last two columns on Q4/Q5 to read game-state fragility)")

    print("\nRBP_TRACKER TAGS (append post-match with outcomes + RBP):")
    print(f"  match profile = {cfg['match_profile']}")
    for label in q:
        print(f"    {label:<42s} -> {CATS[label]}")
    print("=" * 84)


if __name__ == "__main__":
    print_report(CFG)
