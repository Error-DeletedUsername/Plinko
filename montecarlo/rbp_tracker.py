"""RBP+ tracker — Jump Trading Probability Cup.

Tests one hypothesis: is my systematic below-crowd lean (leaning "no, this
won't happen" more than the crowd) a genuine EDGE or a liability BIAS?
Broken down overall, per question category, and by game profile.

RBP+ measures how much I beat (>0) or lost to (<0) the crowd on each question.
Stdlib only. Add a match by appending a dict to GAMES.
"""

from collections import defaultdict

CATEGORIES = ["PLAYER_PROP", "TEAM_ATTACK", "MATCH_OUTCOME", "EVENT_COUNT", "CONJUNCTIVE"]

# Game profile rule: LOW_EVENT (<=2 total goals, cagey), HIGH_EVENT (>=3 goals
# or a blowout), MODERATE_EVENT in between.
GAMES = [
    {"name": "Qatar vs Switzerland", "profile": "MODERATE_EVENT", "questions": [
        {"q": "HT both teams 1+ SOT",            "me": 41, "crowd": 59, "outcome": 1, "rbp": -13.32, "cat": "EVENT_COUNT"},
        {"q": "Switzerland 2+ offsides",         "me": 47, "crowd": 53, "outcome": 0, "rbp":  10.07, "cat": "EVENT_COUNT"},
        {"q": "Switzerland 1+ card 2nd half",    "me": 57, "crowd": 57, "outcome": 0, "rbp":   3.37, "cat": "EVENT_COUNT"},
        {"q": "both score AND 3+ goals",         "me": 18, "crowd": 31, "outcome": 0, "rbp":   9.85, "cat": "CONJUNCTIVE"},
        {"q": "Qatar 2+ SOT 2nd half",           "me": 18, "crowd": 39, "outcome": 1, "rbp": -25.85, "cat": "TEAM_ATTACK"},
        {"q": "Qatar more fouls than Swiss",     "me": 65, "crowd": 64, "outcome": 1, "rbp":   3.10, "cat": "EVENT_COUNT"},
        {"q": "Qatar score 1+",                  "me": 28, "crowd": 40, "outcome": 1, "rbp": -11.40, "cat": "TEAM_ATTACK"},
        {"q": "Qatar 2+ SOT full match",         "me": 39, "crowd": 57, "outcome": 1, "rbp": -14.30, "cat": "TEAM_ATTACK"},
        {"q": "Akram Afif 1+ SOT",               "me": 40, "crowd": 46, "outcome": 0, "rbp":   8.90, "cat": "PLAYER_PROP"},
        {"q": "Granit Xhaka 1+ SOT",             "me": 14, "crowd": 48, "outcome": 0, "rbp":  25.76, "cat": "PLAYER_PROP"},
    ]},
    {"name": "Brazil vs Morocco", "profile": "LOW_EVENT", "questions": [
        {"q": "Brazil 2+ offsides",              "me": 52, "crowd": 59, "outcome": 0, "rbp":  11.81, "cat": "EVENT_COUNT"},
        {"q": "Morocco more fouls than Brazil",  "me": 58, "crowd": 61, "outcome": 0, "rbp":   6.88, "cat": "EVENT_COUNT"},
        {"q": "Morocco scores first in 2nd half","me": 30, "crowd": 32, "outcome": 0, "rbp":   3.72, "cat": "TEAM_ATTACK"},
        {"q": "Brazil more goals 2nd half",      "me": 45, "crowd": 55, "outcome": 0, "rbp":  13.86, "cat": "MATCH_OUTCOME"},
        {"q": "Morocco more SOT than Brazil 2H", "me": 33, "crowd": 33, "outcome": 0, "rbp":   2.86, "cat": "TEAM_ATTACK"},
        {"q": "Brazil more cards than Morocco",  "me": 38, "crowd": 39, "outcome": 1, "rbp":   1.09, "cat": "EVENT_COUNT"},
        {"q": "both score AND 3+ goals",         "me": 33, "crowd": 47, "outcome": 0, "rbp":  15.90, "cat": "CONJUNCTIVE"},
        {"q": "Brazil win",                      "me": 58, "crowd": 67, "outcome": 0, "rbp":  14.39, "cat": "MATCH_OUTCOME"},
        {"q": "Morocco 5+ corners",              "me": 34, "crowd": 38, "outcome": 0, "rbp":   6.42, "cat": "EVENT_COUNT"},
        {"q": "match tied at halftime",          "me": 40, "crowd": 42, "outcome": 1, "rbp":   0.72, "cat": "MATCH_OUTCOME"},
    ]},
]


def analyze(qs):
    """Split a set of questions into below/above-crowd and score each side."""
    below = [x for x in qs if x["me"] < x["crowd"]]
    above = [x for x in qs if x["me"] > x["crowd"]]
    return {
        "n": len(qs),
        "below": below, "above": above,
        "below_rate": len(below) / len(qs) if qs else 0.0,
        # When below, the lean is correct if the event did NOT happen (outcome 0).
        "below_correct": sum(1 for x in below if x["outcome"] == 0),
        "below_rbp": sum(x["rbp"] for x in below),
        # When above, the lean is correct if the event DID happen (outcome 1).
        "above_correct": sum(1 for x in above if x["outcome"] == 1),
        "above_rbp": sum(x["rbp"] for x in above),
        "net": sum(x["rbp"] for x in qs),
    }


def pct(num, den):
    return f"{(100 * num / den):.0f}%" if den else "  -"


def report_block(name, qs):
    s = analyze(qs)
    nb, na = len(s["below"]), len(s["above"])
    print(f"\n{name}  (n={s['n']})")
    print(f"  below-crowd rate : {nb}/{s['n']} ({pct(nb, s['n'])})")
    if nb:
        print(f"  when BELOW       : correct(NO)  {s['below_correct']}/{nb} ({pct(s['below_correct'], nb)})"
              f"  | RBP {s['below_rbp']:+7.2f}")
    if na:
        print(f"  when ABOVE       : correct(YES) {s['above_correct']}/{na} ({pct(s['above_correct'], na)})"
              f"  | RBP {s['above_rbp']:+7.2f}")
    print(f"  net RBP          : {s['net']:+7.2f}")
    if nb:
        verdict = "an EDGE" if s["below_rbp"] > 0 else "a BIAS"
        print(f"  >> below-crowd is {verdict} here (below RBP {s['below_rbp']:+.2f})")


def profile_report(all_qs):
    by_prof = defaultdict(list)
    for g in GAMES:
        by_prof[g["profile"]].append(sum(x["rbp"] for x in g["questions"]))
    # Average RBP per match, plus per-question below-crowd RBP within the profile.
    q_by_prof = defaultdict(list)
    for g in GAMES:
        for x in g["questions"]:
            q_by_prof[g["profile"]].append(x)
    for prof in sorted(by_prof, key=lambda p: -sum(by_prof[p]) / len(by_prof[p])):
        totals = by_prof[prof]
        below = [x for x in q_by_prof[prof] if x["me"] < x["crowd"]]
        below_rbp = sum(x["rbp"] for x in below)
        print(f"  {prof:<15} matches={len(totals)}  avg RBP/match {sum(totals)/len(totals):+7.2f}"
              f"  | below-crowd RBP {below_rbp:+7.2f}")


def main():
    all_qs = [x for g in GAMES for x in g["questions"]]
    print("=" * 64)
    print("  RBP+ TRACKER — below-crowd lean: edge or bias?")
    print(f"  {len(GAMES)} matches, {len(all_qs)} questions")
    print("=" * 64)

    print("\n##### OVERALL #####")
    report_block("OVERALL", all_qs)

    print("\n\n##### BY CATEGORY #####")
    for c in CATEGORIES:
        qs = [x for x in all_qs if x["cat"] == c]
        if qs:
            report_block(c, qs)

    print("\n\n##### GAME PROFILE (hypothesis: below-crowd wins low-event, loses high-event) #####")
    profile_report(all_qs)
    print("=" * 64)


if __name__ == "__main__":
    main()
