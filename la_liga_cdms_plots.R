# ============================================================
#  LA LIGA CDMs — the four dot plots  (CSV / tidyverse version)
#  No jsonlite. Reads events + players from CSV.
#
#  This version AUTO-DETECTS the column layout of your CSV:
#   - a "tags" text column ("[{'id': 1702}, ...]" or "1702;1801")
#   - split tag columns (tag1, tag2, ..., tags_0, ...)
#   - one-hot / boolean columns (yellow_card, accurate, ...)
#  and snake_case vs camelCase names (event_name vs eventName, ...).
#
#  Reproduces:
#    1. Win rate: after vs before
#    2. Win rate: post-card vs own season norm
#    3. Duel volume: after vs before
#    4. Duel volume: post-card vs own season norm
# ============================================================
library(tidyverse)

MIN_AFTER   <- 15     # minutes played after the card (per game)
MIN_SEASON  <- 900    # season minutes to qualify

# ------------------------------------------------------------
# 0. LOAD
# ------------------------------------------------------------
events  <- read_csv("C:/Users/milot/Moneyball Project/data/events_Spain.csv", show_col_types = FALSE)
players <- read_csv("C:/Users/milot/Moneyball Project/data/players.csv",      show_col_types = FALSE)

cat("events columns:\n");  print(names(events))
cat("players columns:\n"); print(names(players))

# ------------------------------------------------------------
# 1. AUTO-MAPPING BLOCK
# ------------------------------------------------------------
# helper: find the first column whose normalised name matches a candidate
norm_nm  <- function(x) gsub("[^a-z0-9]", "", tolower(x))
find_col <- function(df, cands) {
  hit <- names(df)[norm_nm(names(df)) %in% norm_nm(cands)]
  if (length(hit)) hit[1] else NA_character_
}
# helper: rename first matching candidate to a canonical name
canon <- function(df, canonical, cands) {
  if (canonical %in% names(df)) return(df)
  src <- find_col(df, cands)
  if (!is.na(src)) df <- rename(df, !!canonical := all_of(src))
  df
}

# (a) canonical event columns -------------------------------------------------
events <- events %>%
  canon("eventName",    c("eventName", "event_name", "event")) %>%
  canon("subEventName", c("subEventName", "sub_event_name", "subevent", "sub_event")) %>%
  canon("playerId",     c("playerId", "player_id")) %>%
  canon("matchId",      c("matchId", "match_id", "game_id")) %>%
  canon("matchPeriod",  c("matchPeriod", "match_period", "period")) %>%
  canon("eventSec",     c("eventSec", "event_sec", "seconds", "second")) %>%
  canon("start_x",      c("start_x", "startX", "pos_orig_x", "x_start"))

# if there is no start_x but a Wyscout-style "positions" text column,
# pull the first x out of it: "[{'y': 49, 'x': 31}, ...]"
if (!"start_x" %in% names(events) && "positions" %in% names(events)) {
  events <- events %>%
    mutate(start_x = as.numeric(
      str_match(as.character(positions), "'x'\\s*[:=]\\s*([0-9.]+)")[, 2]))
}

# (b) TIME: one continuous minute across both halves --------------------------
events <- events %>%
  mutate(
    abs_min = if ("match_time" %in% names(.)) match_time / 60
              else eventSec / 60 + case_when(matchPeriod == "1H" ~ 0,
                                             matchPeriod == "2H" ~ 45,
                                             matchPeriod == "E1" ~ 90,
                                             matchPeriod == "E2" ~ 105,
                                             TRUE ~ 0)
  )

# (c) TAGS: yellow card (id 1702) and duel won / accurate (id 1801) -----------
tag_id_cols <- names(events)[str_detect(names(events), "^tags?[_. ]?[0-9]+$")]
yellow_flag <- find_col(events, c("yellow_card", "yellowCard", "yellow card",
                                  "is_yellow", "is_yellow_card", "yellow"))
won_flag    <- find_col(events, c("accurate", "is_accurate", "won", "is_won",
                                  "duel_won", "success", "successful", "is_success"))

if ("tags" %in% names(events)) {
  # one text column holding all tag ids
  cat("tag mapping: using text column 'tags'\n")
  events <- events %>%
    mutate(yellow = str_detect(as.character(tags), "\\b1702\\b"),
           won    = str_detect(as.character(tags), "\\b1801\\b"))
} else if (length(tag_id_cols) > 0) {
  # tags split across tag1..tagN columns
  cat("tag mapping: combining columns", paste(tag_id_cols, collapse = ", "), "\n")
  tag_str <- events %>%
    select(all_of(tag_id_cols)) %>%
    mutate(across(everything(), as.character)) %>%
    unite("all_tags", everything(), sep = " ", na.rm = TRUE) %>%
    pull(all_tags)
  events <- events %>%
    mutate(yellow = str_detect(tag_str, "\\b1702\\b"),
           won    = str_detect(tag_str, "\\b1801\\b"))
} else if (!is.na(yellow_flag) && !is.na(won_flag)) {
  # one-hot / boolean columns (common in pre-processed Wyscout CSVs)
  cat("tag mapping: using boolean columns '", yellow_flag, "' and '",
      won_flag, "'\n", sep = "")
  to_bool <- function(x) {
    if (is.logical(x)) x %in% TRUE
    else if (is.numeric(x)) x %in% 1
    else tolower(trimws(as.character(x))) %in% c("1", "true", "t", "yes", "y")
  }
  events <- events %>%
    mutate(yellow = to_bool(.data[[yellow_flag]]),
           won    = to_bool(.data[[won_flag]]))
} else {
  stop(paste0(
    "Could not find tag information in events_Spain.csv.\n",
    "Looked for: a 'tags' column, tag1..tagN columns, or boolean columns ",
    "like yellow_card / accurate.\nYour columns are:\n  ",
    paste(names(events), collapse = ", "),
    "\nEdit section 1(c) and point 'yellow' and 'won' at the right columns."))
}

# (d) PLAYER ROLE + NAME ------------------------------------------------------
players <- players %>%
  canon("wyId",      c("wyId", "wy_id", "playerId", "player_id")) %>%
  canon("shortName", c("shortName", "short_name", "player_name", "name"))

role_col <- find_col(players, c("role_name", "roleName", "role", "position",
                                "role_code2", "position_name"))
if (is.na(role_col)) {
  stop(paste0("Could not find a role/position column in players.csv.\n",
              "Your columns are:\n  ", paste(names(players), collapse = ", ")))
}

players_lookup <- players %>%
  transmute(playerId = wyId,
            player   = shortName,
            # raw JSON exports sometimes keep role as "{'code2': 'MD', 'name': 'Midfielder', ...}"
            role     = if_else(str_detect(as.character(.data[[role_col]]), "Midfielder"),
                               "Midfielder", as.character(.data[[role_col]])))

# quick checks — these should look sensible before you continue
events %>% summarise(yellows = sum(yellow, na.rm = TRUE)) %>% print()   # ~1863
events %>% filter(eventName == "Duel",
                  subEventName == "Ground defending duel") %>%
  summarise(win_rate = mean(won, na.rm = TRUE)) %>% print()             # ~0.50

# ------------------------------------------------------------
# 2. MINUTES, MIDFIELDERS, SUBTYPES
# ------------------------------------------------------------
span_all <- events %>%
  group_by(matchId, playerId) %>%
  summarise(entry = min(abs_min), exit = max(abs_min), .groups = "drop")

season_min <- span_all %>%
  mutate(mins = exit - entry) %>%
  group_by(playerId) %>%
  summarise(season_mins = sum(mins), .groups = "drop")

midfielders <- players_lookup %>%
  filter(role == "Midfielder") %>%
  left_join(season_min, by = "playerId") %>%
  filter(season_mins >= MIN_SEASON)

# subtype from season-average pitch position (tertiles)
mid_type <- events %>%
  inner_join(select(midfielders, playerId), by = "playerId") %>%
  filter(!is.na(start_x)) %>%
  group_by(playerId) %>%
  summarise(avg_x = mean(start_x), .groups = "drop") %>%
  mutate(subtype = cut(avg_x,
                       breaks = quantile(avg_x, c(0, 1/3, 2/3, 1)),
                       labels = c("CDM (deep)", "CM (central)", "CAM (advanced)"),
                       include.lowest = TRUE))

# ------------------------------------------------------------
# 3. FIRST YELLOWS -> CDM EPISODES
# ------------------------------------------------------------
first_yellows <- events %>%
  filter(eventName == "Foul", yellow) %>%
  arrange(matchId, playerId, abs_min) %>%
  group_by(matchId, playerId) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  select(matchId, playerId, card_min = abs_min)

ep_cdm <- first_yellows %>%
  inner_join(select(midfielders, playerId), by = "playerId") %>%
  inner_join(mid_type %>% filter(subtype == "CDM (deep)") %>% select(playerId),
             by = "playerId") %>%
  left_join(span_all, by = c("matchId", "playerId")) %>%
  mutate(mins_before = card_min - entry,
         mins_after  = exit - card_min) %>%
  filter(mins_before > 0, mins_after >= MIN_AFTER) %>%
  arrange(matchId, playerId, card_min) %>%
  group_by(matchId, playerId) %>%
  slice_head(n = 1) %>%
  ungroup()

booked_matches <- ep_cdm %>% distinct(playerId, matchId)

# every ground defending duel inside those episodes, phased
cdm_duels <- events %>%
  inner_join(select(ep_cdm, matchId, playerId, card_min),
             by = c("matchId", "playerId")) %>%
  filter(eventName == "Duel", subEventName == "Ground defending duel") %>%
  mutate(phase = if_else(abs_min < card_min, "before", "after"))

# ------------------------------------------------------------
# 4. WIN RATE  (graphs 1 & 2)
# ------------------------------------------------------------
cdm_players_all <- cdm_duels %>%
  group_by(playerId, phase) %>%
  summarise(duels = n(), won = sum(won, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = phase, values_from = c(duels, won), values_fill = 0) %>%
  filter(duels_before > 0, duels_after > 0) %>%
  mutate(win_before = 100 * won_before / duels_before,
         win_after  = 100 * won_after  / duels_after,
         d_win      = win_after - win_before)

cat("CDMs at >=", MIN_AFTER, "min post-card:", nrow(cdm_players_all), "\n")

# season baseline EXCLUDING the booked matches
clean_duel <- events %>%
  filter(eventName == "Duel", subEventName == "Ground defending duel") %>%
  anti_join(booked_matches, by = c("playerId", "matchId")) %>%
  group_by(playerId) %>%
  summarise(clean_duels  = n(),
            clean_winpct = 100 * mean(won, na.rm = TRUE), .groups = "drop")

cdm_vs_clean <- cdm_players_all %>%
  left_join(clean_duel, by = "playerId") %>%
  left_join(select(midfielders, playerId, player), by = "playerId") %>%
  mutate(before_vs_clean = win_before - clean_winpct,
         after_vs_clean  = win_after  - clean_winpct)

# ------------------------------------------------------------
# 5. DUEL VOLUME  (graphs 3 & 4)
# ------------------------------------------------------------
cdm_mins <- ep_cdm %>%
  group_by(playerId) %>%
  summarise(mins_before = sum(mins_before),
            mins_after  = sum(mins_after), .groups = "drop")

clean_mins <- span_all %>%
  mutate(mins = exit - entry) %>%
  anti_join(booked_matches, by = c("playerId", "matchId")) %>%
  group_by(playerId) %>%
  summarise(clean_mins = sum(mins), .groups = "drop")

cdm_vol <- cdm_duels %>%
  group_by(playerId, phase) %>%
  summarise(duels = n(), .groups = "drop") %>%
  pivot_wider(names_from = phase, values_from = duels, values_fill = 0) %>%
  rename(duels_before = before, duels_after = after) %>%
  left_join(cdm_mins,   by = "playerId") %>%
  left_join(clean_mins, by = "playerId") %>%
  left_join(select(clean_duel, playerId, clean_duels), by = "playerId") %>%
  filter(mins_before > 0, mins_after > 0, clean_mins > 0) %>%
  mutate(before_p90       = 90 * duels_before / mins_before,
         after_p90        = 90 * duels_after  / mins_after,
         season_p90       = 90 * clean_duels  / clean_mins,
         d_p90            = after_p90 - before_p90,
         after_vs_season  = after_p90 - season_p90) %>%
  left_join(select(midfielders, playerId, player), by = "playerId")

# ------------------------------------------------------------
# 6. P-VALUES for all four
# ------------------------------------------------------------
p_row <- function(x, label) {
  x  <- x[is.finite(x)]
  tt <- t.test(x)
  tibble(graph   = label,
         n       = length(x),
         mean    = round(unname(tt$estimate), 2),
         ci_low  = round(tt$conf.int[1], 2),
         ci_high = round(tt$conf.int[2], 2),
         p_value = signif(tt$p.value, 4))
}

bind_rows(
  p_row(cdm_players_all$d_win,       "1. Win rate: after vs before"),
  p_row(cdm_vs_clean$after_vs_clean, "2. Win rate: post-card vs season norm"),
  p_row(cdm_vol$d_p90,               "3. Duel volume: after vs before"),
  p_row(cdm_vol$after_vs_season,     "4. Duel volume: post-card vs season norm")
) %>% print(n = 10)

# ------------------------------------------------------------
# 7. THE PLOT FUNCTION
# ------------------------------------------------------------
dot_plot <- function(df, col, title, sub, xlab, pct = FALSE) {
  d <- df %>% filter(is.finite(.data[[col]]))
  m <- mean(d[[col]])

  ggplot(d, aes(x = .data[[col]], y = 0)) +
    geom_hline(yintercept = 0, linewidth = .6, colour = "grey30") +
    geom_vline(xintercept = 0, linewidth = .5, colour = "grey40") +
    geom_point(aes(fill = .data[[col]] > 0),
               position = position_jitter(height = .12, width = 0, seed = 1),
               shape = 21, size = 3.2, colour = "white", stroke = .6, alpha = .9) +
    geom_point(x = m, y = 0, colour = "firebrick", size = 5, shape = 18) +
    annotate("text", x = m, y = .30, label = sprintf("mean %+.1f", m),
             colour = "firebrick", fontface = "bold", size = 4) +
    scale_fill_manual(values = c(`TRUE` = "steelblue", `FALSE` = "grey60"),
                      guide = "none") +
    scale_y_continuous(limits = c(-.45, .45), breaks = NULL, name = NULL) +
    scale_x_continuous(name = xlab,
                       breaks = scales::pretty_breaks(n = 9),
                       labels = function(x) paste0(ifelse(x > 0, "+", ""), x,
                                                   if (pct) "%" else "")) +
    labs(title = title, subtitle = sprintf("%s | n = %d", sub, nrow(d))) +
    theme_minimal(base_size = 12) +
    theme(panel.grid.major.y = element_blank(),
          panel.grid.minor   = element_blank(),
          plot.title = element_text(face = "bold"))
}

sub1 <- sprintf("Each dot = one player | >=%d season mins, >=%d min post-card",
                MIN_SEASON, MIN_AFTER)
sub2 <- "Baseline excludes booked matches"

# ------------------------------------------------------------
# 8. THE FOUR GRAPHS
# ------------------------------------------------------------
dot_plot(cdm_players_all, "d_win",
         "La Liga CDMs: ground-duel win rate before vs after a first yellow",
         sub1, "win-rate change after first yellow (percentage points)", pct = TRUE)

dot_plot(cdm_vs_clean, "after_vs_clean",
         "La Liga CDMs: post-card duel win rate vs their own season norm",
         sub2, "post-card win rate minus own season baseline (percentage points)",
         pct = TRUE)

dot_plot(cdm_vol, "d_p90",
         "La Liga CDMs: duel volume before vs after a first yellow",
         sub1, "change in ground duels per 90 (after - before)")

dot_plot(cdm_vol, "after_vs_season",
         "La Liga CDMs: post-card duel volume vs their own season norm",
         sub2, "post-card duels per 90 minus own season baseline")
