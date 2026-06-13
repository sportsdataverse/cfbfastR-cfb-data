# Idempotently create the espn_cfb_* release tags on the sportsdataverse-data publish repo.
# Run once (with GITHUB_PAT / SDV_GH_TOKEN that can write to both repos) before the first
# data run. "Already exists" errors are expected and ignored.
suppressPackageStartupMessages({ library(piggyback); library(cli) })

REPOS <- c("sportsdataverse/sportsdataverse-data")

TAGS <- list(
  espn_cfb_pbp               = "College Football play-by-play (from ESPN, enriched).",
  espn_cfb_team_box          = "College Football team box scores (from ESPN).",
  espn_cfb_player_box        = "College Football player box scores (from ESPN).",
  espn_cfb_adv_team          = "College Football advanced team box (EPA/success/explosiveness).",
  espn_cfb_adv_passing       = "College Football advanced passing box.",
  espn_cfb_adv_rushing       = "College Football advanced rushing box.",
  espn_cfb_adv_receiving     = "College Football advanced receiving box.",
  espn_cfb_adv_defensive     = "College Football advanced defensive box.",
  espn_cfb_adv_turnover      = "College Football turnover summary + luck.",
  espn_cfb_adv_drives        = "College Football drive efficiency summary.",
  espn_cfb_adv_situational   = "College Football situational EPA/success splits.",
  espn_cfb_play_participants  = "College Football per-play participants.",
  espn_cfb_drives            = "College Football drive-level table.",
  espn_cfb_game_rosters      = "College Football per-game rosters (one row per athlete per game).",
  espn_cfb_rosters           = "College Football season rosters (ESPN-derived, deduplicated).",
  espn_cfb_betting           = "College Football betting (resolved odds/lines).",
  espn_cfb_schedules         = "College Football schedules / game meta.",
  espn_cfb_linescores        = "College Football per-quarter linescores.",
  espn_cfb_power_index       = "College Football FPI / power index (recent seasons).",
  espn_cfb_injuries          = "College Football game injury reports.",
  espn_cfb_team_summaries    = "College Football season team summaries (opponent-adjusted EPA/success/explosiveness; 'Binion Box Score').",
  espn_cfb_passing           = "College Football season passing leaders (per-team, opponent-adjusted).",
  espn_cfb_rushing           = "College Football season rushing leaders (per-team, opponent-adjusted).",
  espn_cfb_receiving         = "College Football season receiving leaders (per-team, opponent-adjusted).",
  espn_cfb_percentiles       = "College Football season per-metric percentiles (team offense)."
)

token <- Sys.getenv("GITHUB_PAT")
for (repo in REPOS) {
  for (tag in names(TAGS)) {
    tryCatch(
      piggyback::pb_release_create(repo = repo, tag = tag, name = tag,
                                   body = TAGS[[tag]], .token = token),
      error = function(e) cli::cli_alert_info("{repo}@{tag}: {conditionMessage(e)}")
    )
  }
}
cli::cli_alert_success("release tag init complete")
