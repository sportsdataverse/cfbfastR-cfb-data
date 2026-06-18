"""Recover pre-2014 per-play participant names + athlete ids from text + boxscore.

ESPN publishes structured per-play participants only from 2014 on; for 2004-2013
games every ``{role}_player_name`` / ``{role}_player_id`` column is null (verified:
``play_participants`` is empty for *every* season 2004-2013 both in the stored
finals and from ESPN's live endpoint -- the data does not exist upstream). But the
play ``text`` names the participants in a highly regular way and the game
``boxscore`` lists, per stat category, each athlete's canonical ``displayName`` *and*
``id`` -- so the join is recoverable.

Each role is resolved by matching the play text against the boxscore (the text
identifies *which* athlete; the boxscore supplies the canonical spelling **and the
athlete id**, so a player's identity is stable across every play and game), with a
regex fallback for the handful the boxscore misses. A boxscore match fills both the
name and the id; a regex-only fallback fills the name with a null id (the athlete
isn't in the relevant boxscore category, so no id is available). Roles map to the
eight boxscore categories present in pre-2014 boxscores:

==============================  =================  ==================================
column (+ matching _player_id)  boxscore category  text cue
==============================  =================  ==================================
``rusher_player_name``          rushing            ``NAME rush ...``
``passer_player_name``          passing            ``NAME pass ...``
``receiver_player_name``        receiving          ``... (in)complete to NAME ...``
``interception_player_name``    interceptions      ``... intercepted by NAME ...``
``fg_kicker_player_name``       kicking            ``NAME NN yard field goal ...``
``kickoff_player_name``         kicking            ``NAME kickoff ...``
``punter_player_name``          punting            ``NAME punt ...``
``kickoff_return_player_name``  kickReturns        ``... kickoff ... returned by NAME``
``punt_return_player_name``     puntReturns        ``... punt ... returned by NAME``
==============================  =================  ==================================

``sack_player_name`` is also recovered (``... sacked by NAME ...``) by regex only --
ESPN ships no defensive boxscore category pre-2014, so there is neither a canonical
name source nor an athlete id (``sack_player_id`` is left null).

The fill is **null-only**: 2014+ games already carry these columns from the
structured participants pivot, so :func:`fill_participants_from_text` is a safe
no-op there and composes cleanly with the modern data.
"""
from __future__ import annotations

import re

# 1-4 capitalized name tokens; apostrophes / periods / hyphens allowed within a token.
_NAME = r"[A-Z][\w'.\-]+(?:\s+[A-Z][\w'.\-]+){0,3}"

# Boxscore stat categories that supply canonical names + ids pre-2014.
_CATEGORIES = (
    "rushing", "passing", "receiving", "interceptions",
    "kicking", "punting", "kickReturns", "puntReturns",
)

# prefix roles -- NAME leads the narrative; a cue word must be present to disambiguate.
# (column, boxscore_category, cue_substring, leading_regex)
# The cue is wrapped in a scoped case-insensitive group ``(?i:...)`` while the NAME
# capture stays case-SENSITIVE -- a global re.I would make ``[A-Z]`` match lowercase
# and let narrative tails ("for", "and", "at") be folded into the captured name.
_PREFIX_ROLES = [
    ("rusher_player_name",   "rushing", "rush",       re.compile(rf"^\s*({_NAME})(?i:\s+rush)")),
    ("passer_player_name",   "passing", "pass",       re.compile(rf"^\s*({_NAME})(?i:\s+pass\b)")),
    ("punter_player_name",   "punting", "punt",       re.compile(rf"^\s*({_NAME})(?i:\s+punt)")),
    ("fg_kicker_player_name", "kicking", "field goal", re.compile(rf"^\s*({_NAME})(?i:\s+\d+\s+yard field goal)")),
    ("kickoff_player_name",  "kicking", "kickoff",    re.compile(rf"^\s*({_NAME})(?i:\s+kickoff)")),
]

# after-connector roles -- NAME follows a fixed phrase.
# (column, boxscore_category | None, connector_phrase, fallback_regex)
_AFTER_ROLES = [
    ("receiver_player_name",     "receiving",     "complete to ",   re.compile(rf"(?i:(?:in)?complete to )({_NAME})")),
    ("interception_player_name", "interceptions", "intercepted by ", re.compile(rf"(?i:intercepted by )({_NAME})")),
    ("sack_player_name",         None,            "sacked by ",     re.compile(rf"(?i:sacked by )({_NAME})")),
]

# returners share the "returned by" connector; kickoff vs punt disambiguates the column.
_RETURN_RE = re.compile(rf"(?i:returned by )({_NAME})")
_RETURN_CATEGORY = {"kickoff_return_player_name": "kickReturns", "punt_return_player_name": "puntReturns"}


def _id_col(name_col: str) -> str:
    """``rusher_player_name`` -> ``rusher_player_id`` (parity with 2014+ schema)."""
    return name_col[: -len("_player_name")] + "_player_id"


def _boxscore_index(raw: dict) -> dict[str, dict[str, object]]:
    """Per-category ``{canonical_display_name: athlete_id}`` from the final.json boxscore."""
    out: dict[str, dict[str, object]] = {c: {} for c in _CATEGORIES}
    for team in (raw.get("boxscore") or {}).get("players") or []:
        for stat in team.get("statistics") or []:
            cat = stat.get("name")
            if cat not in out:
                continue
            for a in stat.get("athletes") or []:
                ath = a.get("athlete") or {}
                nm = ath.get("displayName")
                if nm:
                    out[cat][str(nm)] = ath.get("id")  # id may be None
    return out


def _names_longest_first(idx_cat: dict[str, object]) -> list[str]:
    """Category names longest-first so "Mike Williams" wins over a "Mike" prefix."""
    return sorted(idx_cat, key=len, reverse=True)


def _norm(name: str) -> str:
    """Collapse a name to alphanumerics+lowercase for fuzzy id lookup.

    The category boxscore stores ``displayName`` verbatim (exact-matchable), but the
    game roster and the names some plays carry differ by case / punctuation / spacing
    ("T.J. Yates" vs "TJ Yates"), so the roster fallback matches on this normal form.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _roster_index(raw: dict) -> dict[str, object]:
    """Game-wide ``{normalized_name: athlete_id}`` from the roster (+ boxscore union).

    Used as the id-pairing fallback when a name is absent from the role's own
    boxscore category -- chiefly **incomplete-pass targets** (a player targeted on an
    incompletion has no reception, so is not in the *receiving* boxscore) and the
    ``sack`` role (no defensive boxscore category exists pre-2014). Names that
    normalize ambiguously (two distinct athletes -> same key) are dropped so the
    fallback never guesses between players.
    """
    name_to_ids: dict[str, set] = {}

    def _add(name, aid):
        # game_rosters stores athlete_id as int; the boxscore stores id as str.
        # Cast to str so the same athlete from both sources isn't read as two ids
        # (which would flag the name ambiguous) -- and so paired ids stay str-typed.
        if name and aid is not None:
            name_to_ids.setdefault(_norm(str(name)), set()).add(str(aid))

    for r in raw.get("game_rosters") or []:
        aid = r.get("athlete_id") or r.get("id") or (r.get("athlete") or {}).get("id")
        _add(r.get("athlete_display_name") or r.get("full_name"), aid)
        _add(r.get("full_name"), aid)
    # boxscore union -- catches an athlete missing from game_rosters but with a stat line
    for team in (raw.get("boxscore") or {}).get("players") or []:
        for stat in team.get("statistics") or []:
            for a in stat.get("athletes") or []:
                ath = a.get("athlete") or {}
                _add(ath.get("displayName"), ath.get("id"))
    return {k: next(iter(ids)) for k, ids in name_to_ids.items() if len(ids) == 1}


def _trim(name: str) -> str:
    """Clean a regex-captured name: drop the sentence-terminating punctuation a
    greedy ``[\\w'.\\-]`` token absorbs ("Brewer." -> "Brewer") and any trailing
    short ALL-CAPS token ("... TD")."""
    name = name.strip().rstrip(".,;:")
    toks = name.split()
    while len(toks) > 1 and toks[-1].isupper() and len(toks[-1]) <= 3:
        toks.pop()
    return " ".join(toks).strip()


def _resolve_prefix(text: str, box_names: list[str], cue: str, regex: re.Pattern) -> str | None:
    """Name leading the play text (box-canonical first, regex fallback)."""
    low = text.lower()
    if cue not in low:
        return None
    head = text[:64]
    for nm in box_names:  # box-canonical: the athlete prefixes the narrative
        if head.startswith(nm + " ") or head.startswith(nm + ","):
            return nm
    m = regex.match(text)  # fallback: leading name token(s) before the cue word
    return _trim(m.group(1)) if m else None


def _resolve_after(text: str, box_names: list[str], connector: str, regex: re.Pattern) -> str | None:
    """Name following a fixed connector phrase (box-canonical first, regex fallback)."""
    low = text.lower()
    conn = connector.lower()
    for nm in box_names:  # box-canonical: connector + canonical name appears verbatim
        i = low.find(conn + nm.lower())
        if i != -1:
            end = i + len(conn) + len(nm)
            if end >= len(low) or not low[end].isalpha():  # word boundary after the name
                return nm
    m = regex.search(text)  # fallback: name token(s) right after the connector
    return _trim(m.group(1)) if m else None


def _resolve_returner(text: str, names: dict[str, list[str]]) -> tuple[str | None, str | None]:
    """Returner name + its column, disambiguating kickoff vs punt by the play wording."""
    low = text.lower()
    if "returned by" not in low:
        return None, None
    if "kickoff" in low:
        col, box_names = "kickoff_return_player_name", names["kickReturns"]
    elif "punt" in low:
        col, box_names = "punt_return_player_name", names["puntReturns"]
    else:
        return None, None
    for nm in box_names:
        i = low.find("returned by " + nm.lower())
        if i != -1:
            end = i + len("returned by ") + len(nm)
            if end >= len(low) or not low[end].isalpha():
                return col, nm
    m = _RETURN_RE.search(text)
    return (col, _trim(m.group(1))) if m else (None, None)


def fill_participants_from_text(plays: list[dict], raw: dict) -> dict[str, int]:
    """In-place backfill of null ``{role}_player_name`` + ``{role}_player_id`` columns.

    Two passes per role, both null-only:

    1. **Name** -- if the name column is null, resolve it from the play text
       (box-canonical first, regex fallback).
    2. **Id** -- if the name is now present (whether just resolved *or* already
       populated by an earlier text parse that recorded no id) and the id column
       is null, pair the athlete id by looking the name up in the boxscore.

    The second pass is why ids fill for names ``CFBPlayProcess`` already extracted
    pre-2014 without ids (passer / receiver / kicker / punter). Id resolution tries
    the role's own boxscore category first (most precise), then falls back to a
    game-wide normalized roster index -- which recovers **incomplete-pass targets**
    (absent from the receiving boxscore because they had no reception) and the
    ``sack`` role (no defensive boxscore category exists pre-2014). A name found in
    neither keeps a null id.

    Args:
        plays: The game's play dicts (mutated in place).
        raw: The game's ``final.json`` payload (for ``boxscore`` + ``game_rosters``).

    Returns:
        ``{"names": {col: n}, "ids": {col: n}}`` -- counts of names and ids filled.
        2014+ games already carry both, so this is a no-op there.
    """
    idx = _boxscore_index(raw)
    names = {c: _names_longest_first(idx[c]) for c in _CATEGORIES}
    roster = _roster_index(raw)
    name_counts: dict[str, int] = {}
    id_counts: dict[str, int] = {}

    def _fill_name(p: dict, col: str, nm: str) -> None:
        p[col] = nm
        name_counts[col] = name_counts.get(col, 0) + 1

    def _pair_id(p: dict, col: str, cat: str | None) -> None:
        nm = p.get(col)
        if not nm:
            return
        ic = _id_col(col)
        if p.get(ic):
            return
        aid = idx[cat].get(nm) if cat is not None else None  # category box: most precise
        if aid is None:
            aid = roster.get(_norm(str(nm)))  # game-wide roster fallback (normalized)
        if aid is not None:
            p[ic] = aid
            id_counts[ic] = id_counts.get(ic, 0) + 1

    for p in plays:
        text = p.get("text") or ""
        for col, cat, cue, regex in _PREFIX_ROLES:
            if text and not p.get(col):
                nm = _resolve_prefix(text, names[cat], cue, regex)
                if nm:
                    _fill_name(p, col, nm)
            _pair_id(p, col, cat)
        for col, cat, connector, regex in _AFTER_ROLES:
            if text and not p.get(col):
                nm = _resolve_after(text, names[cat] if cat else [], connector, regex)
                if nm:
                    _fill_name(p, col, nm)
            _pair_id(p, col, cat)
        if text:
            rcol, rnm = _resolve_returner(text, names)
            if rcol and rnm and not p.get(rcol):
                _fill_name(p, rcol, rnm)
        for rcol, rcat in _RETURN_CATEGORY.items():
            _pair_id(p, rcol, rcat)

    return {"names": name_counts, "ids": id_counts}
