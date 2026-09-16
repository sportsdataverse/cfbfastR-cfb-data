"""Coordinator names from Wikipedia team-season articles, and the 2003+ backfill.

Fixtures are real articles captured 2026-09-16 through the MediaWiki API, each
headed by the title and revision id it came from
(``fixtures/coordinators_wiki/*.wikitext``). They cover both infobox
templates the backfill reads (``Infobox NCAA team season`` and ``Infobox
college sports team season``) and a co-defensive-coordinator pair.

Accuracy of the scrape was measured on 2013-2016, where the imported
coordinator table gives an answer key: head coach 0.994, offensive
coordinators 0.940 and defensive 0.926 any-overlap (0.870 / 0.852 exact), and
continuity derived from Wikipedia agrees with continuity derived from the
imported table on 0.926 / 0.924 of school-seasons. Several disagreements were
typos in the imported table ("Skorsky" for Skrosky, "Regan" for Reagan),
which is why continuity never compares names across the two sources.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cfb_data_build import coordinators_wiki as cw
from cfb_data_build.matchup_reference import (
    build_coach_continuity,
    canonicalize_schools,
    load_coach_continuity,
    load_wikipedia_coordinators,
)

FIX = Path(__file__).parent / "fixtures" / "coordinators_wiki"


def _article(name: str) -> str:
    return (FIX / f"{name}.wikitext").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("name", "season", "expected"),
    [
        (  # "Infobox NCAA team season", the older template
            "2002_Toledo_Rockets_football_team",
            2002,
            ("Tom Amstutz", 2001, "Rob Spence", "Lou West"),
        ),
        (  # "Infobox college sports team season", same parameters
            "2003_Georgia_Bulldogs_football_team",
            2003,
            ("Mark Richt", 2001, "Neil Callaway", "Brian VanGorder"),
        ),
        (  # co-defensive coordinators live in codef_coach1 / codef_coach2
            "2006_Texas_Longhorns_football_team",
            2006,
            ("Mack Brown", 1998, "Greg Davis", "Gene Chizik / Duane Akina"),
        ),
    ],
)
def test_real_articles_parse(name: str, season: int, expected: tuple) -> None:
    got = cw.coaches_from_page(_article(name), season)
    assert got is not None
    assert (
        got["head_coach"],
        got["first_season"],
        got["offensive_coordinators"],
        got["defensive_coordinators"],
    ) == expected


@pytest.mark.parametrize(
    ("raw", "names"),
    [
        ("[[Greg Davis (American football coach)|Greg Davis]]", ["Greg Davis"]),
        # an annotation's slash must not split the name it describes
        ("[[John Klacik]] (2nd season / first as OC)", ["John Klacik"]),
        (
            "[[Steve Spurrier Jr.]] / [[Shawn Elliott]]",
            ["Steve Spurrier Jr.", "Shawn Elliott"],
        ),
        ("Bob Smith, Jr.", ["Bob Smith, Jr."]),
        ("[[Rich Skrosky]]<br>[[Pat Jones]] (interim)", ["Rich Skrosky", "Pat Jones"]),
        ("{{sortname|Kirby|Smart}}", ["Kirby Smart"]),
        ("[[Duane Akina]]<ref>{{cite web|title=x}}</ref>", ["Duane Akina"]),
        ("Vacant", []),
        ("", []),
    ],
)
def test_clean_value(raw: str, names: list[str]) -> None:
    assert cw.clean_value(raw) == names


def test_a_pipe_inside_a_link_does_not_end_a_parameter() -> None:
    params = cw.infobox_params(
        "{{Infobox NCAA team season\n|head_coach=[[A B|A B]]\n|off_coach=[[C|D]]\n}}"
    )
    assert params["head_coach"] == "[[A B|A B]]" and params["off_coach"] == "[[C|D]]"


def test_an_article_without_the_infobox_yields_nothing() -> None:
    assert cw.coaches_from_page("The 2003 team played football.", 2003) is None


def test_titles_try_the_era_spellings() -> None:
    titles = cw.title_candidates(2003, "UL Monroe", "Warhawks")
    assert titles[0] == "2003 UL Monroe Warhawks football team"
    assert "2003 Louisiana–Monroe Indians football team" in titles


def test_pinned_rebuild_reads_the_recorded_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebuild asks for revision ids, not titles, and reproduces the rows."""
    text = _article("2006_Texas_Longhorns_football_team")
    seen: list[list[int]] = []

    def fake_fetch(revids: list[int]) -> dict[int, cw.Page]:
        seen.append(list(revids))
        return {
            1337681621: cw.Page("2006 Texas Longhorns football team", 1337681621, text)
        }

    monkeypatch.setattr(cw, "fetch_revisions", fake_fetch)
    pinned = pl.DataFrame(
        [
            {
                "season": 2006,
                "school_mascot": "Texas Longhorns",
                "head_coach": "Mack Brown",
                "first_season": 1998,
                "offensive_coordinators": "Greg Davis",
                "defensive_coordinators": "Gene Chizik / Duane Akina",
                "source_title": "2006 Texas Longhorns football team",
                "source_revid": 1337681621,
            }
        ],
        schema=cw.COLUMNS,
    )
    rebuilt = cw.rebuild_pinned(pinned)
    assert seen == [[1337681621]]
    assert rebuilt.equals(pinned)


def test_a_pinned_revision_that_no_longer_parses_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cw, "fetch_revisions", lambda ids: {7: cw.Page("t", 7, "no infobox here")}
    )
    pinned = pl.DataFrame(
        [
            {
                "season": 2006,
                "school_mascot": "X",
                "source_title": "t",
                "source_revid": 7,
            }
        ],
        schema_overrides={"source_revid": pl.Int64},
    )
    with pytest.raises(RuntimeError, match="no longer parses"):
        cw.rebuild_pinned(pinned)


# --- the backfill as committed ---------------------------------------------------


def test_wikipedia_table_reaches_2002_and_pins_every_row() -> None:
    t = load_wikipedia_coordinators()
    assert t["season"].min() == 2002
    assert t.unique(subset=["season", "school_mascot"]).height == t.height
    assert t["source_revid"].null_count() == 0
    assert t["head_coach"].null_count() == 0
    # measured 2026-09-16: offensive 0.87-0.97, defensive 0.90-0.99 per season
    fill = t.group_by("season").agg(
        pl.col("offensive_coordinators").is_not_null().mean().alias("oc"),
        pl.col("defensive_coordinators").is_not_null().mean().alias("dc"),
    )
    assert fill["oc"].min() >= 0.85 and fill["dc"].min() >= 0.88


def test_continuity_is_backfilled_to_2003_with_one_source_per_value() -> None:
    cont = load_coach_continuity()
    assert cont["season"].min() == 2003
    assert cont.unique(subset=["season", "school_mascot"]).height == cont.height
    assert set(cont["source"].unique()) == {"imported", "wikipedia"}
    # every season from 2003 on is populated
    assert cont["season"].n_unique() == cont["season"].max() - 2003 + 1


def test_coach_seasons_reach_2003() -> None:
    from cfb_data_build.coaches import load_coach_seasons

    seasons = load_coach_seasons()
    assert seasons["season"].min() == 2003
    assert seasons.filter(pl.col("season") == 2003).height >= 115


# --- continuity across the two sources -------------------------------------------


def test_continuity_never_compares_names_across_sources() -> None:
    """A typo in one source must not read as a coaching change at the seam.

    Wikipedia carries 2012-2013 for school A with the same coordinator; the
    imported table starts in 2013 and misspells him. Comparing imported 2013
    with Wikipedia 2012 would flag a change; the same-source rule does not.
    """
    wiki = pl.DataFrame(
        {
            "season": [2012, 2013],
            "school_mascot": ["A Aces", "A Aces"],
            "offensive_coordinators": ["Rich Skrosky", "Rich Skrosky"],
            "defensive_coordinators": ["X", "X"],
        }
    )
    imported = pl.DataFrame(
        {
            "season": [2013, 2014],
            "school_mascot": ["A Aces", "A Aces"],
            "offensive_coordinators": ["Rich Skorsky", "Rich Skorsky"],
            "defensive_coordinators": ["X", "X"],
        }
    )
    out = build_coach_continuity(imported, wiki).sort("season")
    rows = {r["season"]: (r["oc_cont"], r["source"]) for r in out.to_dicts()}
    assert rows[2013] == (1, "wikipedia")  # 2012 -> 2013 within Wikipedia
    assert rows[2014] == (1, "imported")  # 2013 -> 2014 within the imported table


def test_imported_continuity_wins_where_both_sources_have_it() -> None:
    wiki = pl.DataFrame(
        {
            "season": [2014, 2015],
            "school_mascot": ["A Aces"] * 2,
            "offensive_coordinators": ["P", "Q"],  # Wikipedia says a change
            "defensive_coordinators": ["X", "X"],
        }
    )
    imported = pl.DataFrame(
        {
            "season": [2014, 2015],
            "school_mascot": ["A Aces"] * 2,
            "offensive_coordinators": ["P", "P"],  # imported says retained
            "defensive_coordinators": ["X", "X"],
        }
    )
    out = build_coach_continuity(imported, wiki)
    assert out.height == 1
    assert out.row(0, named=True)["oc_cont"] == 1
    assert out.row(0, named=True)["source"] == "imported"


def test_canonicalizing_across_seasons_tolerates_two_spellings() -> None:
    """One table spells a school two ways in different seasons; that is not a conflict."""
    frame = pl.DataFrame(
        {
            "season": [2020, 2021, 2021],
            "school_mascot": [
                "San José State Spartans",
                "San Jose State Spartans",
                "Brand New Program Owls",
            ],
        }
    )
    out = canonicalize_schools(frame, ["San José State Spartans"])
    assert out["school_mascot"].to_list() == [
        "San José State Spartans",
        "San José State Spartans",
        "Brand New Program Owls",  # unresolved keeps its own spelling
    ]
