# CFB model suite — reproducible deep dive


The per-model cards in this directory are **generated** by
`cfb_model_reports` (stage `cfb_model_70_reports`) from the training
pipeline’s real artifacts — they are already compiled documents, rebuilt
on every reports run, with LOSO metrics and calibration figures produced
by the same code that trains the models. This page is their **Quarto
companion**: a suite-level deep dive the generator does not own,
recomputing the leave-one-season-out evaluations from the committed
out-of-fold artifacts, attributing the boosters with TreeSHAP, and
putting identified, human-readable results tables on the published
surfaces. Everything here is computed at render time — from
`python/artifacts/` (the model pipeline’s output tree; regenerate with
`scripts/cfb_models.sh`) and the published releases.

## The corpus

<div id="nmwujfvoit" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#nmwujfvoit table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#nmwujfvoit thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#nmwujfvoit p { margin: 0; padding: 0; }
 #nmwujfvoit .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #nmwujfvoit .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #nmwujfvoit .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #nmwujfvoit .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #nmwujfvoit .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #nmwujfvoit .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #nmwujfvoit .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #nmwujfvoit .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #nmwujfvoit .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #nmwujfvoit .gt_column_spanner_outer:first-child { padding-left: 0; }
 #nmwujfvoit .gt_column_spanner_outer:last-child { padding-right: 0; }
 #nmwujfvoit .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #nmwujfvoit .gt_spanner_row { border-bottom-style: hidden; }
 #nmwujfvoit .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #nmwujfvoit .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #nmwujfvoit .gt_from_md> :first-child { margin-top: 0; }
 #nmwujfvoit .gt_from_md> :last-child { margin-bottom: 0; }
 #nmwujfvoit .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #nmwujfvoit .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #nmwujfvoit .gt_indent_1 { text-indent: 5px; }
 #nmwujfvoit .gt_indent_2 { text-indent: calc(5px * 2); }
 #nmwujfvoit .gt_indent_3 { text-indent: calc(5px * 3); }
 #nmwujfvoit .gt_indent_4 { text-indent: calc(5px * 4); }
 #nmwujfvoit .gt_indent_5 { text-indent: calc(5px * 5); }
 #nmwujfvoit .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #nmwujfvoit .gt_row_group_first td { border-top-width: 2px; }
 #nmwujfvoit .gt_row_group_first th { border-top-width: 2px; }
 #nmwujfvoit .gt_striped { color: #333333; background-color: #F4F4F4; }
 #nmwujfvoit .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #nmwujfvoit .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #nmwujfvoit .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #nmwujfvoit .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #nmwujfvoit .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #nmwujfvoit .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #nmwujfvoit .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #nmwujfvoit .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #nmwujfvoit .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #nmwujfvoit .gt_left { text-align: left; }
 #nmwujfvoit .gt_center { text-align: center; }
 #nmwujfvoit .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #nmwujfvoit .gt_font_normal { font-weight: normal; }
 #nmwujfvoit .gt_font_bold { font-weight: bold; }
 #nmwujfvoit .gt_font_italic { font-style: italic; }
 #nmwujfvoit .gt_super { font-size: 65%; }
 #nmwujfvoit .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #nmwujfvoit .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #nmwujfvoit .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #nmwujfvoit .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #nmwujfvoit .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #nmwujfvoit .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Training corpus (python/artifacts/pbp_full.parquet) — last 10 of 22 seasons |  |  |  |
|----|----|----|----|
| the exact frame the trainers read; regenerated by the model pipeline |  |  |  |
| season | plays | games | mean_epa |
| 2016 | 94,384 | 577 | −0.1441 |
| 2017 | 94,492 | 586 | 0.0187 |
| 2018 | 92,448 | 571 | 0.0388 |
| 2019 | 88,349 | 553 | 0.0493 |
| 2020 | 66,340 | 415 | 0.0557 |
| 2021 | 116,810 | 748 | 0.0473 |
| 2022 | 126,137 | 810 | 0.0164 |
| 2023 | 131,644 | 869 | 0.0295 |
| 2024 | 134,707 | 889 | 0.0445 |
| 2025 | 132,521 | 871 | 0.0169 |

&#10;</div>

## Exploratory data analysis

<img src="deepdive_files/figure-commonmark/cell-4-output-1.png"
width="420" height="300"
alt="EPA per play by down — the down structure every play-level model must respect." />

<img src="deepdive_files/figure-commonmark/cell-5-output-1.png"
width="420" height="300"
alt="EP by field position (1st &amp; 10), computed from the corpus — the surface the EP model fits." />

## LOSO evaluation, recomputed from the out-of-fold artifacts

The pipeline persists every model’s leave-one-season-out out-of-fold
predictions under `python/artifacts/loso_*_oof.parquet`. Recomputing the
headline metrics from those files — rather than quoting the cards —
keeps this page honest against the artifacts themselves:

<div id="lanbyepghw" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#lanbyepghw table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#lanbyepghw thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#lanbyepghw p { margin: 0; padding: 0; }
 #lanbyepghw .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #lanbyepghw .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #lanbyepghw .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #lanbyepghw .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #lanbyepghw .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #lanbyepghw .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #lanbyepghw .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #lanbyepghw .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #lanbyepghw .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #lanbyepghw .gt_column_spanner_outer:first-child { padding-left: 0; }
 #lanbyepghw .gt_column_spanner_outer:last-child { padding-right: 0; }
 #lanbyepghw .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #lanbyepghw .gt_spanner_row { border-bottom-style: hidden; }
 #lanbyepghw .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #lanbyepghw .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #lanbyepghw .gt_from_md> :first-child { margin-top: 0; }
 #lanbyepghw .gt_from_md> :last-child { margin-bottom: 0; }
 #lanbyepghw .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #lanbyepghw .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #lanbyepghw .gt_indent_1 { text-indent: 5px; }
 #lanbyepghw .gt_indent_2 { text-indent: calc(5px * 2); }
 #lanbyepghw .gt_indent_3 { text-indent: calc(5px * 3); }
 #lanbyepghw .gt_indent_4 { text-indent: calc(5px * 4); }
 #lanbyepghw .gt_indent_5 { text-indent: calc(5px * 5); }
 #lanbyepghw .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #lanbyepghw .gt_row_group_first td { border-top-width: 2px; }
 #lanbyepghw .gt_row_group_first th { border-top-width: 2px; }
 #lanbyepghw .gt_striped { color: #333333; background-color: #F4F4F4; }
 #lanbyepghw .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #lanbyepghw .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #lanbyepghw .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #lanbyepghw .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #lanbyepghw .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #lanbyepghw .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #lanbyepghw .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #lanbyepghw .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #lanbyepghw .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #lanbyepghw .gt_left { text-align: left; }
 #lanbyepghw .gt_center { text-align: center; }
 #lanbyepghw .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #lanbyepghw .gt_font_normal { font-weight: normal; }
 #lanbyepghw .gt_font_bold { font-weight: bold; }
 #lanbyepghw .gt_font_italic { font-style: italic; }
 #lanbyepghw .gt_super { font-size: 65%; }
 #lanbyepghw .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #lanbyepghw .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #lanbyepghw .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #lanbyepghw .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #lanbyepghw .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #lanbyepghw .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| LOSO out-of-fold metrics — recomputed at render time |  |  |  |
|----|----|----|----|
| from python/artifacts/loso\_\*\_oof.parquet; every prediction is from a model that never saw its season |  |  |  |
| model | n_oof | metric | value |
| EP | 2,219,607 | MAE | 4.1030 |
| WP (spread) | 2,219,660 | Brier | 0.1135 |
| WP (naive) | 2,219,607 | Brier | 0.1332 |
| FG | 42,615 | Brier | 0.1749 |
| QBR | 22,833 | MAE | 13.3996 |
| Two-point | 1,622 | Brier | 0.2493 |

&#10;</div>

<img src="deepdive_files/figure-commonmark/cell-7-output-1.png"
width="420" height="300"
alt="WP LOSO reliability — pooled out-of-fold predicted WP vs realized win rate." />

<img src="deepdive_files/figure-commonmark/cell-8-output-1.png"
width="420" height="300"
alt="Per-season LOSO EP calibration MAE — drift check across the era span." />

## SHAP — the promoted boosters attributed

The five score-time boosters below are the **promoted** artifacts — the
`python/artifacts/v2/` fits, byte-identical to the `.ubj` files sdv-py
bundles in `cfb/models/` (the CP booster is read from that bundle).
Their feature matrices come from the trainer’s own analysis export:
`python -m cfb_model_build.model_training export-analysis` writes
`python/artifacts/analysis/analysis_{ep,wp,xpass,cp}.parquet` through
the very `ep_matrix` / `wp_matrix` / `xpass_frame` /
`extract_pass_features` code paths the fits use, so `spread_time`,
`adj_TimeSecsRem`, `ExpScoreDiff_Time_Ratio`, `score_diff` and `era` are
the engineered columns the boosters actually saw, not a re-derivation on
this page. TreeSHAP is XGBoost’s own `pred_contribs=True` (no `shap`
dependency); the multiclass EP tensor is 3-D and is averaged over its
seven class margins, the binary models are 2-D.

<div id="rfdzvwhqjp" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#rfdzvwhqjp table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#rfdzvwhqjp thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#rfdzvwhqjp p { margin: 0; padding: 0; }
 #rfdzvwhqjp .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #rfdzvwhqjp .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #rfdzvwhqjp .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #rfdzvwhqjp .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #rfdzvwhqjp .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #rfdzvwhqjp .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #rfdzvwhqjp .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #rfdzvwhqjp .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #rfdzvwhqjp .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #rfdzvwhqjp .gt_column_spanner_outer:first-child { padding-left: 0; }
 #rfdzvwhqjp .gt_column_spanner_outer:last-child { padding-right: 0; }
 #rfdzvwhqjp .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #rfdzvwhqjp .gt_spanner_row { border-bottom-style: hidden; }
 #rfdzvwhqjp .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #rfdzvwhqjp .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #rfdzvwhqjp .gt_from_md> :first-child { margin-top: 0; }
 #rfdzvwhqjp .gt_from_md> :last-child { margin-bottom: 0; }
 #rfdzvwhqjp .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #rfdzvwhqjp .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #rfdzvwhqjp .gt_indent_1 { text-indent: 5px; }
 #rfdzvwhqjp .gt_indent_2 { text-indent: calc(5px * 2); }
 #rfdzvwhqjp .gt_indent_3 { text-indent: calc(5px * 3); }
 #rfdzvwhqjp .gt_indent_4 { text-indent: calc(5px * 4); }
 #rfdzvwhqjp .gt_indent_5 { text-indent: calc(5px * 5); }
 #rfdzvwhqjp .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #rfdzvwhqjp .gt_row_group_first td { border-top-width: 2px; }
 #rfdzvwhqjp .gt_row_group_first th { border-top-width: 2px; }
 #rfdzvwhqjp .gt_striped { color: #333333; background-color: #F4F4F4; }
 #rfdzvwhqjp .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #rfdzvwhqjp .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #rfdzvwhqjp .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #rfdzvwhqjp .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #rfdzvwhqjp .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #rfdzvwhqjp .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #rfdzvwhqjp .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #rfdzvwhqjp .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #rfdzvwhqjp .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #rfdzvwhqjp .gt_left { text-align: left; }
 #rfdzvwhqjp .gt_center { text-align: center; }
 #rfdzvwhqjp .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #rfdzvwhqjp .gt_font_normal { font-weight: normal; }
 #rfdzvwhqjp .gt_font_bold { font-weight: bold; }
 #rfdzvwhqjp .gt_font_italic { font-style: italic; }
 #rfdzvwhqjp .gt_super { font-size: 65%; }
 #rfdzvwhqjp .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #rfdzvwhqjp .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #rfdzvwhqjp .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #rfdzvwhqjp .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #rfdzvwhqjp .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #rfdzvwhqjp .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Which booster each attribution below explains |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
| analysis frames from python/artifacts/pbp_full_v2.parquet (export-analysis, 2026-09-02); sha256 prefix compared against the .ubj sdv-py ships |  |  |  |  |  |  |
| Model | Artifact | sha256\[:12\] | vs bundle | Features | Frame rows | SHAP sample |
| EP (7-class) | python/artifacts/v2/ep_model.ubj | 44516f561836 | = sdv-py bundle | 8 | 2,219,971 | 4,000 |
| WP (spread) | python/artifacts/v2/wp_spread.ubj | 486699334ef9 | = sdv-py bundle | 13 | 2,219,971 | 4,000 |
| WP (naive) | python/artifacts/v2/wp_naive.ubj | 624ee1494330 | = sdv-py bundle | 12 | 2,219,971 | 4,000 |
| xPass | python/artifacts/v2/xpass_model.ubj | 90a27a21e49d | = sdv-py bundle | 7 | 1,902,481 | 4,000 |
| CP | sdv-py bundle/cfb_cp_model.ubj | 71d52772ee59 | = sdv-py bundle | 8 | 921,377 | 4,000 |

&#10;</div>

<img src="deepdive_files/figure-commonmark/cell-11-output-1.png"
width="420" height="300"
alt="EP TreeSHAP (class-margin space), mean |contribution| across the 7 next-score classes; 4,000-play sample." />

<img src="deepdive_files/figure-commonmark/cell-12-output-1.png"
width="420" height="300"
alt="WP (spread) TreeSHAP, log-odds space; 4,000-play sample. spread_time is the pregame spread decayed by elapsed share." />

<img src="deepdive_files/figure-commonmark/cell-13-output-1.png"
width="420" height="300"
alt="WP (naive) TreeSHAP, log-odds space; the spread model’s 12 shared features without spread_time." />

<img src="deepdive_files/figure-commonmark/cell-14-output-1.png"
width="420" height="300"
alt="xPass TreeSHAP, log-odds space; rush|pass plays, era is the ordinal rule-era factor (2006/2013/2020 cuts)." />

<img src="deepdive_files/figure-commonmark/cell-15-output-1.png"
width="420" height="300"
alt="CP TreeSHAP, log-odds space; pass plays only (the 8 game-state features of Approach A)." />

<div id="aujmqqorjn" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#aujmqqorjn table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#aujmqqorjn thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#aujmqqorjn p { margin: 0; padding: 0; }
 #aujmqqorjn .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #aujmqqorjn .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #aujmqqorjn .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #aujmqqorjn .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #aujmqqorjn .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #aujmqqorjn .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #aujmqqorjn .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #aujmqqorjn .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #aujmqqorjn .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #aujmqqorjn .gt_column_spanner_outer:first-child { padding-left: 0; }
 #aujmqqorjn .gt_column_spanner_outer:last-child { padding-right: 0; }
 #aujmqqorjn .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #aujmqqorjn .gt_spanner_row { border-bottom-style: hidden; }
 #aujmqqorjn .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #aujmqqorjn .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #aujmqqorjn .gt_from_md> :first-child { margin-top: 0; }
 #aujmqqorjn .gt_from_md> :last-child { margin-bottom: 0; }
 #aujmqqorjn .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #aujmqqorjn .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #aujmqqorjn .gt_indent_1 { text-indent: 5px; }
 #aujmqqorjn .gt_indent_2 { text-indent: calc(5px * 2); }
 #aujmqqorjn .gt_indent_3 { text-indent: calc(5px * 3); }
 #aujmqqorjn .gt_indent_4 { text-indent: calc(5px * 4); }
 #aujmqqorjn .gt_indent_5 { text-indent: calc(5px * 5); }
 #aujmqqorjn .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #aujmqqorjn .gt_row_group_first td { border-top-width: 2px; }
 #aujmqqorjn .gt_row_group_first th { border-top-width: 2px; }
 #aujmqqorjn .gt_striped { color: #333333; background-color: #F4F4F4; }
 #aujmqqorjn .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #aujmqqorjn .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #aujmqqorjn .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #aujmqqorjn .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #aujmqqorjn .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #aujmqqorjn .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #aujmqqorjn .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #aujmqqorjn .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #aujmqqorjn .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #aujmqqorjn .gt_left { text-align: left; }
 #aujmqqorjn .gt_center { text-align: center; }
 #aujmqqorjn .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #aujmqqorjn .gt_font_normal { font-weight: normal; }
 #aujmqqorjn .gt_font_bold { font-weight: bold; }
 #aujmqqorjn .gt_font_italic { font-style: italic; }
 #aujmqqorjn .gt_super { font-size: 65%; }
 #aujmqqorjn .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #aujmqqorjn .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #aujmqqorjn .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #aujmqqorjn .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #aujmqqorjn .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #aujmqqorjn .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Top-3 features per booster — mean \|SHAP\| and share of total attribution |  |  |  |  |
|----|----|----|----|----|
| margin space (log-odds; EP: class-margin mean); shares sum to 1 over each model's full feature set |  |  |  |  |
| Model | \# | Feature | mean \|SHAP\| | Share |
| EP (7-class) | 1 | yards_to_goal | 0.388 | 34.4% |
| EP (7-class) | 2 | TimeSecsRem | 0.380 | 33.7% |
| EP (7-class) | 3 | pos_score_diff_start | 0.190 | 16.8% |
| WP (spread) | 1 | ExpScoreDiff_Time_Ratio | 1.134 | 32.8% |
| WP (spread) | 2 | pos_score_diff_start | 1.108 | 32.1% |
| WP (spread) | 3 | spread_time | 0.808 | 23.4% |
| WP (naive) | 1 | pos_score_diff_start | 1.451 | 48.0% |
| WP (naive) | 2 | ExpScoreDiff_Time_Ratio | 0.943 | 31.2% |
| WP (naive) | 3 | pos_team_receives_2H_kickoff | 0.167 | 5.5% |
| xPass | 1 | down | 0.425 | 31.3% |
| xPass | 2 | distance | 0.322 | 23.7% |
| xPass | 3 | pos_score_diff | 0.230 | 16.9% |
| CP | 1 | down | 0.136 | 27.1% |
| CP | 2 | score_diff | 0.098 | 19.6% |
| CP | 3 | yards_to_goal | 0.077 | 15.4% |

&#10;</div>

<img src="deepdive_files/figure-commonmark/cell-17-output-1.png"
width="420" height="300"
alt="WP (spread) mean |SHAP| by season, 1,500 plays per season. A feature whose weight jumps at a rule-era cut (2006/2013/2020) would be doing the era’s work; a flat line means the model reads it the same way in every era." />

What the attributions say — 4,000-play samples (seed 11); the exact
values are in the top-3 table above:

- **EP.** Field position and clock split the top: `yards_to_goal` 34.4%
  and `TimeSecsRem` 33.7% of total attribution, score state
  (`pos_score_diff_start`) third at 16.8%, the four down dummies
  together about 14%. That is the cfbscrapR ordering attributed
  play-by-play — on the promoted booster the first two are effectively
  tied rather than field position clearly first.
- **WP (spread).** Two score-state terms carry it —
  `ExpScoreDiff_Time_Ratio` 32.8% and `pos_score_diff_start` 32.1% —
  with `spread_time` the third pillar at 23.4%; no clock or field term
  exceeds 3%. **WP (naive)** drops `spread_time` and that share does not
  disappear: `pos_score_diff_start` absorbs it (32.1% → 48.0%),
  `ExpScoreDiff_Time_Ratio` holds at 31.2%, and
  `pos_team_receives_2H_kickoff` becomes the third feature (5.5%).
- **xPass** reads the situation first — `down` 31.3%, `distance` 23.7% —
  then score (`pos_score_diff` 16.9%), field position (10.6%) and clock
  (7.9%); `era` and `period` are the tail, so rule-era pass tendency is
  a small effect once down-and-distance is in.
- **CP** is the game-state-only Approach A and attributes like one:
  `down` 27.1%, `score_diff` 19.6%, `yards_to_goal` 15.4%, `distance`
  9.9%, `period` 8.3%. What an air-yards feature would add is the
  recorded CPOE feasibility finding; SHAP cannot supply it here.
- **By season (WP spread, 1,500 plays × 22 seasons).** The score-state
  terms move within a narrow band — `ExpScoreDiff_Time_Ratio` 1.04–1.15,
  `pos_score_diff_start` 0.97–1.17 (extremes at 2016 and 2013),
  `adj_TimeSecsRem` 0.07–0.12 — with no step at the 2006/2013/2020
  rule-era cuts. `spread_time` is the exception: 0.31 in 2004, about 1.0
  from 2013 on (0.98 in 2025). That ramp is neither a rule effect nor
  missing data (`spread_time` is non-null in every season): it is the
  ESPN **default spread**. In the training frame every 2004 and 2005
  game carries `homeTeamSpread = ±2.5` with `gameSpreadAvailable = 0`,
  as do 57% of 2006 games, 67% of 2008, 8.7% of 2013 and 5.1% of 2025 —
  the `cfb_line_odds` consensus backfill closed 2,167 games but not the
  earliest seasons. A 2.5-point placeholder decays to a near-constant,
  so the booster has nothing to attribute to it, and the early-season
  WP-spread model runs close to the naive one. The fix is upstream of
  SHAP — a historical-spread source for 2004–2008 — and belongs with the
  backfill work, not this page.

## Results — the scored surfaces, identified

<div id="kqkamnupdt" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#kqkamnupdt table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#kqkamnupdt thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#kqkamnupdt p { margin: 0; padding: 0; }
 #kqkamnupdt .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #kqkamnupdt .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #kqkamnupdt .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #kqkamnupdt .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #kqkamnupdt .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #kqkamnupdt .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #kqkamnupdt .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #kqkamnupdt .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #kqkamnupdt .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #kqkamnupdt .gt_column_spanner_outer:first-child { padding-left: 0; }
 #kqkamnupdt .gt_column_spanner_outer:last-child { padding-right: 0; }
 #kqkamnupdt .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #kqkamnupdt .gt_spanner_row { border-bottom-style: hidden; }
 #kqkamnupdt .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #kqkamnupdt .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #kqkamnupdt .gt_from_md> :first-child { margin-top: 0; }
 #kqkamnupdt .gt_from_md> :last-child { margin-bottom: 0; }
 #kqkamnupdt .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #kqkamnupdt .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #kqkamnupdt .gt_indent_1 { text-indent: 5px; }
 #kqkamnupdt .gt_indent_2 { text-indent: calc(5px * 2); }
 #kqkamnupdt .gt_indent_3 { text-indent: calc(5px * 3); }
 #kqkamnupdt .gt_indent_4 { text-indent: calc(5px * 4); }
 #kqkamnupdt .gt_indent_5 { text-indent: calc(5px * 5); }
 #kqkamnupdt .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #kqkamnupdt .gt_row_group_first td { border-top-width: 2px; }
 #kqkamnupdt .gt_row_group_first th { border-top-width: 2px; }
 #kqkamnupdt .gt_striped { color: #333333; background-color: #F4F4F4; }
 #kqkamnupdt .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #kqkamnupdt .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #kqkamnupdt .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #kqkamnupdt .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #kqkamnupdt .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #kqkamnupdt .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #kqkamnupdt .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #kqkamnupdt .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #kqkamnupdt .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #kqkamnupdt .gt_left { text-align: left; }
 #kqkamnupdt .gt_center { text-align: center; }
 #kqkamnupdt .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #kqkamnupdt .gt_font_normal { font-weight: normal; }
 #kqkamnupdt .gt_font_bold { font-weight: bold; }
 #kqkamnupdt .gt_font_italic { font-style: italic; }
 #kqkamnupdt .gt_super { font-size: 65%; }
 #kqkamnupdt .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #kqkamnupdt .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #kqkamnupdt .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #kqkamnupdt .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #kqkamnupdt .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #kqkamnupdt .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Top 10 offenses by EPA/play — 2025 model_pbp |  |  |  |
|----|----|----|----|
| source: python/artifacts/model_pbp_2025.parquet (stage 10 build tree) |  |  |  |
|  | Offense | Plays | EPA/play |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/2011.png"
height="36" /> | Alabama State | 64 | 0.521 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/238.png"
height="36" /> | Vanderbilt | 796 | 0.341 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/2241.png"
height="36" /> | Gardner-Webb | 61 | 0.301 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/2449.png"
height="36" /> | North Dakota State | 795 | 0.300 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/249.png"
height="36" /> | North Texas | 984 | 0.293 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/149.png"
height="36" /> | Montana | 59 | 0.290 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/2426.png"
height="36" /> | Navy | 813 | 0.289 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/254.png"
height="36" /> | Utah | 945 | 0.278 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/87.png"
height="36" /> | Notre Dame | 748 | 0.275 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/84.png"
height="36" /> | Indiana | 1,059 | 0.262 |

&#10;</div>

<div id="siotgymeei" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#siotgymeei table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#siotgymeei thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#siotgymeei p { margin: 0; padding: 0; }
 #siotgymeei .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #siotgymeei .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #siotgymeei .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #siotgymeei .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #siotgymeei .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #siotgymeei .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #siotgymeei .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #siotgymeei .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #siotgymeei .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #siotgymeei .gt_column_spanner_outer:first-child { padding-left: 0; }
 #siotgymeei .gt_column_spanner_outer:last-child { padding-right: 0; }
 #siotgymeei .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #siotgymeei .gt_spanner_row { border-bottom-style: hidden; }
 #siotgymeei .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #siotgymeei .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #siotgymeei .gt_from_md> :first-child { margin-top: 0; }
 #siotgymeei .gt_from_md> :last-child { margin-bottom: 0; }
 #siotgymeei .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #siotgymeei .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #siotgymeei .gt_indent_1 { text-indent: 5px; }
 #siotgymeei .gt_indent_2 { text-indent: calc(5px * 2); }
 #siotgymeei .gt_indent_3 { text-indent: calc(5px * 3); }
 #siotgymeei .gt_indent_4 { text-indent: calc(5px * 4); }
 #siotgymeei .gt_indent_5 { text-indent: calc(5px * 5); }
 #siotgymeei .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #siotgymeei .gt_row_group_first td { border-top-width: 2px; }
 #siotgymeei .gt_row_group_first th { border-top-width: 2px; }
 #siotgymeei .gt_striped { color: #333333; background-color: #F4F4F4; }
 #siotgymeei .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #siotgymeei .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #siotgymeei .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #siotgymeei .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #siotgymeei .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #siotgymeei .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #siotgymeei .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #siotgymeei .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #siotgymeei .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #siotgymeei .gt_left { text-align: left; }
 #siotgymeei .gt_center { text-align: center; }
 #siotgymeei .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #siotgymeei .gt_font_normal { font-weight: normal; }
 #siotgymeei .gt_font_bold { font-weight: bold; }
 #siotgymeei .gt_font_italic { font-style: italic; }
 #siotgymeei .gt_super { font-size: 65%; }
 #siotgymeei .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #siotgymeei .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #siotgymeei .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #siotgymeei .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #siotgymeei .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #siotgymeei .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Passer EPA/play leaders with CPOE — 2025 (min 150 dropbacks) |  |  |  |  |  |
|----|----|----|----|----|----|
| source: python/artifacts/model_pbp_2025.parquet (stage 10 build tree); keyed by ESPN athlete id (passer_player_id), headshots from a.espncdn.com |  |  |  |  |  |
|  | Passer | Team | Dropbacks | EPA/play | CPOE |
| <img
src="https://a.espncdn.com/i/headshots/college-football/players/full/4879250.png"
height="40" /> | Cole Payton | North Dakota State | 244 | 0.452 | 0.077 |
| <img
src="https://a.espncdn.com/i/headshots/college-football/players/full/5079712.png"
height="40" /> | Julian Sayin | Ohio State | 405 | 0.437 | 0.150 |
| <img
src="https://a.espncdn.com/i/headshots/college-football/players/full/5084180.png"
height="40" /> | Diego Pavia | Vanderbilt | 397 | 0.421 | 0.091 |
| <img
src="https://a.espncdn.com/i/headshots/college-football/players/full/5081504.png"
height="40" /> | Blake Horvath | Navy | 165 | 0.419 | 0.023 |
| <img
src="https://a.espncdn.com/i/headshots/college-football/players/full/4685454.png"
height="40" /> | Jayden Maiava | USC | 411 | 0.387 | 0.073 |
| <img
src="https://a.espncdn.com/i/headshots/college-football/players/full/4429582.png"
height="40" /> | Joe Fagnano | UConn | 421 | 0.364 | 0.101 |
| <img
src="https://a.espncdn.com/i/headshots/college-football/players/full/5219834.png"
height="40" /> | Drew Mestemaker | North Texas | 480 | 0.348 | 0.068 |
| <img
src="https://a.espncdn.com/i/headshots/college-football/players/full/4837248.png"
height="40" /> | Fernando Mendoza | Indiana | 404 | 0.336 | 0.086 |
| <img
src="https://a.espncdn.com/i/headshots/college-football/players/full/4899046.png"
height="40" /> | Brendan Sorsby | Cincinnati | 346 | 0.335 | 0.028 |
| <img
src="https://a.espncdn.com/i/headshots/college-football/players/full/5079369.png"
height="40" /> | CJ Carr | Notre Dame | 305 | 0.335 | 0.056 |

&#10;</div>

<div id="seazlsjpsb" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#seazlsjpsb table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#seazlsjpsb thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#seazlsjpsb p { margin: 0; padding: 0; }
 #seazlsjpsb .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #seazlsjpsb .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #seazlsjpsb .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #seazlsjpsb .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #seazlsjpsb .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #seazlsjpsb .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #seazlsjpsb .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #seazlsjpsb .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #seazlsjpsb .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #seazlsjpsb .gt_column_spanner_outer:first-child { padding-left: 0; }
 #seazlsjpsb .gt_column_spanner_outer:last-child { padding-right: 0; }
 #seazlsjpsb .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #seazlsjpsb .gt_spanner_row { border-bottom-style: hidden; }
 #seazlsjpsb .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #seazlsjpsb .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #seazlsjpsb .gt_from_md> :first-child { margin-top: 0; }
 #seazlsjpsb .gt_from_md> :last-child { margin-bottom: 0; }
 #seazlsjpsb .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #seazlsjpsb .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #seazlsjpsb .gt_indent_1 { text-indent: 5px; }
 #seazlsjpsb .gt_indent_2 { text-indent: calc(5px * 2); }
 #seazlsjpsb .gt_indent_3 { text-indent: calc(5px * 3); }
 #seazlsjpsb .gt_indent_4 { text-indent: calc(5px * 4); }
 #seazlsjpsb .gt_indent_5 { text-indent: calc(5px * 5); }
 #seazlsjpsb .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #seazlsjpsb .gt_row_group_first td { border-top-width: 2px; }
 #seazlsjpsb .gt_row_group_first th { border-top-width: 2px; }
 #seazlsjpsb .gt_striped { color: #333333; background-color: #F4F4F4; }
 #seazlsjpsb .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #seazlsjpsb .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #seazlsjpsb .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #seazlsjpsb .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #seazlsjpsb .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #seazlsjpsb .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #seazlsjpsb .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #seazlsjpsb .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #seazlsjpsb .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #seazlsjpsb .gt_left { text-align: left; }
 #seazlsjpsb .gt_center { text-align: center; }
 #seazlsjpsb .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #seazlsjpsb .gt_font_normal { font-weight: normal; }
 #seazlsjpsb .gt_font_bold { font-weight: bold; }
 #seazlsjpsb .gt_font_italic { font-style: italic; }
 #seazlsjpsb .gt_super { font-size: 65%; }
 #seazlsjpsb .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #seazlsjpsb .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #seazlsjpsb .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #seazlsjpsb .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #seazlsjpsb .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #seazlsjpsb .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| cfb_ratings top 25 — 2025 (opponent-adjusted EPA components) |  |  |  |  |  |
|----|----|----|----|----|----|
| published cfb_ratings release; net = off − def + st |  |  |  |  |  |
| logo | team_id | adj_off_epa | adj_def_epa | adj_st_epa | adj_net |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/84.png"
height="32" /> | 84 | 0.336 | −0.200 | 0.356 | 0.536 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/87.png"
height="32" /> | 87 | 0.298 | −0.225 | −0.386 | 0.523 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/194.png"
height="32" /> | 194 | 0.250 | −0.270 | −0.568 | 0.520 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/2390.png"
height="32" /> | 2390 | 0.214 | −0.281 | −0.088 | 0.494 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/2483.png"
height="32" /> | 2483 | 0.204 | −0.227 | 0.635 | 0.431 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/2641.png"
height="32" /> | 2641 | 0.054 | −0.370 | 0.687 | 0.425 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/245.png"
height="32" /> | 245 | 0.224 | −0.200 | −0.396 | 0.425 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/145.png"
height="32" /> | 145 | 0.237 | −0.127 | 0.862 | 0.364 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/333.png"
height="32" /> | 333 | 0.145 | −0.187 | −0.057 | 0.332 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/264.png"
height="32" /> | 264 | 0.202 | −0.129 | 0.033 | 0.332 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/201.png"
height="32" /> | 201 | −0.011 | −0.342 | 0.813 | 0.331 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/61.png"
height="32" /> | 61 | 0.162 | −0.168 | 0.933 | 0.329 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/251.png"
height="32" /> | 251 | 0.112 | −0.214 | 0.321 | 0.326 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/254.png"
height="32" /> | 254 | 0.223 | −0.081 | 0.116 | 0.305 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/30.png"
height="32" /> | 30 | 0.242 | −0.060 | −0.676 | 0.302 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/142.png"
height="32" /> | 142 | 0.081 | −0.212 | −0.106 | 0.293 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/52.png"
height="32" /> | 52 | 0.208 | −0.071 | −0.300 | 0.280 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/238.png"
height="32" /> | 238 | 0.329 | 0.054 | 0.783 | 0.275 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/2294.png"
height="32" /> | 2294 | 0.044 | −0.225 | 0.443 | 0.269 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/213.png"
height="32" /> | 213 | 0.138 | −0.119 | 0.551 | 0.257 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/221.png"
height="32" /> | 221 | 0.059 | −0.192 | −0.076 | 0.251 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/97.png"
height="32" /> | 97 | 0.102 | −0.147 | 0.083 | 0.249 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/9.png"
height="32" /> | 9 | 0.082 | −0.160 | −0.435 | 0.242 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/252.png"
height="32" /> | 252 | 0.130 | −0.108 | −0.135 | 0.239 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/2.png"
height="32" /> | 2 | 0.070 | −0.166 | 0.301 | 0.237 |

&#10;</div>

<div id="eqgbyrnrtk" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#eqgbyrnrtk table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#eqgbyrnrtk thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#eqgbyrnrtk p { margin: 0; padding: 0; }
 #eqgbyrnrtk .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #eqgbyrnrtk .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #eqgbyrnrtk .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #eqgbyrnrtk .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #eqgbyrnrtk .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #eqgbyrnrtk .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #eqgbyrnrtk .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #eqgbyrnrtk .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #eqgbyrnrtk .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #eqgbyrnrtk .gt_column_spanner_outer:first-child { padding-left: 0; }
 #eqgbyrnrtk .gt_column_spanner_outer:last-child { padding-right: 0; }
 #eqgbyrnrtk .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #eqgbyrnrtk .gt_spanner_row { border-bottom-style: hidden; }
 #eqgbyrnrtk .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #eqgbyrnrtk .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #eqgbyrnrtk .gt_from_md> :first-child { margin-top: 0; }
 #eqgbyrnrtk .gt_from_md> :last-child { margin-bottom: 0; }
 #eqgbyrnrtk .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #eqgbyrnrtk .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #eqgbyrnrtk .gt_indent_1 { text-indent: 5px; }
 #eqgbyrnrtk .gt_indent_2 { text-indent: calc(5px * 2); }
 #eqgbyrnrtk .gt_indent_3 { text-indent: calc(5px * 3); }
 #eqgbyrnrtk .gt_indent_4 { text-indent: calc(5px * 4); }
 #eqgbyrnrtk .gt_indent_5 { text-indent: calc(5px * 5); }
 #eqgbyrnrtk .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #eqgbyrnrtk .gt_row_group_first td { border-top-width: 2px; }
 #eqgbyrnrtk .gt_row_group_first th { border-top-width: 2px; }
 #eqgbyrnrtk .gt_striped { color: #333333; background-color: #F4F4F4; }
 #eqgbyrnrtk .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #eqgbyrnrtk .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #eqgbyrnrtk .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #eqgbyrnrtk .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #eqgbyrnrtk .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #eqgbyrnrtk .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #eqgbyrnrtk .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #eqgbyrnrtk .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #eqgbyrnrtk .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #eqgbyrnrtk .gt_left { text-align: left; }
 #eqgbyrnrtk .gt_center { text-align: center; }
 #eqgbyrnrtk .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #eqgbyrnrtk .gt_font_normal { font-weight: normal; }
 #eqgbyrnrtk .gt_font_bold { font-weight: bold; }
 #eqgbyrnrtk .gt_font_italic { font-style: italic; }
 #eqgbyrnrtk .gt_super { font-size: 65%; }
 #eqgbyrnrtk .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #eqgbyrnrtk .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #eqgbyrnrtk .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #eqgbyrnrtk .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #eqgbyrnrtk .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #eqgbyrnrtk .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| cfb_recruiting_proj top 10 — 2025 published asset |  |  |  |  |
|----|----|----|----|----|
| the recruiting-talent projection surface (full writeup: cfb_recruiting_proj.md) |  |  |  |  |
| logo | season | pred_wins | pred_margin | pred_net_epa |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/61.png"
height="32" /> | 2,025.000 | 9.883 | 18.772 | <na> |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/333.png"
height="32" /> | 2,025.000 | 8.889 | 14.324 | <na> |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/251.png"
height="32" /> | 2,025.000 | 10.033 | 18.911 | <na> |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/2483.png"
height="32" /> | 2,025.000 | 9.689 | 17.380 | <na> |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/194.png"
height="32" /> | 2,025.000 | 10.338 | 20.234 | <na> |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/87.png"
height="32" /> | 2,025.000 | 10.620 | 21.412 | <na> |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/245.png"
height="32" /> | 2,025.000 | 8.823 | 13.921 | <na> |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/213.png"
height="32" /> | 2,025.000 | 10.359 | 20.006 | <na> |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/130.png"
height="32" /> | 2,025.000 | 7.868 | 9.567 | <na> |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/99.png"
height="32" /> | 2,025.000 | 8.857 | 13.702 | <na> |

&#10;</div>

## Provenance & reproducibility

- **Per-model cards** (`ep.md` … `pregame_wp.md`): generated by
  `python -m cfb_model_reports` (stage `cfb_model_70_reports` /
  `scripts/cfb_models.sh 70`) from the training pipeline’s artifacts —
  the cards are compiled documents already; regenerating them is part of
  the model pipeline.
- **This page** (`deepdive.qmd` → `deepdive.md`): rendered by
  `scripts/render_model_docs.sh` (Quarto → GFM, `--output-dir .` because
  this directory is also a Quarto website project). Inputs:
  `python/artifacts/` (LOSO OOF parquets, `pbp_full.parquet`, boosters —
  regenerate via `scripts/cfb_models.sh`), the per-model **analysis
  frames** `python/artifacts/analysis/`
  (`CFB_MODELS_ARGS="export-analysis --pbp   python/artifacts/pbp_full.parquet --out-dir python/artifacts/analysis"   scripts/cfb_models.sh 30`,
  or the same subcommand via `python -m cfb_model_build.model_training`;
  the CI pipeline runs it right after `ingest`), the stage-10 scored
  frame `python/artifacts/model_pbp_2025.parquet` when present
  (`scripts/cfb_models.sh 10` for the season), and the published
  `espn_cfb_model_pbp` / `cfb_ratings` releases (cached under
  `docs/models/.cache/`, gitignored).
- **Training data:** seasons 2004–2025 (the corpus table above is
  computed from the exact training frame). Models, gates and lineage:
  `models/manifest.yaml`, `models/REGISTRY.md`.
- **Forward-looking notes** live in [roadmap.md](roadmap.md) — the
  hand-authored companion the generator cannot clobber.

## Avenues for improvement & open issues

- **Resolved (2026-09-01, PR \#56):** *Per-model SHAP for the full
  suite* — the trainer now exports the engineered feature matrix per
  model (`export-analysis` →
  `python/artifacts/analysis/analysis_{ep,wp,xpass,cp}.parquet`, built
  through the same `*_matrix` code paths the fits use) and this page
  attributes EP, WP-spread, WP-naive, xPass and CP against the promoted
  boosters, with a by-season slice for WP.
- **Resolved (2026-09-01, PR \#56):** *Athlete ids in model_pbp* — the
  ids were never absent upstream (sdv-py’s participants module emits
  them and every final.json carried them); the model_pbp builder’s
  `DESCRIPTOR_COLS` was the drop point. The frame now carries
  `passer_player_id`, `rusher_player_name`/`_id`,
  `receiver_player_name`/`_id` (Int64, additive), and the passer table
  keys on the id and ships headshots. The published release gains the
  columns on its next stage-10 + publish run.
- **Known issue:** `python/artifacts/` is a build tree, not committed —
  a fresh clone must run the model pipeline before this page renders.
  That is deliberate (artifacts are large), but it makes this page’s
  reproducibility contingent on the pipeline run; the cards’ generator
  has the same property.
