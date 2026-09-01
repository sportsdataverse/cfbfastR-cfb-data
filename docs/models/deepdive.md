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

<div id="fnfarcfnui" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#fnfarcfnui table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#fnfarcfnui thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#fnfarcfnui p { margin: 0; padding: 0; }
 #fnfarcfnui .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #fnfarcfnui .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #fnfarcfnui .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #fnfarcfnui .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #fnfarcfnui .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #fnfarcfnui .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #fnfarcfnui .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #fnfarcfnui .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #fnfarcfnui .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #fnfarcfnui .gt_column_spanner_outer:first-child { padding-left: 0; }
 #fnfarcfnui .gt_column_spanner_outer:last-child { padding-right: 0; }
 #fnfarcfnui .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #fnfarcfnui .gt_spanner_row { border-bottom-style: hidden; }
 #fnfarcfnui .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #fnfarcfnui .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #fnfarcfnui .gt_from_md> :first-child { margin-top: 0; }
 #fnfarcfnui .gt_from_md> :last-child { margin-bottom: 0; }
 #fnfarcfnui .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #fnfarcfnui .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #fnfarcfnui .gt_indent_1 { text-indent: 5px; }
 #fnfarcfnui .gt_indent_2 { text-indent: calc(5px * 2); }
 #fnfarcfnui .gt_indent_3 { text-indent: calc(5px * 3); }
 #fnfarcfnui .gt_indent_4 { text-indent: calc(5px * 4); }
 #fnfarcfnui .gt_indent_5 { text-indent: calc(5px * 5); }
 #fnfarcfnui .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #fnfarcfnui .gt_row_group_first td { border-top-width: 2px; }
 #fnfarcfnui .gt_row_group_first th { border-top-width: 2px; }
 #fnfarcfnui .gt_striped { color: #333333; background-color: #F4F4F4; }
 #fnfarcfnui .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #fnfarcfnui .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #fnfarcfnui .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #fnfarcfnui .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #fnfarcfnui .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #fnfarcfnui .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #fnfarcfnui .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #fnfarcfnui .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #fnfarcfnui .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #fnfarcfnui .gt_left { text-align: left; }
 #fnfarcfnui .gt_center { text-align: center; }
 #fnfarcfnui .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #fnfarcfnui .gt_font_normal { font-weight: normal; }
 #fnfarcfnui .gt_font_bold { font-weight: bold; }
 #fnfarcfnui .gt_font_italic { font-style: italic; }
 #fnfarcfnui .gt_super { font-size: 65%; }
 #fnfarcfnui .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #fnfarcfnui .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #fnfarcfnui .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #fnfarcfnui .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #fnfarcfnui .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #fnfarcfnui .gt_asterisk { font-size: 100%; vertical-align: 0; }
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

<div id="hbmemweabj" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#hbmemweabj table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#hbmemweabj thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#hbmemweabj p { margin: 0; padding: 0; }
 #hbmemweabj .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #hbmemweabj .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #hbmemweabj .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #hbmemweabj .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #hbmemweabj .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #hbmemweabj .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #hbmemweabj .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #hbmemweabj .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #hbmemweabj .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #hbmemweabj .gt_column_spanner_outer:first-child { padding-left: 0; }
 #hbmemweabj .gt_column_spanner_outer:last-child { padding-right: 0; }
 #hbmemweabj .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #hbmemweabj .gt_spanner_row { border-bottom-style: hidden; }
 #hbmemweabj .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #hbmemweabj .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #hbmemweabj .gt_from_md> :first-child { margin-top: 0; }
 #hbmemweabj .gt_from_md> :last-child { margin-bottom: 0; }
 #hbmemweabj .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #hbmemweabj .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #hbmemweabj .gt_indent_1 { text-indent: 5px; }
 #hbmemweabj .gt_indent_2 { text-indent: calc(5px * 2); }
 #hbmemweabj .gt_indent_3 { text-indent: calc(5px * 3); }
 #hbmemweabj .gt_indent_4 { text-indent: calc(5px * 4); }
 #hbmemweabj .gt_indent_5 { text-indent: calc(5px * 5); }
 #hbmemweabj .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #hbmemweabj .gt_row_group_first td { border-top-width: 2px; }
 #hbmemweabj .gt_row_group_first th { border-top-width: 2px; }
 #hbmemweabj .gt_striped { color: #333333; background-color: #F4F4F4; }
 #hbmemweabj .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #hbmemweabj .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #hbmemweabj .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #hbmemweabj .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #hbmemweabj .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #hbmemweabj .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #hbmemweabj .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #hbmemweabj .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #hbmemweabj .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #hbmemweabj .gt_left { text-align: left; }
 #hbmemweabj .gt_center { text-align: center; }
 #hbmemweabj .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #hbmemweabj .gt_font_normal { font-weight: normal; }
 #hbmemweabj .gt_font_bold { font-weight: bold; }
 #hbmemweabj .gt_font_italic { font-style: italic; }
 #hbmemweabj .gt_super { font-size: 65%; }
 #hbmemweabj .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #hbmemweabj .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #hbmemweabj .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #hbmemweabj .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #hbmemweabj .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #hbmemweabj .gt_asterisk { font-size: 100%; vertical-align: 0; }
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

## SHAP — the EP booster attributed

<img src="deepdive_files/figure-commonmark/cell-9-output-1.png"
width="420" height="300"
alt="TreeSHAP (class-margin space), mean |contribution| across the 7 next-score classes; 4,000-play sample." />

Field position dominates, clock second, score state third — the
cfbscrapR ordering, now attributed play-by-play rather than asserted.
The multiclass `pred_contribs` tensor is 3-D (the recorded gotcha); the
chart aggregates \|contribution\| across the seven class margins and is
labeled accordingly.

## Results — the published surfaces, identified

<div id="crjllnxorl" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#crjllnxorl table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#crjllnxorl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#crjllnxorl p { margin: 0; padding: 0; }
 #crjllnxorl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #crjllnxorl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #crjllnxorl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #crjllnxorl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #crjllnxorl .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #crjllnxorl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #crjllnxorl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #crjllnxorl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #crjllnxorl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #crjllnxorl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #crjllnxorl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #crjllnxorl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #crjllnxorl .gt_spanner_row { border-bottom-style: hidden; }
 #crjllnxorl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #crjllnxorl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #crjllnxorl .gt_from_md> :first-child { margin-top: 0; }
 #crjllnxorl .gt_from_md> :last-child { margin-bottom: 0; }
 #crjllnxorl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #crjllnxorl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #crjllnxorl .gt_indent_1 { text-indent: 5px; }
 #crjllnxorl .gt_indent_2 { text-indent: calc(5px * 2); }
 #crjllnxorl .gt_indent_3 { text-indent: calc(5px * 3); }
 #crjllnxorl .gt_indent_4 { text-indent: calc(5px * 4); }
 #crjllnxorl .gt_indent_5 { text-indent: calc(5px * 5); }
 #crjllnxorl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #crjllnxorl .gt_row_group_first td { border-top-width: 2px; }
 #crjllnxorl .gt_row_group_first th { border-top-width: 2px; }
 #crjllnxorl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #crjllnxorl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #crjllnxorl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #crjllnxorl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #crjllnxorl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #crjllnxorl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #crjllnxorl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #crjllnxorl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #crjllnxorl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #crjllnxorl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #crjllnxorl .gt_left { text-align: left; }
 #crjllnxorl .gt_center { text-align: center; }
 #crjllnxorl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #crjllnxorl .gt_font_normal { font-weight: normal; }
 #crjllnxorl .gt_font_bold { font-weight: bold; }
 #crjllnxorl .gt_font_italic { font-style: italic; }
 #crjllnxorl .gt_super { font-size: 65%; }
 #crjllnxorl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #crjllnxorl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #crjllnxorl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #crjllnxorl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #crjllnxorl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #crjllnxorl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Top 10 offenses by EPA/play — 2025 published model_pbp |  |  |  |
|----|----|----|----|
|  | Offense | Plays | EPA/play |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/2011.png"
height="36" /> | Alabama State | 64 | 0.504 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/238.png"
height="36" /> | Vanderbilt | 796 | 0.287 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/2241.png"
height="36" /> | Gardner-Webb | 61 | 0.272 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/2449.png"
height="36" /> | North Dakota State | 795 | 0.268 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/149.png"
height="36" /> | Montana | 59 | 0.266 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/87.png"
height="36" /> | Notre Dame | 748 | 0.247 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/2132.png"
height="36" /> | Cincinnati | 764 | 0.241 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/2426.png"
height="36" /> | Navy | 813 | 0.238 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/249.png"
height="36" /> | North Texas | 984 | 0.231 |
| <img src="https://a.espncdn.com/i/teamlogos/ncaa/500/41.png"
height="36" /> | UConn | 846 | 0.228 |

&#10;</div>

<div id="iyjgjkxjjj" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#iyjgjkxjjj table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#iyjgjkxjjj thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#iyjgjkxjjj p { margin: 0; padding: 0; }
 #iyjgjkxjjj .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #iyjgjkxjjj .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #iyjgjkxjjj .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #iyjgjkxjjj .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #iyjgjkxjjj .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #iyjgjkxjjj .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #iyjgjkxjjj .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #iyjgjkxjjj .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #iyjgjkxjjj .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #iyjgjkxjjj .gt_column_spanner_outer:first-child { padding-left: 0; }
 #iyjgjkxjjj .gt_column_spanner_outer:last-child { padding-right: 0; }
 #iyjgjkxjjj .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #iyjgjkxjjj .gt_spanner_row { border-bottom-style: hidden; }
 #iyjgjkxjjj .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #iyjgjkxjjj .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #iyjgjkxjjj .gt_from_md> :first-child { margin-top: 0; }
 #iyjgjkxjjj .gt_from_md> :last-child { margin-bottom: 0; }
 #iyjgjkxjjj .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #iyjgjkxjjj .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #iyjgjkxjjj .gt_indent_1 { text-indent: 5px; }
 #iyjgjkxjjj .gt_indent_2 { text-indent: calc(5px * 2); }
 #iyjgjkxjjj .gt_indent_3 { text-indent: calc(5px * 3); }
 #iyjgjkxjjj .gt_indent_4 { text-indent: calc(5px * 4); }
 #iyjgjkxjjj .gt_indent_5 { text-indent: calc(5px * 5); }
 #iyjgjkxjjj .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #iyjgjkxjjj .gt_row_group_first td { border-top-width: 2px; }
 #iyjgjkxjjj .gt_row_group_first th { border-top-width: 2px; }
 #iyjgjkxjjj .gt_striped { color: #333333; background-color: #F4F4F4; }
 #iyjgjkxjjj .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #iyjgjkxjjj .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #iyjgjkxjjj .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #iyjgjkxjjj .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #iyjgjkxjjj .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #iyjgjkxjjj .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #iyjgjkxjjj .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #iyjgjkxjjj .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #iyjgjkxjjj .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #iyjgjkxjjj .gt_left { text-align: left; }
 #iyjgjkxjjj .gt_center { text-align: center; }
 #iyjgjkxjjj .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #iyjgjkxjjj .gt_font_normal { font-weight: normal; }
 #iyjgjkxjjj .gt_font_bold { font-weight: bold; }
 #iyjgjkxjjj .gt_font_italic { font-style: italic; }
 #iyjgjkxjjj .gt_super { font-size: 65%; }
 #iyjgjkxjjj .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #iyjgjkxjjj .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #iyjgjkxjjj .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #iyjgjkxjjj .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #iyjgjkxjjj .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #iyjgjkxjjj .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Passer EPA/play leaders with CPOE — 2025 (min 150 dropbacks) |  |  |  |  |
|----|----|----|----|----|
| epa and cpoe are the published model columns; ESPN ships no per-athlete id in this frame, so names identify |  |  |  |  |
| Passer | Team | Dropbacks | EPA/play | CPOE |
| Cole Payton | North Dakota State | 244 | 0.424 | 0.076 |
| Julian Sayin | Ohio State | 405 | 0.402 | 0.150 |
| Blake Horvath | Navy | 165 | 0.381 | 0.023 |
| Jayden Maiava | USC | 411 | 0.362 | 0.072 |
| Diego Pavia | Vanderbilt | 397 | 0.339 | 0.091 |
| Joe Fagnano | UConn | 421 | 0.323 | 0.100 |
| Steve Angeli | Syracuse | 159 | 0.316 | 0.030 |
| CJ Carr | Notre Dame | 305 | 0.303 | 0.055 |
| Fernando Mendoza | Indiana | 404 | 0.301 | 0.086 |
| Drew Mestemaker | North Texas | 480 | 0.294 | 0.068 |

&#10;</div>

<div id="maqmukslhb" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#maqmukslhb table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#maqmukslhb thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#maqmukslhb p { margin: 0; padding: 0; }
 #maqmukslhb .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #maqmukslhb .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #maqmukslhb .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #maqmukslhb .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #maqmukslhb .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #maqmukslhb .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #maqmukslhb .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #maqmukslhb .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #maqmukslhb .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #maqmukslhb .gt_column_spanner_outer:first-child { padding-left: 0; }
 #maqmukslhb .gt_column_spanner_outer:last-child { padding-right: 0; }
 #maqmukslhb .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #maqmukslhb .gt_spanner_row { border-bottom-style: hidden; }
 #maqmukslhb .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #maqmukslhb .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #maqmukslhb .gt_from_md> :first-child { margin-top: 0; }
 #maqmukslhb .gt_from_md> :last-child { margin-bottom: 0; }
 #maqmukslhb .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #maqmukslhb .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #maqmukslhb .gt_indent_1 { text-indent: 5px; }
 #maqmukslhb .gt_indent_2 { text-indent: calc(5px * 2); }
 #maqmukslhb .gt_indent_3 { text-indent: calc(5px * 3); }
 #maqmukslhb .gt_indent_4 { text-indent: calc(5px * 4); }
 #maqmukslhb .gt_indent_5 { text-indent: calc(5px * 5); }
 #maqmukslhb .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #maqmukslhb .gt_row_group_first td { border-top-width: 2px; }
 #maqmukslhb .gt_row_group_first th { border-top-width: 2px; }
 #maqmukslhb .gt_striped { color: #333333; background-color: #F4F4F4; }
 #maqmukslhb .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #maqmukslhb .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #maqmukslhb .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #maqmukslhb .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #maqmukslhb .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #maqmukslhb .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #maqmukslhb .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #maqmukslhb .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #maqmukslhb .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #maqmukslhb .gt_left { text-align: left; }
 #maqmukslhb .gt_center { text-align: center; }
 #maqmukslhb .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #maqmukslhb .gt_font_normal { font-weight: normal; }
 #maqmukslhb .gt_font_bold { font-weight: bold; }
 #maqmukslhb .gt_font_italic { font-style: italic; }
 #maqmukslhb .gt_super { font-size: 65%; }
 #maqmukslhb .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #maqmukslhb .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #maqmukslhb .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #maqmukslhb .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #maqmukslhb .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #maqmukslhb .gt_asterisk { font-size: 100%; vertical-align: 0; }
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

<div id="kyaetgkjxe" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#kyaetgkjxe table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#kyaetgkjxe thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#kyaetgkjxe p { margin: 0; padding: 0; }
 #kyaetgkjxe .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #kyaetgkjxe .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #kyaetgkjxe .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #kyaetgkjxe .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #kyaetgkjxe .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #kyaetgkjxe .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #kyaetgkjxe .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #kyaetgkjxe .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #kyaetgkjxe .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #kyaetgkjxe .gt_column_spanner_outer:first-child { padding-left: 0; }
 #kyaetgkjxe .gt_column_spanner_outer:last-child { padding-right: 0; }
 #kyaetgkjxe .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #kyaetgkjxe .gt_spanner_row { border-bottom-style: hidden; }
 #kyaetgkjxe .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #kyaetgkjxe .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #kyaetgkjxe .gt_from_md> :first-child { margin-top: 0; }
 #kyaetgkjxe .gt_from_md> :last-child { margin-bottom: 0; }
 #kyaetgkjxe .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #kyaetgkjxe .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #kyaetgkjxe .gt_indent_1 { text-indent: 5px; }
 #kyaetgkjxe .gt_indent_2 { text-indent: calc(5px * 2); }
 #kyaetgkjxe .gt_indent_3 { text-indent: calc(5px * 3); }
 #kyaetgkjxe .gt_indent_4 { text-indent: calc(5px * 4); }
 #kyaetgkjxe .gt_indent_5 { text-indent: calc(5px * 5); }
 #kyaetgkjxe .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #kyaetgkjxe .gt_row_group_first td { border-top-width: 2px; }
 #kyaetgkjxe .gt_row_group_first th { border-top-width: 2px; }
 #kyaetgkjxe .gt_striped { color: #333333; background-color: #F4F4F4; }
 #kyaetgkjxe .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #kyaetgkjxe .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #kyaetgkjxe .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #kyaetgkjxe .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #kyaetgkjxe .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #kyaetgkjxe .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #kyaetgkjxe .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #kyaetgkjxe .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #kyaetgkjxe .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #kyaetgkjxe .gt_left { text-align: left; }
 #kyaetgkjxe .gt_center { text-align: center; }
 #kyaetgkjxe .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #kyaetgkjxe .gt_font_normal { font-weight: normal; }
 #kyaetgkjxe .gt_font_bold { font-weight: bold; }
 #kyaetgkjxe .gt_font_italic { font-style: italic; }
 #kyaetgkjxe .gt_super { font-size: 65%; }
 #kyaetgkjxe .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #kyaetgkjxe .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #kyaetgkjxe .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #kyaetgkjxe .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #kyaetgkjxe .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #kyaetgkjxe .gt_asterisk { font-size: 100%; vertical-align: 0; }
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
  regenerate via `scripts/cfb_models.sh`) and the published
  `espn_cfb_model_pbp` / `cfb_ratings` releases (cached under
  `docs/models/.cache/`, gitignored).
- **Training data:** seasons 2004–2025 (the corpus table above is
  computed from the exact training frame). Models, gates and lineage:
  `models/manifest.yaml`, `models/REGISTRY.md`.
- **Forward-looking notes** live in [roadmap.md](roadmap.md) — the
  hand-authored companion the generator cannot clobber.

## Avenues for improvement & open issues

- **Per-model SHAP for the full suite** — this page attributes EP;
  wiring the same pass for WP/CP/xpass needs their engineered features
  (`spread_time`, `adj_TimeSecsRem`, …) exported into an analysis frame
  by the trainer, which is the cleaner fix than re-deriving them here.
- **Athlete ids in model_pbp** — the published frame carries passer
  *names* only; adding ESPN athlete ids would let the leader tables ship
  headshots the way the NFL/MLB docs do.
- **Known issue:** `python/artifacts/` is a build tree, not committed —
  a fresh clone must run the model pipeline before this page renders.
  That is deliberate (artifacts are large), but it makes this page’s
  reproducibility contingent on the pipeline run; the cards’ generator
  has the same property.
