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

<div id="wfgletsely" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#wfgletsely table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#wfgletsely thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#wfgletsely p { margin: 0; padding: 0; }
 #wfgletsely .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #wfgletsely .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #wfgletsely .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #wfgletsely .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #wfgletsely .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #wfgletsely .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #wfgletsely .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #wfgletsely .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #wfgletsely .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #wfgletsely .gt_column_spanner_outer:first-child { padding-left: 0; }
 #wfgletsely .gt_column_spanner_outer:last-child { padding-right: 0; }
 #wfgletsely .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #wfgletsely .gt_spanner_row { border-bottom-style: hidden; }
 #wfgletsely .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #wfgletsely .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #wfgletsely .gt_from_md> :first-child { margin-top: 0; }
 #wfgletsely .gt_from_md> :last-child { margin-bottom: 0; }
 #wfgletsely .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #wfgletsely .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #wfgletsely .gt_indent_1 { text-indent: 5px; }
 #wfgletsely .gt_indent_2 { text-indent: calc(5px * 2); }
 #wfgletsely .gt_indent_3 { text-indent: calc(5px * 3); }
 #wfgletsely .gt_indent_4 { text-indent: calc(5px * 4); }
 #wfgletsely .gt_indent_5 { text-indent: calc(5px * 5); }
 #wfgletsely .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #wfgletsely .gt_row_group_first td { border-top-width: 2px; }
 #wfgletsely .gt_row_group_first th { border-top-width: 2px; }
 #wfgletsely .gt_striped { color: #333333; background-color: #F4F4F4; }
 #wfgletsely .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #wfgletsely .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #wfgletsely .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #wfgletsely .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #wfgletsely .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #wfgletsely .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #wfgletsely .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #wfgletsely .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #wfgletsely .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #wfgletsely .gt_left { text-align: left; }
 #wfgletsely .gt_center { text-align: center; }
 #wfgletsely .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #wfgletsely .gt_font_normal { font-weight: normal; }
 #wfgletsely .gt_font_bold { font-weight: bold; }
 #wfgletsely .gt_font_italic { font-style: italic; }
 #wfgletsely .gt_super { font-size: 65%; }
 #wfgletsely .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #wfgletsely .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #wfgletsely .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #wfgletsely .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #wfgletsely .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #wfgletsely .gt_asterisk { font-size: 100%; vertical-align: 0; }
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

<div id="zzkarzjrdk" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#zzkarzjrdk table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#zzkarzjrdk thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#zzkarzjrdk p { margin: 0; padding: 0; }
 #zzkarzjrdk .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #zzkarzjrdk .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #zzkarzjrdk .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #zzkarzjrdk .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #zzkarzjrdk .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #zzkarzjrdk .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #zzkarzjrdk .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #zzkarzjrdk .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #zzkarzjrdk .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #zzkarzjrdk .gt_column_spanner_outer:first-child { padding-left: 0; }
 #zzkarzjrdk .gt_column_spanner_outer:last-child { padding-right: 0; }
 #zzkarzjrdk .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #zzkarzjrdk .gt_spanner_row { border-bottom-style: hidden; }
 #zzkarzjrdk .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #zzkarzjrdk .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #zzkarzjrdk .gt_from_md> :first-child { margin-top: 0; }
 #zzkarzjrdk .gt_from_md> :last-child { margin-bottom: 0; }
 #zzkarzjrdk .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #zzkarzjrdk .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #zzkarzjrdk .gt_indent_1 { text-indent: 5px; }
 #zzkarzjrdk .gt_indent_2 { text-indent: calc(5px * 2); }
 #zzkarzjrdk .gt_indent_3 { text-indent: calc(5px * 3); }
 #zzkarzjrdk .gt_indent_4 { text-indent: calc(5px * 4); }
 #zzkarzjrdk .gt_indent_5 { text-indent: calc(5px * 5); }
 #zzkarzjrdk .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #zzkarzjrdk .gt_row_group_first td { border-top-width: 2px; }
 #zzkarzjrdk .gt_row_group_first th { border-top-width: 2px; }
 #zzkarzjrdk .gt_striped { color: #333333; background-color: #F4F4F4; }
 #zzkarzjrdk .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #zzkarzjrdk .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #zzkarzjrdk .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #zzkarzjrdk .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #zzkarzjrdk .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #zzkarzjrdk .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #zzkarzjrdk .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #zzkarzjrdk .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #zzkarzjrdk .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #zzkarzjrdk .gt_left { text-align: left; }
 #zzkarzjrdk .gt_center { text-align: center; }
 #zzkarzjrdk .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #zzkarzjrdk .gt_font_normal { font-weight: normal; }
 #zzkarzjrdk .gt_font_bold { font-weight: bold; }
 #zzkarzjrdk .gt_font_italic { font-style: italic; }
 #zzkarzjrdk .gt_super { font-size: 65%; }
 #zzkarzjrdk .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #zzkarzjrdk .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #zzkarzjrdk .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #zzkarzjrdk .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #zzkarzjrdk .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #zzkarzjrdk .gt_asterisk { font-size: 100%; vertical-align: 0; }
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

<div id="dcoxgxsvqp" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#dcoxgxsvqp table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#dcoxgxsvqp thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#dcoxgxsvqp p { margin: 0; padding: 0; }
 #dcoxgxsvqp .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #dcoxgxsvqp .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #dcoxgxsvqp .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #dcoxgxsvqp .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #dcoxgxsvqp .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #dcoxgxsvqp .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #dcoxgxsvqp .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #dcoxgxsvqp .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #dcoxgxsvqp .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #dcoxgxsvqp .gt_column_spanner_outer:first-child { padding-left: 0; }
 #dcoxgxsvqp .gt_column_spanner_outer:last-child { padding-right: 0; }
 #dcoxgxsvqp .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #dcoxgxsvqp .gt_spanner_row { border-bottom-style: hidden; }
 #dcoxgxsvqp .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #dcoxgxsvqp .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #dcoxgxsvqp .gt_from_md> :first-child { margin-top: 0; }
 #dcoxgxsvqp .gt_from_md> :last-child { margin-bottom: 0; }
 #dcoxgxsvqp .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #dcoxgxsvqp .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #dcoxgxsvqp .gt_indent_1 { text-indent: 5px; }
 #dcoxgxsvqp .gt_indent_2 { text-indent: calc(5px * 2); }
 #dcoxgxsvqp .gt_indent_3 { text-indent: calc(5px * 3); }
 #dcoxgxsvqp .gt_indent_4 { text-indent: calc(5px * 4); }
 #dcoxgxsvqp .gt_indent_5 { text-indent: calc(5px * 5); }
 #dcoxgxsvqp .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #dcoxgxsvqp .gt_row_group_first td { border-top-width: 2px; }
 #dcoxgxsvqp .gt_row_group_first th { border-top-width: 2px; }
 #dcoxgxsvqp .gt_striped { color: #333333; background-color: #F4F4F4; }
 #dcoxgxsvqp .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #dcoxgxsvqp .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #dcoxgxsvqp .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #dcoxgxsvqp .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #dcoxgxsvqp .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #dcoxgxsvqp .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #dcoxgxsvqp .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #dcoxgxsvqp .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #dcoxgxsvqp .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #dcoxgxsvqp .gt_left { text-align: left; }
 #dcoxgxsvqp .gt_center { text-align: center; }
 #dcoxgxsvqp .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #dcoxgxsvqp .gt_font_normal { font-weight: normal; }
 #dcoxgxsvqp .gt_font_bold { font-weight: bold; }
 #dcoxgxsvqp .gt_font_italic { font-style: italic; }
 #dcoxgxsvqp .gt_super { font-size: 65%; }
 #dcoxgxsvqp .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #dcoxgxsvqp .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #dcoxgxsvqp .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #dcoxgxsvqp .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #dcoxgxsvqp .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #dcoxgxsvqp .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Which booster each attribution below explains |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
| analysis frames from pbp_full_v2.parquet (sha256 48d7e7c8561b, export-analysis 2026-09-02); booster sha256 compared against the .ubj bundled in sdv-py 0.1.3; 'trained on' from each booster's card |  |  |  |  |  |  |  |  |
| Model | Artifact | sha256\[:12\] | vs bundle | Trained on | Frame | Features | Frame rows | SHAP sample |
| EP (7-class) | python/artifacts/v2/ep_model.ubj | 44516f561836 | = sdv-py bundle | pbp_full_v2.parquet | same frame | 8 | 2,219,971 | 4,000 |
| WP (spread) | python/artifacts/v2/wp_spread.ubj | 486699334ef9 | = sdv-py bundle | pbp_full_v2.parquet | same frame | 13 | 2,219,971 | 4,000 |
| WP (naive) | python/artifacts/v2/wp_naive.ubj | 624ee1494330 | = sdv-py bundle | pbp_full_v2.parquet | same frame | 12 | 2,219,971 | 4,000 |
| xPass | python/artifacts/v2/xpass_model.ubj | 90a27a21e49d | = sdv-py bundle | pbp_full_v2.parquet | same frame | 7 | 1,902,481 | 4,000 |
| CP | sdv-py bundle/cfb_cp_model.ubj | 71d52772ee59 | = sdv-py bundle | n/a (no card) | n/a | 8 | 921,377 | 4,000 |

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

<div id="saejofcunn" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#saejofcunn table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#saejofcunn thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#saejofcunn p { margin: 0; padding: 0; }
 #saejofcunn .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #saejofcunn .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #saejofcunn .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #saejofcunn .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #saejofcunn .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #saejofcunn .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #saejofcunn .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #saejofcunn .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #saejofcunn .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #saejofcunn .gt_column_spanner_outer:first-child { padding-left: 0; }
 #saejofcunn .gt_column_spanner_outer:last-child { padding-right: 0; }
 #saejofcunn .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #saejofcunn .gt_spanner_row { border-bottom-style: hidden; }
 #saejofcunn .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #saejofcunn .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #saejofcunn .gt_from_md> :first-child { margin-top: 0; }
 #saejofcunn .gt_from_md> :last-child { margin-bottom: 0; }
 #saejofcunn .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #saejofcunn .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #saejofcunn .gt_indent_1 { text-indent: 5px; }
 #saejofcunn .gt_indent_2 { text-indent: calc(5px * 2); }
 #saejofcunn .gt_indent_3 { text-indent: calc(5px * 3); }
 #saejofcunn .gt_indent_4 { text-indent: calc(5px * 4); }
 #saejofcunn .gt_indent_5 { text-indent: calc(5px * 5); }
 #saejofcunn .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #saejofcunn .gt_row_group_first td { border-top-width: 2px; }
 #saejofcunn .gt_row_group_first th { border-top-width: 2px; }
 #saejofcunn .gt_striped { color: #333333; background-color: #F4F4F4; }
 #saejofcunn .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #saejofcunn .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #saejofcunn .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #saejofcunn .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #saejofcunn .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #saejofcunn .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #saejofcunn .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #saejofcunn .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #saejofcunn .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #saejofcunn .gt_left { text-align: left; }
 #saejofcunn .gt_center { text-align: center; }
 #saejofcunn .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #saejofcunn .gt_font_normal { font-weight: normal; }
 #saejofcunn .gt_font_bold { font-weight: bold; }
 #saejofcunn .gt_font_italic { font-style: italic; }
 #saejofcunn .gt_super { font-size: 65%; }
 #saejofcunn .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #saejofcunn .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #saejofcunn .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #saejofcunn .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #saejofcunn .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #saejofcunn .gt_asterisk { font-size: 100%; vertical-align: 0; }
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

<div id="hnkmizsyib" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#hnkmizsyib table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#hnkmizsyib thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#hnkmizsyib p { margin: 0; padding: 0; }
 #hnkmizsyib .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #hnkmizsyib .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #hnkmizsyib .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #hnkmizsyib .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #hnkmizsyib .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #hnkmizsyib .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #hnkmizsyib .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #hnkmizsyib .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #hnkmizsyib .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #hnkmizsyib .gt_column_spanner_outer:first-child { padding-left: 0; }
 #hnkmizsyib .gt_column_spanner_outer:last-child { padding-right: 0; }
 #hnkmizsyib .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #hnkmizsyib .gt_spanner_row { border-bottom-style: hidden; }
 #hnkmizsyib .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #hnkmizsyib .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #hnkmizsyib .gt_from_md> :first-child { margin-top: 0; }
 #hnkmizsyib .gt_from_md> :last-child { margin-bottom: 0; }
 #hnkmizsyib .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #hnkmizsyib .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #hnkmizsyib .gt_indent_1 { text-indent: 5px; }
 #hnkmizsyib .gt_indent_2 { text-indent: calc(5px * 2); }
 #hnkmizsyib .gt_indent_3 { text-indent: calc(5px * 3); }
 #hnkmizsyib .gt_indent_4 { text-indent: calc(5px * 4); }
 #hnkmizsyib .gt_indent_5 { text-indent: calc(5px * 5); }
 #hnkmizsyib .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #hnkmizsyib .gt_row_group_first td { border-top-width: 2px; }
 #hnkmizsyib .gt_row_group_first th { border-top-width: 2px; }
 #hnkmizsyib .gt_striped { color: #333333; background-color: #F4F4F4; }
 #hnkmizsyib .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #hnkmizsyib .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #hnkmizsyib .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #hnkmizsyib .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #hnkmizsyib .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #hnkmizsyib .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #hnkmizsyib .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #hnkmizsyib .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #hnkmizsyib .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #hnkmizsyib .gt_left { text-align: left; }
 #hnkmizsyib .gt_center { text-align: center; }
 #hnkmizsyib .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #hnkmizsyib .gt_font_normal { font-weight: normal; }
 #hnkmizsyib .gt_font_bold { font-weight: bold; }
 #hnkmizsyib .gt_font_italic { font-style: italic; }
 #hnkmizsyib .gt_super { font-size: 65%; }
 #hnkmizsyib .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #hnkmizsyib .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #hnkmizsyib .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #hnkmizsyib .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #hnkmizsyib .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #hnkmizsyib .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Top 10 offenses by EPA/play — 2025 model_pbp |  |  |  |
|----|----|----|----|
| source: python/artifacts/model_pbp_2025.parquet (stage 10 build tree, scored 2026-09-01 with cfb_cp_model.ubj) |  |  |  |
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

<div id="xgqavmfuyh" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#xgqavmfuyh table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#xgqavmfuyh thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#xgqavmfuyh p { margin: 0; padding: 0; }
 #xgqavmfuyh .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #xgqavmfuyh .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #xgqavmfuyh .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #xgqavmfuyh .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #xgqavmfuyh .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #xgqavmfuyh .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #xgqavmfuyh .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #xgqavmfuyh .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #xgqavmfuyh .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #xgqavmfuyh .gt_column_spanner_outer:first-child { padding-left: 0; }
 #xgqavmfuyh .gt_column_spanner_outer:last-child { padding-right: 0; }
 #xgqavmfuyh .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #xgqavmfuyh .gt_spanner_row { border-bottom-style: hidden; }
 #xgqavmfuyh .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #xgqavmfuyh .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #xgqavmfuyh .gt_from_md> :first-child { margin-top: 0; }
 #xgqavmfuyh .gt_from_md> :last-child { margin-bottom: 0; }
 #xgqavmfuyh .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #xgqavmfuyh .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #xgqavmfuyh .gt_indent_1 { text-indent: 5px; }
 #xgqavmfuyh .gt_indent_2 { text-indent: calc(5px * 2); }
 #xgqavmfuyh .gt_indent_3 { text-indent: calc(5px * 3); }
 #xgqavmfuyh .gt_indent_4 { text-indent: calc(5px * 4); }
 #xgqavmfuyh .gt_indent_5 { text-indent: calc(5px * 5); }
 #xgqavmfuyh .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #xgqavmfuyh .gt_row_group_first td { border-top-width: 2px; }
 #xgqavmfuyh .gt_row_group_first th { border-top-width: 2px; }
 #xgqavmfuyh .gt_striped { color: #333333; background-color: #F4F4F4; }
 #xgqavmfuyh .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #xgqavmfuyh .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #xgqavmfuyh .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #xgqavmfuyh .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #xgqavmfuyh .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #xgqavmfuyh .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #xgqavmfuyh .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #xgqavmfuyh .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #xgqavmfuyh .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #xgqavmfuyh .gt_left { text-align: left; }
 #xgqavmfuyh .gt_center { text-align: center; }
 #xgqavmfuyh .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #xgqavmfuyh .gt_font_normal { font-weight: normal; }
 #xgqavmfuyh .gt_font_bold { font-weight: bold; }
 #xgqavmfuyh .gt_font_italic { font-style: italic; }
 #xgqavmfuyh .gt_super { font-size: 65%; }
 #xgqavmfuyh .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #xgqavmfuyh .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #xgqavmfuyh .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #xgqavmfuyh .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #xgqavmfuyh .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #xgqavmfuyh .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Passer EPA/play leaders with CPOE — 2025 (min 150 dropbacks) |  |  |  |  |  |
|----|----|----|----|----|----|
| source: python/artifacts/model_pbp_2025.parquet (stage 10 build tree, scored 2026-09-01 with cfb_cp_model.ubj); keyed by ESPN athlete id (passer_player_id), headshots from a.espncdn.com |  |  |  |  |  |
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

<div id="zjqzehxskp" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#zjqzehxskp table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#zjqzehxskp thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#zjqzehxskp p { margin: 0; padding: 0; }
 #zjqzehxskp .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #zjqzehxskp .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #zjqzehxskp .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #zjqzehxskp .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #zjqzehxskp .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #zjqzehxskp .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #zjqzehxskp .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #zjqzehxskp .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #zjqzehxskp .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #zjqzehxskp .gt_column_spanner_outer:first-child { padding-left: 0; }
 #zjqzehxskp .gt_column_spanner_outer:last-child { padding-right: 0; }
 #zjqzehxskp .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #zjqzehxskp .gt_spanner_row { border-bottom-style: hidden; }
 #zjqzehxskp .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #zjqzehxskp .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #zjqzehxskp .gt_from_md> :first-child { margin-top: 0; }
 #zjqzehxskp .gt_from_md> :last-child { margin-bottom: 0; }
 #zjqzehxskp .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #zjqzehxskp .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #zjqzehxskp .gt_indent_1 { text-indent: 5px; }
 #zjqzehxskp .gt_indent_2 { text-indent: calc(5px * 2); }
 #zjqzehxskp .gt_indent_3 { text-indent: calc(5px * 3); }
 #zjqzehxskp .gt_indent_4 { text-indent: calc(5px * 4); }
 #zjqzehxskp .gt_indent_5 { text-indent: calc(5px * 5); }
 #zjqzehxskp .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #zjqzehxskp .gt_row_group_first td { border-top-width: 2px; }
 #zjqzehxskp .gt_row_group_first th { border-top-width: 2px; }
 #zjqzehxskp .gt_striped { color: #333333; background-color: #F4F4F4; }
 #zjqzehxskp .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #zjqzehxskp .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #zjqzehxskp .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #zjqzehxskp .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #zjqzehxskp .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #zjqzehxskp .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #zjqzehxskp .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #zjqzehxskp .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #zjqzehxskp .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #zjqzehxskp .gt_left { text-align: left; }
 #zjqzehxskp .gt_center { text-align: center; }
 #zjqzehxskp .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #zjqzehxskp .gt_font_normal { font-weight: normal; }
 #zjqzehxskp .gt_font_bold { font-weight: bold; }
 #zjqzehxskp .gt_font_italic { font-style: italic; }
 #zjqzehxskp .gt_super { font-size: 65%; }
 #zjqzehxskp .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #zjqzehxskp .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #zjqzehxskp .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #zjqzehxskp .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #zjqzehxskp .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #zjqzehxskp .gt_asterisk { font-size: 100%; vertical-align: 0; }
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

<div id="lxxelxsyfv" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#lxxelxsyfv table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#lxxelxsyfv thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#lxxelxsyfv p { margin: 0; padding: 0; }
 #lxxelxsyfv .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #lxxelxsyfv .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #lxxelxsyfv .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #lxxelxsyfv .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #lxxelxsyfv .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #lxxelxsyfv .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #lxxelxsyfv .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #lxxelxsyfv .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #lxxelxsyfv .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #lxxelxsyfv .gt_column_spanner_outer:first-child { padding-left: 0; }
 #lxxelxsyfv .gt_column_spanner_outer:last-child { padding-right: 0; }
 #lxxelxsyfv .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #lxxelxsyfv .gt_spanner_row { border-bottom-style: hidden; }
 #lxxelxsyfv .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #lxxelxsyfv .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #lxxelxsyfv .gt_from_md> :first-child { margin-top: 0; }
 #lxxelxsyfv .gt_from_md> :last-child { margin-bottom: 0; }
 #lxxelxsyfv .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #lxxelxsyfv .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #lxxelxsyfv .gt_indent_1 { text-indent: 5px; }
 #lxxelxsyfv .gt_indent_2 { text-indent: calc(5px * 2); }
 #lxxelxsyfv .gt_indent_3 { text-indent: calc(5px * 3); }
 #lxxelxsyfv .gt_indent_4 { text-indent: calc(5px * 4); }
 #lxxelxsyfv .gt_indent_5 { text-indent: calc(5px * 5); }
 #lxxelxsyfv .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #lxxelxsyfv .gt_row_group_first td { border-top-width: 2px; }
 #lxxelxsyfv .gt_row_group_first th { border-top-width: 2px; }
 #lxxelxsyfv .gt_striped { color: #333333; background-color: #F4F4F4; }
 #lxxelxsyfv .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #lxxelxsyfv .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #lxxelxsyfv .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #lxxelxsyfv .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #lxxelxsyfv .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #lxxelxsyfv .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #lxxelxsyfv .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #lxxelxsyfv .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #lxxelxsyfv .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #lxxelxsyfv .gt_left { text-align: left; }
 #lxxelxsyfv .gt_center { text-align: center; }
 #lxxelxsyfv .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #lxxelxsyfv .gt_font_normal { font-weight: normal; }
 #lxxelxsyfv .gt_font_bold { font-weight: bold; }
 #lxxelxsyfv .gt_font_italic { font-style: italic; }
 #lxxelxsyfv .gt_super { font-size: 65%; }
 #lxxelxsyfv .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #lxxelxsyfv .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #lxxelxsyfv .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #lxxelxsyfv .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #lxxelxsyfv .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #lxxelxsyfv .gt_asterisk { font-size: 100%; vertical-align: 0; }
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
