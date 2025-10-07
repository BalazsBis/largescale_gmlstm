<!-- README.md -->
<h1 align="center">GM‑LSTM for Spatiotemporal Forecasting (K‑Fold CV)</h1>

<p align="center">
  Probabilistic time-series forecasting with an <b>LSTM</b> backbone and a <b>Mixture Density Network (MDN)</b> head.<br/>
  Includes <b>K-fold</b> cross-validation by location, train-only normalization, AMP training, and CSV outputs per fold and per location.
</p>

<hr/>

## Features

- <b>BaseDataset</b>: loads per-location CSV time series + a static attributes table, builds rolling sequences, handles NaNs.
- <b>Normalization</b>: global train mean/std for dynamic/static (and optionally target); reused for val/test.
- <b>MDN-LSTM</b>: predicts Gaussian mixture parameters (π, μ, σ) for each step; temperature scaling for π.
- <b>Training loop</b>: AMP + grad clipping + optional <code>torch.compile</code>; forget-gate bias init for LSTM stability.
- <b>K-fold CV by location</b>: prevents spatial leakage; val batches are grouped per location.
- <b>Outputs</b>: per-fold checkpoints, progress logs, and CSVs with predictions & MDN params per location.

## Data requirements

### 1) Time-series files
- <b>Location</b>: <code>data/time_series/&lt;entity_id&gt;.csv</code>
- <b>Required columns</b>:
  - <code>date</code> (parseable datetime)
  - every name listed in <code>dynamic_input</code>
  - every name listed in <code>target</code>
- <b>Example <code>entity_id</code></b>: <code>Abrams_0.05</code> (prefix = location name)

### 2) Static attributes table
- <b>Path</b>: <code>data/static_attributes.csv</code>
- Must have an index column named <code>location_id</code> that matches the location prefix (e.g., <code>Abrams</code>)
- Contains all names listed in <code>static_input</code> (e.g., <code>lon</code>, <code>lat</code>, soil class, etc.)

### 3) Entity IDs
Either:
- A list passed directly to <code>BaseDataset(path_entities_ids=ids_list)</code>, or
- A text file (one ID per line) read by <code>_load_entity_ids()</code>

## Outputs

<b>Per fold</b> (e.g., <code>path_save_folder/fold_1/</code>):
- <code>epoch_&lt;E&gt;.pt</code> — model checkpoints.
- <code>run_progress.txt</code> — per-epoch logs: <code>Fold k | Epoch E | Loss ... | LR ... | Time ...</code>
- <code>results/total_locations_fold_&lt;k&gt;.csv</code> — concatenated predictions (columns may include):
  - <code>date</code>, <code>soil_moisture</code>, optional dynamic features (unscaled if requested),
