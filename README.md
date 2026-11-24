# OMEN

This is the **PyTorch implementation** of [OMEN: Open-Modality Region Representation Learning via Anchor-Relative Observations](#) (KDD 2027).

OMEN is a region representation encoder for open-modality learning: it is pretrained with an initial set of sensing modalities (e.g. satellite and street-view imagery) and, once deployed, can incorporate emerging modalities (e.g. POI and map data) that were never seen during pretraining — without retraining or adding any modality-specific parameters. It does so by converting every modality into an **anchor-relative profile** over the same fixed anchor regions, and combining the currently available profiles with a **content-adaptive, modality-agnostic aggregator**.

## Repository Structure

```
OMEN/
├── data/
│   ├── london/
│   │   └── labels.csv
│   └── nyc/
│       └── labels.csv
├── encode_emb_london/
│   ├── emb_osm.pkl
│   ├── emb_poi.pkl
│   ├── emb_satellite.pkl
│   └── emb_street.pkl
├── encode_emb_nyc/
│   ├── emb_osm.pkl
│   ├── emb_poi.pkl
│   ├── emb_satellite.pkl
│   └── emb_street.pkl
├── OMEN.py        # model architecture, pretraining loop, downstream probe
├── run.py         # experiment entry point
└── README.md
```

## Datasets

### Region systems and downstream labels

We evaluate OMEN on two urban region systems: **NYC** (2,324 census tracts, indexed by `GEOID`) and **London** (1,000 MSOAs, indexed by `MSOA21CD`). Each city has 15 downstream prediction tasks spanning socioeconomic, health, environmental, safety, and mobility indicators (education, income, unemployment, population density, public transport, crime rate, etc.). The full task list and original data sources (ACS, UK Census, WorldPop, Sentinel-2, GHS-BUILT-H, NYC OpenData, UK Police, ONS, IMD, OHID Fingertips) are listed in the paper's Appendix B.

Labels for each city are provided as `data/{city}/labels.csv`, with one row per region. The CSV must contain:
- an ID column matching `label_key` (`GEOID` for NYC, `MSOA21CD` for London)
- one column per downstream task name (matching the `task_names` list configured in `run.py`)

### Modality embeddings

OMEN operates on **frozen, pretrained embeddings** for four modalities — satellite imagery, street-view imagery, POI text, and OSM map tiles — produced by external, off-the-shelf encoders (these encoders are not part of OMEN and are never fine-tuned):

| Modality | External encoder | Notes |
|---|---|---|
| Satellite | AlphaEarth Foundations | 100 m grid embeddings, mean-pooled per region |
| Street-view | CLIP ViT-L/14 ([CLS] token) | up to 10 Google Street View images per region, averaged |
| POI | Qwen3-Embedding-8B | templated text summary of each region's POI category/brand statistics |
| OSM tiles | DINOv2 ViT-B/14 ([CLS] token) | zoom-16 CartoDB Voyager basemap tiles, averaged per region |

Full preprocessing details are in the paper's Appendix A.1. Each modality's embeddings must be saved as a pickled Python dict at `encode_emb_{city}/emb_{modality}.pkl`, where:
- keys are region IDs (matching the ID column in `labels.csv`)
- values are either a 1D `np.float32` array (one pooled embedding per region) or a 2D array (multiple raw embeddings per region, e.g. per image/tile — these are mean-pooled automatically when loaded)

`emb_satellite.pkl`, `emb_street.pkl`, `emb_poi.pkl`, and `emb_osm.pkl` correspond to the four modalities above; the dictionary keys must match across all four files for a given city.

## Method

`OMEN.py` implements the full pipeline described in the paper:
- **Anchor-Relative Modality Recoding** (Section 3.2) — selects a fixed set of anchor regions via Farthest Point Sampling and recodes every modality into a profile of relations to these anchors.
- **Modality-Agnostic Evidence Aggregation** (Section 3.3) — a shared, content-adaptive scorer combines the currently available profiles, followed by a region-graph-based context refinement step.
- **Composition-Robust Pretraining** (Section 3.4) — a VICReg-style self-supervised objective trained over randomly sampled modality-composition views of the initial modalities.
- **Frozen-Encoder Modality Extension** (Section 3.5) — after pretraining, emerging modalities are appended as additional anchor-relative profiles and processed by the same frozen aggregator, with no retraining.

Downstream evaluation trains a Ridge regression probe (5-fold cross-validation) on the frozen region representations, both before (`OMEN (Initial)`) and after (`OMEN (Extended)`) emerging modalities are incorporated.

## Running Steps

All hyperparameters (city, initial vs. emerging modality split, number of anchors, training schedule, etc.) are set directly at the bottom of `run.py`. Edit the values there before running.

1. Place the modality embeddings and labels as described above under `data/{city}/` and `encode_emb_{city}/`.
2. Open `run.py` and set:
   - `CITY` — `"nyc"` or `"london"`
   - `train_modalities` — the initial modality set `M_init` used for pretraining
   - `test_modalities` — `M_init` plus the emerging modalities to evaluate after deployment
   - any other hyperparameters (`n_anchors`, `ssl_epochs`, `lr`, etc.)
3. Run:
   ```
   python run.py
   ```

Training logs (loss curves) are written to `log_dir`, and the checkpoint is saved to `checkpoint_dir` (both configurable in `run.py`). At the end of the run, the downstream Ridge probe results — MAE, RMSE, and R² for every task, comparing `OMEN (Initial)` against `OMEN (Extended)` — are printed and logged.
