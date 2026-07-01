# Datasets for PIG (Paired Image Generation)

## Paper's Original Dataset — NOT Public

**DBTMassSeg** — private dataset from Suzhou Municipal Hospital & Guizhou Provincial People's Hospital.
- 367 patients, 8,723 annotated DBT slices
- CC and MLO views with mass lesion masks, annotated by 2 radiologists
- Ethics approval No. 2024320 & 2024328
- Not publicly released by the authors

---

## Small Public Datasets (< 500 GB) — Recommended

### 1. In Silico DBT Dataset — Zenodo (SMALLEST, ~few MB)
- https://zenodo.org/records/14270329
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13103193/
- https://link.springer.com/article/10.1007/s10278-025-01626-z
- 230 2D ROIs derived from FDA-cleared DBT software
- Includes pixel-level tumor segmentation masks
- Covers a range of breast densities and tumor complexities
- **Note:** Very small — useful for pipeline testing, not full training

### 2. T-SYNTH — HuggingFace / FDA DIDSR (SYNTHETIC DBT, manageable size)
- https://huggingface.co/datasets/didsr/tsynth
- https://github.com/DIDSR/tsynth-release
- https://arxiv.org/abs/2507.04038
- 9,000 synthetic DBT samples across 4 breast density categories
- Includes pixel-level segmentation masks and bounding boxes
- Generated via Monte Carlo x-ray simulation (VICTRE toolkit)
- Paired 2D mammography + 3D DBT images
- Freely available, no registration required
- **Best option for getting started without data access barriers**

### 3. DBT-2026 — De-identified Public Dataset (558 exams, small)
- https://www.medrxiv.org/content/10.64898/2026.03.03.25337924v1
- https://www.medrxiv.org/content/10.64898/2026.03.03.25337924v1.full.pdf
- 558 DBT exams from 558 patients with BI-RADS 0/1/2 scores
- Expert annotations + free-text radiology reports, fully de-identified
- Free for non-commercial research
- **Note:** Annotations are radiologist reports, not pixel masks — need SAM/GrabCut to generate masks

### 4. CBIS-DDSM — Kaggle / TCIA (~163 GB, has pixel masks)
- https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
- https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM
- 10,239 images from 6,671 subjects — 163.6 GB total, ~10 GB on Kaggle (compressed)
- Has pixel-level ROI segmentation masks + bounding boxes + BI-RADS scores
- **IMPORTANT:** This is 2D digital mammography, NOT 3D DBT — architecture is different
- Still usable to test the PIG pipeline on 2D slices before getting real DBT data

### 5. VinDr-Mammo — PhysioNet
- https://vindr.ai/datasets/mammo
- https://physionet.org/content/vindr-mammo/1.0.0/
- https://arxiv.org/abs/2203.11205
- 20,000 images (5,000 exams × 4 views), bounding box + BI-RADS annotations
- **IMPORTANT:** 2D mammography, NOT DBT — same caveat as CBIS-DDSM
- Free with PhysioNet credentialed access

---

## BCS-DBT Subsets & Selective Download

### Does a public subset exist anywhere?
**No.** Nobody has shared a preprocessed subset of BCS-DBT on HuggingFace, Kaggle, or Zenodo.
The search found no third-party redistributions of slices or patches from BCS-DBT.

### What does exist from the Mazurowski Lab (original authors):
- **duke-dbt-data** (official utils repo): https://github.com/mazurowski-lab/duke-dbt-data
- **DBTex-baseline** (preprocessing notebooks, DICOM → PNG slices): https://github.com/mazurowski-lab/DBTex-baseline
- **DBT-cancer-detection-algorithms** (challenge entries): https://github.com/mazurowski-lab/DBT-cancer-detection-algorithms
- **DBTex Challenge page** (official splits info): https://www.aapm.org/GrandChallenge/DBTex/
- **Community discussion (Reddit):** https://www.reddit.com/r/DukeDBTData/

### Official TCIA download splits (separate downloads, smaller than full dataset):
- **Training boxes only** (annotations, tiny): https://www.cancerimagingarchive.net/tcia-version-dnlds/breast-cancer-screening-dbt-ver1-other-1/bcs-dbt-boxes-train/
- **Challenge validation set**: https://www.cancerimagingarchive.net/tcia-version-dnlds/breast-cancer-screening-dbt-ver2-rad/bcs-dbt-challenge-validation/

### How to download only a subset from TCIA (officially supported):

**Method 1 — NBIA Data Retriever (GUI, easiest):**
1. Go to https://www.cancerimagingarchive.net/collection/breast-cancer-screening-dbt/
2. Click "Search" → browse by patient/series
3. Add only the patients you want to your cart
4. Download a `.tcia` manifest file for just those patients
5. Open manifest in NBIA Data Retriever app → downloads only selected series
- NBIA Data Retriever guide: https://wiki.cancerimagingarchive.net/display/NBIA/8.1+Downloading+Images+Using+the+NBIA+Data+Retriever

**Method 2 — tcia-utils Python package (programmatic, most flexible):**
```bash
pip install tcia-utils
```
```python
from tcia_utils import nbia
# List all series in the collection
series = nbia.getSeries(collection="Breast-Cancer-Screening-DBT")
# Download only N series of your choice
nbia.downloadSeries(series[:50])  # download first 50 series only
```
- tcia-utils on PyPI: https://pypi.org/project/tcia-utils/
- Official notebooks: https://github.com/kirbyju/TCIA_Notebooks/blob/main/TCIA_REST_API_Downloads.ipynb
- TCIA REST API docs: https://wiki.cancerimagingarchive.net/x/NIIiAQ

**Method 3 — Download annotations only first (near-zero size):**
The bounding box CSV/Excel files are a few MB. Download those first,
pick the patient IDs you want (e.g. cancer cases only), then use
tcia-utils to pull only those specific patients' DICOM volumes.

---

## Large Datasets (>500 GB) — Listed for Reference Only

### BCS-DBT — Cancer Imaging Archive (~1.75 TB full, selectively downloadable)
- https://www.cancerimagingarchive.net/collection/breast-cancer-screening-dbt/
- https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=64685580
- https://arxiv.org/abs/2011.07995
- 22,032 DBT volumes from 5,060 patients — bounding boxes only, no pixel masks
- True DBT; download selectively using tcia-utils (see section above)

### Duke DBT Database (Mazurowski Lab) — size unknown, likely large
- https://sites.duke.edu/mazurowski/resources/digital-breast-tomosynthesis-database/
- Referenced in PIG paper (ref [9] — Konz et al.); has segmentation masks

---

## Recommended Path for Getting Started

1. **Prototype/test pipeline:** Use **In Silico DBT (Zenodo)** or **T-SYNTH (HuggingFace)** — no barriers, small size
2. **Proper DBT training:** Use **T-SYNTH** (synthetic but true DBT structure with masks)
3. **Real annotated DBT data (selective):** Register on TCIA → download annotations CSV → pick ~100–200 cancer patients → use `tcia-utils` to pull only those volumes from BCS-DBT

---

## Related Papers & Resources

- PIG paper (arXiv): https://arxiv.org/pdf/2507.14833
- PIG paper (MICCAI proceedings): https://papers.miccai.org/miccai-2025/paper/4386_paper.pdf
- In Silico DBT paper (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC13103193/
- Mass detection with 3D Mask RCNN (PMC): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7686533/
- DBT mass segmentation dataset paper (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC8369362/
