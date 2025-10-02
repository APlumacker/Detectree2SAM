# 🌳 Detectree2SAM — Reproducibility Package

This repository contains **all the scripts and resources** used for the article *Monitoring plot-level tropical forest canopy structure with automated crown segmentation from low-cost drone imagery *.  
It enables full **reproducibility** of the study, from automatic tree crown segmentation with **Detectree2SAM** to the statistical analyses and figures presented in the paper.

---

## 📂 Repository structure
- **`detectree2SAM/`** — Python pipeline, Dockerfile and dependencies to run Detectree2SAM and SAM.  
- **`analysis/`** — RMarkdown script reproducing the statistical analyses and figures.  
- Raw data and pre-trained models are hosted externally (see below).

---

## 📥 Data & pre-trained models

The raw data used in the study (orthophotos, field inventory, segmentation results, and pre-trained models for Detectree2 and SAM) are hosted here:

➡️ [**Download data & models (DOX ULiège)**](https://dox.uliege.be/index.php/s/gDN6S28iYSDgGM8)

Contents of the shared folder:
- **Field data** (inventory, shapefiles, orthophotos used for validation).
- Pre-trained **Detectree2** and **SAM** models.
- Intermediate segmentation results (to reproduce the R analyses directly if needed).
- GDAL wheel used in requirement.txt

## 🧾 License
- **Code**: MIT License
- **Data**: Creative Commons Attribution 4.0 International
