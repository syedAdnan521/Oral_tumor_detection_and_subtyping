**AI Tumour Detection (Multi-stage MIL + Uncertainty)**

This repository contains code  that implements a multi-stage Multiple Instance Learning (MIL) pipeline to detect and classify oral tumour slides (OSCC and others). The project includes training/evaluation scripts, uncertainty estimation via Monte Carlo Dropout, visualization utilities, and a Streamlit app for quick inference/visualization.

**Highlights**
- **Modeling:** Two-stage MIL with Top-K attention and MC Dropout for uncertainty.
- **Data:** Organized `train/`, `val/`, `test/` splits with class subfolders (e.g., `mdoscc`, `normal`, `osmf`, `pdoscc`, `wdoscc`).
- **Notebooks:** Uncertainty analysis (e.g., `notebooks/07_uncertainty.ipynb`).
- **App:** `app/streamlit_app.py` for local inference/visualization.

**Repository Structure**
- `app/` : Streamlit demo app (`streamlit_app.py`).
- `data/` : Dataset splits (`train/`, `val/`, `test/`) with class subfolders.
- `outputs/` : Trained models, metrics, heatmaps, figures, and logs.
- `notebooks/` : Analysis and visualization notebooks (e.g., uncertainty analysis).
- `src/` : Core code (config, models, data loading, training, evaluation).
- `requirements.txt` : Python dependencies.

**Quick Setup**
1. Create a Python 3.8+ virtual environment (Python 3.11 recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. (Optional) If you use a GPU, create and use the appropriate CUDA-enabled environment and install matching PyTorch build.

3. Ensure the dataset is organized under the `data/` directory as in the repo tree. If you have precomputed embeddings, place them in an `embeddings/` folder (not included in repo):

   - `embeddings/train/<class>/<slide_id>.npy`
   - `embeddings/test/<class>/<slide_id>.npy`

**Configuration**
- Global configuration values are in `src/config/config.py`. Edit paths or hyperparameters there if needed.

**Training**
- Stage 1 training script: `src/train/train_stage1.py`
- Stage 2 training script: `src/train/train_stage2.py`

Example command (adjust arguments in scripts or `config.py` as needed):

```bash
python src/train/train_stage1.py --config src/config/config.py
python src/train/train_stage2.py --config src/config/config.py
```

Training outputs are saved to `outputs/models/` and logs to `outputs/metrics/`.

**Evaluation & Uncertainty**
- Evaluation and calibration scripts live in `src/eval/` (e.g., `evaluate.py`, `calibration.py`, `conformal.py`).
- The notebook `notebooks/07_uncertainty.ipynb` demonstrates MC Dropout-based uncertainty estimation and produces plots saved to `results/` (or `outputs/figures/`).

Example to run the uncertainty notebook (from project root):

```bash
jupyter notebook notebooks/07_uncertainty.ipynb
```

Or convert/run programmatically with `nbconvert` / `papermill` if you prefer automation.

**Inference / Streamlit App**
- Quick visual inference and exploration: run the Streamlit app in `app/`.

Run the app locally:

```bash
cd app
streamlit run streamlit_app.py
```

The app expects model weights and embeddings/tiles accessible via paths configured in `src/config/config.py` or environment variables used in the app. Edit the app or config to point to `outputs/models/stage2_best.pt` (or the relevant checkpoint).

**Outputs & Artifacts**
- `outputs/models/` : Saved model checkpoints (e.g., `stage1_best.pt`, `stage2_best.pt`).
- `outputs/metrics/` : Training logs and CSVs (e.g., `stage1_train_log.csv`).
- `outputs/figures/` and `outputs/heatmaps/` : Visualizations and heatmaps.
- `outputs/manifest.csv` : Data manifest used by some preprocessing utilities.

**Preprocessing**
- Tile extraction, stain normalization, and embedding extraction are expected to be performed before training. Scripts/utilities are in `src/preprocess/` and `src/data/`.

**Reproducibility Tips**
- Fix random seeds in training scripts for reproducibility.
- Use the same device configuration (MPS/CPU/CUDA) as used for model checkpoints when evaluating.

**Common Troubleshooting**
- Missing embeddings: create or point `EMB_ROOT` to your embeddings folder.
- Device errors on macOS: if running on MPS, ensure `torch` build supports MPS; otherwise use CPU or CUDA-enabled environment.

**Contact & Credit**
- Author / Student: syedAdnan521 (local repo owner). For questions, update issues in the repo or message the project owner.

**License**
- This project is academic course work. Add a license file if you intend to share it externally.

---

If you'd like, I can now: generate a slide-by-slide PPT outline, produce speaker notes, or create an actual PowerPoint file with the slides. Tell me which you'd prefer.
