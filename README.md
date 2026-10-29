# RetinaConcept-Net GUI

Desktop research interface for **RetinaConcept-Net**, a concept bottleneck model for explainable diabetic retinopathy (DR) grading from colour fundus photographs. The application also supports a black-box ResNet-152 classifier for comparison.

- **Author:** Ali Golbaf
- **Application version:** 1.5
- **Purpose:** Research and demonstration only — **not** a medical device or a tool for clinical diagnosis.

## Features

- **Overview:** import a fundus photograph, load either or both models, and inspect five-grade classification scores.
- **Lesion evidence explorer:** inspect the seven predicted retinal lesion concepts and their spatial evidence heatmaps.
- **Concept bottleneck decision explorer:** inspect signed contributions to the predicted-versus-runner-up logit margin, read a case-level explanation, and explore *what-if* sensitivity by adjusting concept inputs to the trained linear grading head.
- **Checkpoint loading:** automatically load recognised checkpoints from the local `models/` folder, download missing weights from Google Drive, use **Reload models folder**, or choose a checkpoint manually with **Load…** for each model.
- **Export:** save model probabilities and concept-level outputs as JSON.

All inference runs locally. No checkpoint weights, clinical images, or datasets are distributed in this repository.

## Requirements

- Python **3.10 or newer**.
- NumPy, Pillow, PyTorch, torchvision and gdown, as specified in `requirements.txt`.
- **Tkinter**, which is normally included in Python for Windows; on some Linux distributions, install the system's `python3-tk` package separately.
- A genuine, architecture-compatible PyTorch checkpoint for each model you wish to run.

The initial download and installation of PyTorch dependencies may be substantial. A CUDA-enabled installation is optional; the app can run on CPU.

## Quick start (Windows)

Open PowerShell in the repository folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_windows.bat
```

After installation, double-click `run_windows.bat` for subsequent launches. The launcher uses `.venv` if it exists, or your system Python otherwise.

On macOS or Linux, use the corresponding virtual environment commands and launch with `python app.py`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

## Model checkpoints

Place your trained state-dictionary checkpoints in `models/` using the recommended filenames:

```text
models/
├── best_model.pt       # black-box ResNet-152
└── best_cbm_model.pt   # RetinaConcept-Net
```

The app attempts to load these automatically on startup. If they are added later, use **Reload models folder**. Alternatively, click **Load…** for a model and select a `.pt` or `.pth` checkpoint from anywhere on your computer. Manual loading also works when `models/` is empty.

### Download the trained weights

The two research checkpoints are hosted separately on Google Drive:

| Model | File | Source |
| --- | --- | --- |
| Black-box ResNet-152 | `best_model.pt` | [Google Drive](https://drive.google.com/file/d/1K8agioezWh8Tasy5fnfCtANZOHFZ3W7N/view) |
| RetinaConcept-Net | `best_cbm_model.pt` | [Google Drive](https://drive.google.com/file/d/1pc2ZFBo6eNpxS8xTbJrhxcypERiUqPtC/view) |

On the first launch without any checkpoints, the GUI offers to download them. You can also click **Download weights** at any time, or download them from a terminal:

```bash
python model_download.py
```

Downloading is optional and requires internet access. The application stores the files only in `models/`, preserves existing checkpoints, and automatically loads successfully downloaded weights. Temporary download files are removed if a transfer fails. If Google Drive limits access or downloading fails, use the browser links above and put the files in `models/` manually.

The downloader does not download files during `git clone` or `pip install`; it requires user approval in the GUI or an explicit terminal command. Model weights remain excluded from Git.

The loader recognises certain alternative filenames, but it requires an unambiguous match per model type and does not select between multiple folds automatically. Checkpoints must be compatible with the architecture in `models.py` and be state dictionaries. For reproducibility, record the SHA-256 checksums of the final weights before public release.

You can open the interface without weights, but model inference remains unavailable until at least one checkpoint loads successfully. The black-box classifier and the concept bottleneck model can be used separately.

## Repository structure

```text
retinaconcept-net-gui/
├── app.py              # Tkinter interface
├── model_runtime.py    # checkpoint discovery, inference, and explanations
├── model_download.py   # optional Google Drive model downloader
├── models.py           # PyTorch model definitions
├── requirements.txt    # Python dependencies
├── run_windows.bat     # Windows launcher
├── smoke_test.py       # tests that do not require checkpoints
├── models/             # local checkpoints; ignored by Git
├── .gitignore
└── README.md
```

The interface optionally uses the **Azure ttk theme** when `azure/azure.tcl` and its companion files are present. This optional third-party theme is **not included** in this repository. Without it, the application uses its built-in fallback theme; some controls may look different from a themed local installation.

## Check the project

Run these tests after installing NumPy:

```bash
python -m unittest smoke_test -v
```

The tests cover checkpoint discovery, offline download handling, numerical probability calculations, and the exact reconstruction of the linear classifier's logit margin. They do **not** check Google Drive connectivity, trained-model accuracy, or the full model-loading operation.

## Interpretation and limitations

- Classification scores are softmax outputs, **not calibrated clinical probabilities**.
- Spatial concept maps are model evidence activations, **not validated lesion segmentations**.
- The *what-if* controls evaluate the trained linear grading head using modified concept values; they do not rerun the image model and do not support causal or clinical intervention claims.
- Any real-world use requires independent clinical and external validation, appropriate governance, and regulatory assessment.

## Licence

Copyright © 2026 Ali Golbaf. No open-source licence has yet been specified; permission to reuse or redistribute the code is not implied by public availability. Choose and add a `LICENSE` file before inviting outside contributions or reuse.
