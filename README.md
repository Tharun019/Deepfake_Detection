# DeepSentinel

**Intelligent Metadata-Driven Multimodal Deepfake Detection: A Three-Layer Forensic Framework**

Final-year project, Department of AI & ML, Global Academy of Technology, Bengaluru.
Team: Tharun, Niyathi Nagesh · Supervisor: Dr. Roopa B S

A Flask web app that analyses an image, video or audio file with three independent
forensic layers and fuses them into one verdict, with a per-layer explanation.

## How it works

| Layer | Weight | What it does |
|---|---|---|
| L1 — Metadata forensics | 0.30 | Rule-based EXIF / container checks (`exifread`, `ffprobe`) |
| L2 — Content analysis | 0.45 | Image: EfficientNet-B4 · Video: VideoMAE-base (top 4 blocks fine-tuned) + MLP head · Audio: Wav2Vec2 (fine-tuned) |
| L3 — Binary file structure | 0.25 | Shannon entropy, byte-frequency uniformity, JPEG quantization tables |

Fused score = `L1×0.30 + L2×0.45 + L3×0.25`; **≥ 0.35 → DEEPFAKE DETECTED**, else LIKELY AUTHENTIC.
A layer switched off in the UI contributes a neutral 0.5.

Explainability: Grad-CAM heatmap for images (hand-written, on EfficientNet-B4 `conv_head`);
per-signal Feature Contribution Breakdown for L1 and L3; raw model output for L2
(including face-crop status and, on failure, the error message).

## Measured results (L2 models)

| Modality | Training data | Result |
|---|---|---|
| Image | rvf10k (FFHQ real vs StyleGAN fake faces) | ~87.9% validation accuracy |
| Video | FaceForensics++ c23, identity-disjoint split | 78.7% balanced accuracy on 900 held-out test clips (real 75.3%, fake 82.0%, AUC 0.884); frozen-backbone baseline 67.4% |
| Audio | ASVspoof 2019 LA, then In-the-Wild (speaker-disjoint) | 93.8% balanced accuracy on 7 unseen speakers (AUC 0.991, EER 5.1%) |

## Known limitations

- Image: not trained on diffusion-generated images; false positives possible on real photos unlike rvf10k.
- Video: trained on single frontal faces with a static camera; multi-person / moving-camera phone video is out of scope (observed false positive).
- Audio: modern neural TTS / voice cloning is not detected (MLAAD validation: 46.4% balanced accuracy, chance level).
- Cross-modal attacks (AI video + real audio and vice versa) and single face-swaps in multi-person video are not handled.
- L1 video/audio score is the average of red-flag weights, so an extra weaker flag can lower it.
- L3 treats HEIC as a camera container, but macOS screenshots can also be HEIC.
- The wrong-tab upload check uses the file extension only.

## Requirements

- Python 3.13 (developed and tested on 3.13.9, macOS on Apple Silicon)
- `ffprobe` from FFmpeg: `brew install ffmpeg`
- Model weights (not in git, see below)

## Setup

```zsh
git clone https://github.com/Tharun019/Deepfake_Detection.git
cd Deepfake_Detection
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Model weights

Weights are not committed (`models/` and `checkpoints/` are gitignored) and are not
publicly distributed until dataset licence terms are checked. The app loads:

| Model | Path |
|---|---|
| Image | `models/image/efficientnet_b4.pth` |
| Video | `models/video/vmae_ft_v1/best/` (`videomae-base-ft/`, `video_head_ft.pth`, `feature_norm_stats_facecrop.npz`) |
| Audio | `checkpoints/wav2vec2_itw_ft_v1/epoch_3/` |

Restore from the private backup zip (run inside the project folder, then verify checksums):

```zsh
unzip /path/to/DeepSentinel_models_2026-10-12.zip
shasum -a 256 -c SHA256SUMS_models.txt
```

Every line must print `OK`. The image loader uses `strict=True` and fails loudly if weights are missing.

## Run

```zsh
source venv/bin/activate
python run.py
```

Open http://127.0.0.1:8080 in **Safari** (Chrome returned HTTP 403 on the development machine; cause not investigated).
The server runs in Flask debug mode, bound to localhost only; it auto-restarts when a source file changes.

Supported formats: image JPG/PNG/WEBP/HEIC · video MP4/MOV/AVI/M4V · audio WAV/MP3/FLAC/M4A/AAC/OGG.
A file uploaded on the wrong tab is rejected with a message naming the right tab.

## Training and evaluation

Extra packages: `pip install -r requirements-train.txt`.

- `train_image_model.py` — image model (rvf10k)
- `train_itw_finetune.py` — audio fine-tune on In-the-Wild
- `vmae_ft_cell.py`, `notebooks/video_ft_v1_kaggle.ipynb` — video fine-tune (Kaggle)
- `notebooks/video_head_frozen_kaggle.ipynb` — frozen-backbone video baseline
- `eval_video_ft.py --split val|test` — video evaluation on the held-out split
