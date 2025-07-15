
# RotNet: Rotation-Aware Self-Supervised Learning

This project contains a complete pipeline for training and evaluating **RotNet**—a self-supervised model that learns image representations by predicting the rotation angle (0°, 90°, 180°, 270°) applied to an input image.
The repository supports two datasets:

1. **CIFAR-10** (two-stage training: rotation pre-training → downstream classification fine-tuning)
2. **Your own images** (single-stage rotation pre-training with ReLU or Mish activations)

---

## Quick Start

### 1. Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 2. Data

| Dataset      | Location                  | Notes |
|--------------|---------------------------|-------|
| CIFAR-10     | `data/cifar-10-batches-py/` | Auto-downloads on first run |
| Your images (train) | `data/data/`         | Put your training `.jpg/.png` files here |
| Your images (test)  | `data/test_images/`  | Put your test images here |

---

## Training

### A. CIFAR-10 (Two-Stage)

#### Stage 1 – Rotation Pre-training
```bash
cd train/cifar-10-train
python basic_model.py          # trains from scratch
```
Outputs:  
- `model/cifar10_rotnet_encoder.pth` (frozen encoder weights)

#### Stage 2 – Downstream Classification
```bash
python stage2.py
```
Prompt: choose whether to load the encoder weights.  
Outputs:  
- Best downstream classifier checkpoint (saved automatically).

### B. Your Own Dataset (Single-Stage)

Choose an activation function:

| Activation | Script |
|------------|--------|
| ReLU       | `train/mydataset/relu/resnet18-relu.py` |
| Mish       | `train/mydataset/mish/resnet18-mish.py` |

```bash
cd train/mydataset/<relu|mish>
python resnet18-<relu|mish>.py
```
Outputs:  
- `model/mydata_rotnet_<relu|mish>.pth` (full model)  
- `model/mydata_rotnet_<relu|mish>_encoder.pth` (encoder only)

---

## Prediction / Evaluation

Place test images in `data/test_images`.

| Model Used | Command |
|------------|---------|
| ReLU       | `python predict/predict_on_my_testset_relu.py` |
| Mish       | `python predict/predict_on_my_testset_mish.py` |

Each script:
- tests every image at all four rotations,
- prints per-angle accuracy,
- saves side-by-side comparison images to `results_enhanced`.

---

## Utilities

| Script | Purpose |
|--------|---------|
| `script/rename_images.py` | Rename your own images to `image_001.jpg`, `image_002.jpg`, … |
| `script/check_duplicates.py` | Detect near-duplicates between training and test folders |
| `classify_images_to_csv.py` | (Optional) Auto-label your images with Gemini AI |

---

## File Tree (key paths)

```
RotNet/
├── data/
│   ├── cifar-10-batches-py/   # CIFAR-10
│   └── image/                 # Your images
├── model/
│   ├── cifar10_rotnet_encoder.pth
│   ├── mydata_rotnet_relu.pth
│   └── mydata_rotnet_mish.pth
├── script/
│   ├── rename_images.py
│   └── check_duplicates.py
├── train/
│   ├── cifar-10-train/
│   └── mydataset/
├── predict/
├── results_enhanced/          # Visual results
└── requirements.txt
```

---

## Citation

If you use this code, please cite the original RotNet paper:

> Gidaris, Spyros, et al. “Unsupervised Representation Learning by Predicting Image Rotations.” *ICLR 2018*.