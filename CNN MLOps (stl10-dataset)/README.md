# STL10 Multiclass Classification (Small CNN)

This project trains a simple CNN on the **STL10** dataset (10 classes) and compares two input resolutions:
- `img_size=64`
- `img_size=96`

The goal is to see how **image resolution affects accuracy and errors** (confusion matrix), while keeping the pipeline simple and reproducible.

---

## 1) Repo structure

Create this folder layout:

```
my-stl10-cnn/
  notebooks/
    01_explore_data.ipynb
  src/
    model.py
    train.py
    predict.py
  outputs/
    run_stl10/
      img_size=64/
      img_size=96/
  data/              # auto-created by torchvision download
  requirements.txt
  README.md
```

Notes:
- `data/` is where STL10 is downloaded automatically.
- `outputs/` stores metrics, plots, and checkpoints.

---

## 2) Install dependencies

Create/activate a virtual environment (recommended), then run:

```bash
pip install -r requirements.txt
```

---

## 3) Explore the dataset (notebook)

Open:

- `notebooks/01_explore_data.ipynb`

What to check:
- image tensor shape (should be `[3, 96, 96]` before resizing)
- class names
- sample image grid
- class counts

This is just to understand the dataset before training.

---

## 4) Train model at 64×64

From the repo root:

```bash
python src/train.py --img_size 64 --epochs 60 --batch_size 128 --early_stop --patience 6
```

Outputs saved to:

```
outputs/run_stl10/img_size=64/
  model.pt
  metrics.json
  config.json
  training_curve.png
  confusion_matrix.png
```

---

## 5) Train model at 96×96

From the repo root:

```bash
python src/train.py --img_size 96 --epochs 60 --batch_size 64 --early_stop --patience 6
```

Outputs saved to:

```
outputs/run_stl10/img_size=96/
  model.pt
  metrics.json
  config.json
  training_curve.png
  confusion_matrix.png
```

---

## 6) Run inference examples (predict script)

This saves a grid of example predictions (image + true/pred labels).

### For 64×64 checkpoint
```bash
python src/predict.py --run_dir outputs/run_stl10/img_size=64 --img_size 64
```

### For 96×96 checkpoint
```bash
python src/predict.py --run_dir outputs/run_stl10/img_size=96 --img_size 96
```

---

## 7) Compare runs (what to look at)

Look inside:

- `outputs/run_stl10/img_size=64/`
- `outputs/run_stl10/img_size=96/`

Compare:
- `metrics.json` → final test accuracy
- `confusion_matrix.png` → which classes are confused
- `pred_examples.png` → qualitative differences
- `training_curve.png` → training stability and progress

---

# Model card (simple)

## Task
Multiclass image classification on STL10 (10 classes).

## Data
- Dataset: STL10
- Input: RGB images resized to **64×64** and **96×96**
- Output: one of 10 classes:
  `airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck`

## Model
Small CNN:
- 4× (Conv → ReLU → MaxPool)
- AdaptiveAvgPool2d → Linear layer

## Training setup
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Metric: test accuracy
- Augmentation: RandomHorizontalFlip (train only)


Artifacts per run:
- `training_curve.png`
- `confusion_matrix.png`
- `pred_examples.png`
- `metrics.json`
- `config.json`
- `model.pt`




## For Docker
- Train img_size 64
```bash
docker run --rm -it -v "${PWD}/outputs:/app/outputs" -v "${PWD}/data:/app/data" stl10-cnn python src/train.py --img_size 64 --epochs 60 --batch_size 128 --early_stop --patience 6
```


- Train img_size 96
```bash
docker run --rm -it -v "${PWD}/outputs:/app/outputs" -v "${PWD}/data:/app/data" stl10-cnn python src/train.py --img_size 96 --epochs 60 --batch_size 64 --early_stop --patience 6
```


- Predict img_size 64
```bash
docker run --rm -it -v "${PWD}/outputs:/app/outputs" -v "${PWD}/data:/app/data" stl10-cnn python src/predict.py --run_dir outputs/run_stl10/img_size=64 --img_size 64
```

- Predict img_size 96
```bash
docker run --rm -it -v "${PWD}/outputs:/app/outputs" -v "${PWD}/data:/app/data" stl10-cnn python src/predict.py --run_dir outputs/run_stl10/img_size=96 --img_size 96
```

