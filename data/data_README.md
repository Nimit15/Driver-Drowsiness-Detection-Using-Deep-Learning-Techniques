# Data

This project uses the [Drowsiness Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset) by dheerajperumandla on Kaggle. The raw images aren't stored in this repo — download them yourself with the steps below.

## Setting it up

1. Download the dataset from the Kaggle link above (or via the Kaggle API — `kaggle datasets download -d dheerajperumandla/drowsiness-dataset`)
2. Unzip it so the folder structure looks like this:

```
data/
└── train/
    ├── Closed/     726 images — cropped eye, eye closed
    ├── Open/       726 images — cropped eye, eye open
    ├── yawn/       723 images — full driver face, mid-yawn
    └── no_yawn/    725 images — full driver face, neutral
```

The `train/` subfolder is intentional — that's how the dataset ships from Kaggle, and every notebook in this repo expects the path `data/train/<class_name>/`.

## Two visual domains, not one

`Closed`/`Open` are close-up eye crops. `yawn`/`no_yawn` are full driver-face photos taken inside a car. These look nothing alike, which is why this project trains two separate models rather than one 4-class classifier — see the [project report](../docs/PROJECT_REPORT.md) for the full story of how we found this out.

## `label_map.json`

Maps class folder names to the integer labels used internally:

```json
{
  "Closed": 0,
  "Open": 1
}
```

(and separately for the yawn/no_yawn pair — each model has its own label map since they're trained independently.)

## Processed arrays

The notebooks also generate `.npy` files (`X_train.npy`, `y_train.npy`, etc.) as a preprocessing cache so you don't have to reload and resize every image on every run. These are excluded from version control via `.gitignore` since they're regenerable and large — running the Week 2/5 notebooks will recreate them locally.
