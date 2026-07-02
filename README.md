# Driver Drowsiness Detection Using Deep Learning Techniques

![Python](https://img.shields.io/badge/Python-3.10-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)  ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A real-time system that watches a driver through a webcam and estimates how drowsy they are — not a simple yes/no label, but a continuous score built from two deep learning models and three classical facial-geometry signals working together.

NTCC Project | B.Tech CSE | Amity School of Engineering & Technology | 2026

**Team**
- Nimit Sharma — Enrollment A2305224203 | Roll No. 4203
- Ayush Rawat — Enrollment A2305224213 | Roll No. 4213

[Full project report](docs/PROJECT_REPORT.md) · [Literature review](docs/literature_review.md)

---

## The problem we ran into (and why the architecture looks the way it does)

We started out assuming one model could classify all four dataset categories together. That was wrong. The dataset actually contains two visually unrelated kinds of images — close-up eye crops, and full driver-face photos taken inside a car — and a single model trained across both got stuck near random chance (33–45% accuracy). Once we diagnosed that, we split it into two independent binary classifiers, one per visual domain, and accuracy jumped to 99%+ and 90%+ respectively.

We go into more detail on how we found this and fixed it in the [full project report](docs/PROJECT_REPORT.md).

---

## Results

| Model | Task | Test Accuracy |
|-------|------|---------------|
| `eye_model.h5` | Closed vs Open eye | **99.08%** |
| `face_model.h5` | Yawn vs No Yawn | **90.37%** |

Both are MobileNetV2 (ImageNet pretrained), fine-tuned in two phases — frozen base first, then the deepest 30–50 layers unfrozen at a reduced learning rate. Full confusion matrices, training curves, and classification reports are in [`results/`](results/).

---

## System pipeline

```
Webcam → MediaPipe Face Mesh (468 landmarks)
       → Eye crop  → eye_model.h5  → Closed/Open probability
       → Face crop → face_model.h5 → Yawn probability
       → EAR + MAR + head tilt (classical geometry)
       → Weighted drowsiness score (0.0 – 1.0)
       → AWAKE / ALERT / DROWSY + live alert
```

We combine five signals into one score instead of a plain awake/drowsy label — 30% eye model, 30% yawn model, 20% EAR, 10% MAR, 10% head tilt. Most of the papers we read only give a binary output (see the [literature review](docs/literature_review.md)).

---

## Real-time system

Runs on a normal laptop CPU — no GPU needed once the models are trained. Getting from a Colab-trained model to a working local application meant solving two environment issues along the way:

- **MediaPipe's 0.10.x API restructuring** — the `mp.solutions.face_mesh` path used in most tutorials changed; the pipeline was updated to the current API.
- **TensorFlow/Keras version mismatch between Colab and the local machine** — models trained on Colab wouldn't load locally because Keras 3 didn't recognise a `quantization_config` field in the older saved format. Fixed by rewriting the model's stored metadata directly inside the `.h5` file before loading — that held up better than just trying to match library versions across the two machines.

Full implementation in [`src/realtime_detection.py`](src/realtime_detection.py).

---

## Dataset

[dheerajperumandla/drowsiness-dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset) on Kaggle.

| Class | Images | Type |
|-------|--------|------|
| Closed | 726 | Cropped eye images |
| Open | 726 | Cropped eye images |
| yawn | 723 | Full-face driver images |
| no_yawn | 725 | Full-face driver images |

Setup instructions in [`data/data_README.md`](data/data_README.md).

---

## Tech stack

Python · TensorFlow / Keras · OpenCV · MediaPipe · MobileNetV2 · scikit-learn · NumPy · Pandas · Matplotlib

---

## Project structure

```
├── notebooks/    Weekly Colab notebooks — EDA, preprocessing, training, tuning
├── src/
│   ├── ear_mar_utils.py        EAR, MAR, drowsiness score functions
│   └── realtime_detection.py   Full webcam pipeline
├── models/        Trained model weights + architecture notes
├── data/          Dataset info + label map
├── docs/          Literature review, synopsis, full project report
├── results/       Confusion matrices, training curves, sample grids
└── requirements.txt
```

---

## How to run

```bash
git clone https://github.com/Nimit15/Driver-Drowsiness-Detection-Using-Deep-Learning-Techniques.git
cd Driver-Drowsiness-Detection-Using-Deep-Learning-Techniques
pip install -r requirements.txt
python src/realtime_detection.py
```

Press `Q` to quit. See [`models/models_README.md`](models/models_README.md) if the model files are missing.

---

## Roadmap

Ideas for where this goes next, beyond the current scope:

- **Web dashboard** — wrap the existing pipeline in a Streamlit interface with a live score gauge and session event log, running locally first with public cloud deployment as a stretch goal (`streamlit-webrtc`, since a cloud server has no camera of its own to read from directly)
- **Personalised calibration** — a short per-user calibration step at session start to set individual EAR/MAR baselines instead of fixed thresholds
- **Broader validation** — testing across more users, lighting conditions, and camera hardware than was feasible within the project timeline
- **Yawn model improvement** — exploring alternate architectures (e.g. EfficientNetB0) for the one task where the largest accuracy gap remains

---


