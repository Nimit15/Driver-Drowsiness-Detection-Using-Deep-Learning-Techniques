# Driver Drowsiness Detection Using Deep Learning Techniques

NTCC Project | B.Tech CSE | Amity School of Engineering & Technology | May–July 2026

**Team:**
- Nimit Sharma — Enrollment: A2305224203 | Roll No: 4203
- Ayush Rawat — Enrollment: A2305224213 | Roll No: 4213

---

## What this project does

This project builds a real-time system that detects whether a driver is drowsy using a webcam feed. It uses deep learning models trained on eye and face images, combined with facial landmark detection (MediaPipe) to compute Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR) and head tilt angle. The final output is a live drowsiness score with an audio alert when the driver is detected as drowsy.

The system is deployed as a web interface using Streamlit so it can be accessed through any browser without installation.

---

## System pipeline

```
Webcam → MediaPipe Face Detection → Eye Crop → Eye State Model (Closed/Open)
                                  → MAR Calculation (yawn detection)
                                  → Head Pose Estimation
                                  → Drowsiness Score (weighted fusion)
                                  → Alert + Streamlit Dashboard
```

---

## Models used

We ended up training two separate MobileNetV2 models instead of one 4-class model because the dataset has two completely different image types — close-up eye crops for open/closed, and full driver face images for yawn/no_yawn. A single model kept confusing the two domains and gave ~35% accuracy. Splitting into two binary models fixed this.

| Model | Task | Architecture | Test Accuracy |
|-------|------|-------------|---------------|
| eye_model.h5 | Closed vs Open eye | MobileNetV2 fine-tuned | 99.54% |
| face_model.h5 | Yawn vs No Yawn | MobileNetV2 fine-tuned | 88.99%+ |

---

## Dataset

**Source:** [dheerajperumandla/drowsiness-dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset) on Kaggle

| Class | Images | Type |
|-------|--------|------|
| Closed | 726 | Cropped eye images |
| Open | 726 | Cropped eye images |
| yawn | 723 | Full face driver images |
| no_yawn | 725 | Full face driver images |

---

## Tech stack

- Python 3.10
- TensorFlow / Keras
- OpenCV
- MediaPipe
- MobileNetV2 (ImageNet pretrained)
- Streamlit
- scikit-learn
- NumPy, Pandas, Matplotlib

---

## Project structure

```
Driver-Drowsiness-Detection-Using-Deep-Learning-Techniques/
├── notebooks/
│   ├── Week1_EDA.ipynb
│   ├── Week2_Preprocessing.ipynb
│   ├── Week3_ModelDesign.ipynb
│   ├── Week4_Training_Baseline.ipynb
│   ├── Week5_Training_Baseline.ipynb
│   └── Week6_Tuning.ipynb
├── src/
│   ├── ear_mar_utils.py       # EAR, MAR, drowsiness score functions
│   └── realtime_detection.py  # webcam pipeline (Week 7)
├── app/
│   └── app.py                 # Streamlit web app (Week 8)
├── results/
│   ├── training_curves.png
│   ├── Eye_State_confusion_matrix.png
│   ├── Yawn_Detection_confusion_matrix.png
│   ├── eye_samples.png
│   ├── face_samples.png
│   └── classification_report.txt
├── data/
│   └── label_map.json
└── requirements.txt
```

---

## Weekly progress

| Week | Task | Status |
|------|------|--------|
| 1 | Topic finalization, literature survey, environment setup | ✅ Done |
| 2 | Dataset download (Kaggle), exploratory data analysis | ✅ Done |
| 3 | Data preprocessing, augmentation, train/val/test split | ✅ Done |
| 4 | CNN architecture design, EAR/MAR module implementation | ✅ Done |
| 5 | MobileNetV2 training — Eye 99.54%, Yawn 88.99% | ✅ Done |
| 6 | Hyperparameter tuning, yawn model fine-tuning above 90% | ✅ Done |
| 7 | Real-time webcam integration, MediaPipe, alert system | 🔄 In Progress |
| 8 | Streamlit app, session logging, analytics dashboard | ⏳ Pending |
| 9 | Research paper writing, final submission | ⏳ Pending |

---

## Results so far

Eye State Model — 99.54% test accuracy. Confusion matrix shows near-perfect classification with almost no misclassifications between Closed and Open eye images.

Yawn Detection Model — trained on full face driver images. Achieved 88.99% at end of Week 5, pushed further above 90% in Week 6 through extended fine-tuning (50 layers unfrozen, lr=3e-5).

---

## How to run (once complete)

```bash
pip install -r requirements.txt
streamlit run app/app.py
```
