# Literature Review

Six papers most relevant to this project's approach, reviewed to understand the current state of the art and identify what gap this project could actually address.

| # | Paper | Authors, Year | Method | Accuracy | Limitation |
|---|-------|---------------|--------|----------|------------|
| 1 | Real-Time Drowsiness Detection Using Eye Aspect Ratio and Facial Landmark Detection | Prerana et al., 2024 | dlib 68-point landmarks + EAR threshold | ~91% | Fixed threshold, no deep learning, fails with glasses |
| 2 | A Real-Time Embedded System for Driver Drowsiness Detection Based on CNN and MAR | Espinosa et al., 2024 | CNN + EAR/MAR on NVIDIA Jetson Nano | 97.44% | Needs specialised hardware, no accessible interface |
| 3 | Improving Driver Drowsiness Detection via Personalised EAR/MAR Thresholds and CNN Classification | Sanchez-Gendriz et al., 2025 | Personalised EAR/MAR + CNN | 94% | No head pose signal, no deployment interface |
| 4 | Driver Monitoring System Using MediaPipe and MobileNetV2 for Real-Time Fatigue Detection | Rosero-Montalvo et al., 2026 | MediaPipe 468 landmarks + MobileNetV2 + EAR + MAR + head pose | 88.89% | No session logging, no web deployment |
| 5 | CNN-Based Eye State Classification for Drowsiness Detection Using MediaPipe | Castro-Ospina et al., 2023 | ResNet50V2 / VGG16 / InceptionV3 on eye images | 99.71% | Too heavy for real-time inference on regular hardware |
| 6 | Real-Time Driver Drowsiness Detection Using Vision Transformer Architectures | Jarndal et al., 2025 | ViT / Swin Transformer | 99.15% | Too heavy for real-time PC deployment |

## What we took from this

Every paper on this list is good at one thing and weak at another — the fast ones use brittle fixed thresholds, and the accurate ones are too heavy to actually deploy anywhere. None of them combine a CNN, EAR, MAR, and head pose into one fused score, and none report a working interface a person could actually sit down and use.

That's the specific gap this project tries to fill: not necessarily the single highest accuracy number, but a system that combines all of those signals and actually runs in real time on ordinary hardware.

## Full citations

[1] Prerana et al., "Real-Time Drowsiness Detection Using Eye Aspect Ratio and Facial Landmark Detection," arXiv:2408.05836, 2024.

[2] Espinosa et al., "A Real-Time Embedded System for Driver Drowsiness Detection Based on Visual Analysis of the Eyes and Mouth Using CNN and MAR," PMC, 2024.

[3] Sanchez-Gendriz et al., "Improving Driver Drowsiness Detection via Personalized EAR/MAR Thresholds and CNN-Based Classification," arXiv:2604.22479, 2025.

[4] Rosero-Montalvo et al., "Driver Monitoring System Using Computer Vision for Real-Time Detection of Fatigue, Distraction and Emotion via Facial Landmarks and Deep Learning," Sensors / PMC, 2026.

[5] Castro-Ospina et al., "A CNN-Based Approach for Driver Drowsiness Detection by Real-Time Eye State Identification," Applied Sciences, MDPI, 2023.

[6] Jarndal et al., "Real-Time Driver Drowsiness Detection Using Transformer Architectures," Scientific Reports / Nature, 2025.
