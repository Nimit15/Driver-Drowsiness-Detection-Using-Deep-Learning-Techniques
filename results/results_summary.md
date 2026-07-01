# Results Summary — Weeks 5 & 6

## Model Training Results

### Eye State Model (Closed vs Open)

Architecture: MobileNetV2 (ImageNet pretrained) + custom head  
Training: 2-phase — frozen base (20 epochs) then fine-tuned last 30 layers (20 epochs)  
Input: 96x96 RGB eye crop images

| Metric | Value |
|--------|-------|
| Test Accuracy | 99.54% |
| Test Loss | 0.0295 |
| Precision (Closed) | ~1.00 |
| Precision (Open) | ~0.99 |
| Recall (Closed) | ~0.99 |
| Recall (Open) | ~1.00 |

The eye model performs almost perfectly. The visual difference between open and closed eye images is very clear at 96x96 RGB so MobileNetV2 has no trouble learning this distinction.

---

### Yawn Detection Model (yawn vs no_yawn)

Architecture: MobileNetV2 (ImageNet pretrained) + custom head  
Training: 2-phase — frozen base (20 epochs) then fine-tuned last 30 layers (20 epochs) + Week 6 extended fine-tuning (50 layers, lr=3e-5)  
Input: 96x96 RGB full face driver images

### Yawn Detection Model (yawn vs no_yawn)

| Metric | Week 5 | Week 6 (fine-tuned: 50 layers, lr=3e-5) |
|--------|--------|------------------------------------------|
| Test Accuracy | 88.99% | 90.37% |

Fine-tuning unfroze 50 MobileNetV2 layers (up from 30) and used a lower
learning rate of 3e-5, which allowed the model to adapt more of its
feature extractors to the full-face driver images without destroying
the pretrained weights.

Yawn detection is harder because the images are full face shots in a car environment with varying lighting, glasses, different people. 88-90% is solid for this task.

---

## Why we used two separate models

The dataset has two completely different image types:
- Closed/Open folders → close-up cropped eye images
- yawn/no_yawn folders → full driver face images taken from inside a car

We initially tried training a single 4-class model but it kept getting 33-45% accuracy because the model was trying to learn two completely different visual styles at once and kept predicting the same class for everything (confusion matrix showed all predictions going into one or two columns).

The fix was to train two separate binary MobileNetV2 models, one per visual domain. This immediately gave 99.54% on eye state and 88.99% on yawn detection.

In the real-time system both models run simultaneously on each webcam frame. Their outputs are combined with the EAR/MAR geometry signals into a single drowsiness score.

---

## Previous baselines (for comparison in research paper)

| Attempt | Accuracy | Reason for failure |
|---------|----------|-------------------|
| Week 4 custom CNN | 33.56% | Wrong data path, labels not loading |
| First fix attempt | 45.29% | Still wrong path, grayscale instead of RGB |
| Two-model MobileNetV2 | 99.54% + 88.99% | Correct path, RGB, separate domains |

---

## Files saved

| File | Location |
|------|----------|
| eye_model.h5 | Google Drive/models/ |
| face_model.h5 | Google Drive/models/ |
| training_curves.png | results/ |
| Eye_State_confusion_matrix.png | results/ |
| Yawn_Detection_confusion_matrix.png | results/ |
| eye_samples.png | results/ |
| face_samples.png | results/ |
