# Models

Two trained MobileNetV2 models, each solving a separate binary classification task. See the [project report](../docs/PROJECT_REPORT.md) for why two models instead of one.

| File | Task | Size | Test Accuracy |
|------|------|------|----------------|
| `eye_model.h5` | Closed vs Open eye | 26.2 MB | 99.08% |
| `face_model.h5` | Yawn vs No Yawn | 11.3 MB | 90.37% |

Both were trained on Google Colab using two-phase transfer learning — the MobileNetV2 base frozen for the first phase, then the deepest 30 layers (50 for the yawn model) unfrozen and fine-tuned at a lower learning rate for the second phase.

## Loading the models

```python
from tensorflow.keras.models import load_model

eye_model  = load_model('models/eye_model.h5')
face_model = load_model('models/face_model.h5')
```

Both expect 96×96 RGB input, preprocessed with MobileNetV2's own `preprocess_input`:

```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

img = preprocess_input(np.expand_dims(image_array.astype('float32'), 0))
prediction = eye_model.predict(img)
```

## If you're loading these on a different machine than they were trained on

TensorFlow/Keras model files aren't always portable across versions. If you get an error mentioning `quantization_config` when loading, that means the Keras version on this machine is newer than the one the model was saved with. `src/realtime_detection.py` includes a fix for this that rewrites the model's internal metadata before loading — see the `load_models_safe()` function there rather than trying to downgrade Keras.

## Architecture reference

`cnn_architecture.json` (if present) holds the layer structure from an earlier custom-CNN experiment (Week 3–4), before we switched to MobileNetV2 transfer learning after the custom model plateaued around 33–45% accuracy due to a data-loading issue we hadn't caught yet. Kept here for the record — the models actually in use are the two `.h5` files above.
