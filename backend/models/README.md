# Model Files

This directory contains pre-trained model files for face parsing and other AI features.

## BiSeNet Face Parsing Model

The face parsing service uses BiSeNet for semantic segmentation of facial parts.

### Download Instructions

1. **ONNX Model (Recommended)**

   Download the ONNX model from one of these sources:

   - Google Drive: [face_parsing.onnx](https://drive.google.com/file/d/1GoS7lWxWQ6LwA7-6UKQO3xPBXgYjvbYW/view)
   - Or convert from PyTorch model (see below)

2. **PyTorch Model (Alternative)**

   The original PyTorch model can be found at:
   https://github.com/zllrunning/face-parsing.PyTorch

   Download `79999_iter.pth` and rename to `face_parsing.pth`

### Model Placement

Place the model file in this directory:
```
backend/models/face_parsing.onnx   # ONNX version (preferred)
backend/models/face_parsing.pth    # PyTorch version (requires torch)
```

### Converting PyTorch to ONNX

If you have the PyTorch model and want to convert to ONNX:

```python
import torch
from model import BiSeNet

# Load model
net = BiSeNet(n_classes=19)
net.load_state_dict(torch.load('face_parsing.pth'))
net.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 512, 512)
torch.onnx.export(
    net,
    dummy_input,
    'face_parsing.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=11
)
```

### Model Classes

BiSeNet outputs 19 classes for CelebAMask-HQ:

| Index | Label |
|-------|-------|
| 0 | background |
| 1 | skin |
| 2 | left_eyebrow |
| 3 | right_eyebrow |
| 4 | left_eye |
| 5 | right_eye |
| 6 | eyeglasses |
| 7 | left_ear |
| 8 | right_ear |
| 9 | earrings |
| 10 | nose |
| 11 | mouth (inner) |
| 12 | upper_lip |
| 13 | lower_lip |
| 14 | neck |
| 15 | necklace |
| 16 | cloth |
| 17 | hair |
| 18 | hat |

## Fallback Behavior

If no model is available, the face parsing service falls back to landmark-based mask generation using MediaPipe Face Mesh.
