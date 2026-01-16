# Model Conversion Guide

## Successfully Generated

✅ **ONNX Model**: `segment_model.onnx` (233 MB)

The PyTorch model has been successfully converted to ONNX format, which is a universal intermediate format for neural networks.

## Converting ONNX to TensorFlow Lite

Unfortunately, the `onnx2tf` library has compatibility issues with macOS ARM64 (Apple Silicon) due to broken native library dependencies in `ai-edge-litert`.

### Option 1: Use Online Converters (Easiest)

1. **Convertio** - https://convertio.co/onnx-tflite/
   - Upload your `segment_model.onnx`
   - Download the converted `.tflite` file

2. **ONNX Model Zoo Tools** - Various online tools available

### Option 2: Use Docker with Linux Environment

```bash
# Pull TensorFlow Docker image
docker run -it --rm -v $(pwd):/workspace tensorflow/tensorflow:latest bash

# Inside container
cd /workspace
pip install onnx2tf onnx onnxruntime tensorflow
onnx2tf -i segment_model.onnx -o segment_model_tf -osd -cotof
```

### Option 3: Use Google Colab (Free, Cloud-based)

Create a new Colab notebook and run:

```python
!pip install onnx2tf onnx onnxruntime tensorflow tf-keras

from google.colab import files
uploaded = files.upload()  # Upload your segment_model.onnx

import onnx2tf
onnx2tf.convert(
    input_onnx_file_path='segment_model.onnx',
    output_folder_path='output',
    copy_onnx_input_output_names_to_tflite=True,
    output_signaturedefs=True,
)

# Download the generated .tflite file
from google.colab import files
import os
tflite_files = [f for f in os.listdir('output') if f.endswith('.tflite')]
for f in tflite_files:
    files.download(f'output/{f}')
```

### Option 4: Use ONNX Runtime Directly (Recommended for Python)

Instead of converting to TFLite, use ONNX Runtime which is more reliable:

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load model
session = ort.InferenceSession('segment_model.onnx')

# Prepare input
image = Image.open('pig.jpg').convert('RGB')
image = image.resize((256, 256))
input_data = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
input_data = np.expand_dims(input_data, axis=0)

# Normalize
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
input_data = (input_data - mean) / std

# Run inference
outputs = session.run(None, {'input': input_data})
mask = outputs[0]

# Apply sigmoid and threshold
mask = 1 / (1 + np.exp(-mask))  # sigmoid
binary_mask = (mask > 0.7).astype(np.uint8)
```

### Option 5: Convert on Linux/Windows

If you have access to a Linux or Windows machine, the conversion should work fine:

```bash
pip install onnx2tf onnx onnxruntime tensorflow tf-keras ai-edge-litert
python transform.py
```

## Model Details

- **Input**: `(1, 3, 256, 256)` - NCHW format (batch, channels, height, width)
- **Output**: `(1, 1, 256, 256)` - Segmentation mask
- **Architecture**: DeepLabV3 with ResNet101 backbone
- **Input Normalization**:
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- **Output**: Logits (apply sigmoid for probabilities)
- **Threshold**: 0.7 for binary mask

## Notes

- The ONNX model is fully functional and can be used for inference
- ONNX Runtime provides excellent performance and is available on all platforms
- For mobile deployment, use one of the TFLite conversion options above
- The issue is specifically with `ai-edge-litert` on macOS ARM64, not with your model or code
