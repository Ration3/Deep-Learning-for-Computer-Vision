
# Deep Learning for Computer Vision

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Advanced deep learning models and techniques for various computer vision tasks, including image classification, object detection, and segmentation.

## Features
- Implementations of popular CNN architectures (e.g., ResNet)
- Modular design for easy experimentation
- Utilizes PyTorch for efficient computation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import torch
from src.models import SimpleCNN, ResNet18

# Example usage
dummy_input = torch.randn(1, 3, 32, 32)
model = SimpleCNN(num_classes=10)
output = model(dummy_input)
print(output.shape)
```

## Project Structure

```
. \
├── src\
│   └── models.py
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
