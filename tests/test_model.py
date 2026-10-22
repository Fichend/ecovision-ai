import torch
from torch import nn
from torchvision.models import mobilenet_v3_large


def test_mobilenet_v3_12class_output_shape() -> None:
    model = mobilenet_v3_large(weights=None)
    final = model.classifier[-1]
    model.classifier[-1] = nn.Linear(final.in_features, 12)
    output = model(torch.randn(2, 3, 96, 96))
    assert output.shape == (2, 12)
