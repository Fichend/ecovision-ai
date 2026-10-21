import torch
from torch import nn
from torchvision.models import mobilenet_v2


def test_mobilenet_output_shape() -> None:
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 6)
    output = model(torch.randn(2, 3, 96, 96))
    assert output.shape == (2, 6)
