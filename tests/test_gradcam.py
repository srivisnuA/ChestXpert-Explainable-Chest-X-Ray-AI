import torch
from torch import nn

from app.explainability.gradcam import GradCAM


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_gradcam_shape():
    model = TinyCNN()
    cam = GradCAM(model, model.features[0])

    result = cam.generate(torch.randn(1, 3, 32, 32), class_index=0)

    assert result.shape == (1, 1, 32, 32)
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)

    cam.close()
