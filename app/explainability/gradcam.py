"""Grad-CAM utilities for visual model explanations.

This implementation targets the final convolutional feature map of the
ResNet-18 backbone. It is intended for research/demo visualization only.
"""

import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._forward_handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient.detach()

    def generate(self, input_tensor: torch.Tensor, class_index: int) -> torch.Tensor:
        if isinstance(class_index, bool) or not isinstance(class_index, int) or class_index < 0:
            raise ValueError("class_index must be a non-negative integer")
        self.activations = self.gradients = None
        input_tensor = input_tensor.detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if logits.ndim != 2 or class_index >= logits.shape[1] or not torch.isfinite(logits).all():
            raise ValueError("Invalid model output or class_index")

        score = logits[:, class_index].sum()
        score.backward()
        if self.gradients is None or self.activations is None:
            raise ValueError("Target layer did not provide activations and gradients")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach()

    def close(self):
        self._forward_handle.remove()
