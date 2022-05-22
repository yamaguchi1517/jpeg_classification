import torch
import numpy as np
import PIL.Image as Image
import matplotlib as plt

class ToTensor:
    def __call__(self, image) -> torch.Tensor:
        image_array = np.array(image).astype(np.float).reshape(-1, 32, 32)
        return torch.from_numpy(image_array)