import torch
import numpy as np
import torch.nn as nn
from torchvision.io import encode_jpeg

class EncodeJPEG:
    def __call__(self, image) -> torch.Tensor:
        return encode_jpeg(image.to(torch.uint8), 75).to(torch.int32)