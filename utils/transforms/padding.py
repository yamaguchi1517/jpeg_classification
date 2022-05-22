import torch
import torch.nn.functional as F

class PadSequence:
  def __init__(self, fixed_length: int):
    self.fixed_length = fixed_length

  def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
    return F.pad(sequence, (self.fixed_length - sequence.size(0), 0), "constant", 0)