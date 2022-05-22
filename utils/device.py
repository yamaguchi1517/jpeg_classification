from typing import Optional

import torch

#最適なデバイスを使用
class AutoDevice:
    def __init__(self, disable_cuda: bool = False):
        self.use_cuda = not disable_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.cuda_devices = []
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(f'cuda:{i}')
                self.cuda_devices.append({
                    'key': f'cuda:{i}',
                    'name': prop.name,
                    'capability': (prop.major, prop.minor),
                    'memory': prop.total_memory,
                    'num_processors': prop.multi_processor_count,
                })
            self.cuda_devices.sort(key=lambda d: d['memory'], reverse=True)
        else:
            self.cuda_devices = None

    def __call__(self, index: Optional[int] = None) -> str:
        if self.use_cuda:
            if index is None:
                return 'cuda'
            else:
                return self.cuda_devices[min(index, len(self.cuda_devices) - 1)]['key']
        else:
            return 'cpu'