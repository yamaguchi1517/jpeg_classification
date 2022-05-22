import torch
from scipy import stats
import torch.distributions as tdist


class RandomMASK:
    def __call__(self, input):
        noise = torch.FloatTensor(len(input),).uniform_(0.8, 1.8).to(torch.int32)
        return (input * noise).to(torch.int32)

class MultNORM:
    def __call__(self, input):
        noise = torch.tensor(stats.norm.rvs(loc=1.0, scale=0.01, size=len(input)))
        return (input * noise).to(torch.int32)

class UniformOR:
    def __call__(self, input):
        noise = torch.FloatTensor(len(input),).uniform_(0.0, 1.9).to(torch.int32)
        return (input | noise).to(torch.int32)

class RandomFLIP:
    def __call__(self, input):
        shoulder = torch.FloatTensor(len(input),).uniform_(0.0, 7.9).to(torch.int32)
        two = torch.FloatTensor(len(input),).uniform_(0.50, 1.50).to(torch.int32)*2
        noise = two ** shoulder
        return (input ^ noise).to(torch.int32)