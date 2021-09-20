import torch
from torchsummary import summary


class NN(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(NN, self).__init__()
        self.out_channels = out_channels
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, out_channels),
        )

    def normalize_pose(self, x):
        divisor = torch.cat([torch.ones(x.size(0), 3, device=x.device)] + [
            torch.norm(x[..., 3:], dim=-1, keepdim=True)] * 4, dim=-1)
        return x / divisor

    def forward(self, x):
        res = self.layers(x)
        if self.out_channels == 7:
            res = self.normalize_pose(res)
        return res


if __name__ == '__main__':
    model = NN(7, 7)
    summary(model, (7,), device='cpu')
