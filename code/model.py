import torch
import torchvision
from torchsummary import summary


class NN(torchvision.models.MobileNetV2):
    def __init__(self, in_channels=3, out_channels=1, settings=None):
        if settings is None:
            settings = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 32, 2, 2],
                [6, 64, 3, 2],
                [6, 128, 2, 2],
                [6, 256, 1, 1]
            ]

        super(NN, self).__init__(
            out_channels,
            round_nearest=8,
            inverted_residual_setting=settings)

        self.out_channels = out_channels

        if in_channels != 3:
            raise ValueError(
                'model only supports in_channels=3, received ' + str(in_channels))

    def normalize_pose(self, x):
        divisor = torch.cat([torch.ones(x.size(0), 3, device=x.device)] + [
            torch.norm(x[..., 3:], dim=-1, keepdim=True)] * 4, dim=-1)
        return x / divisor

    def forward(self, x):
        res = self._forward_impl(x)
        if self.out_channels == 7:
            res = self.normalize_pose(res)
        return res


if __name__ == "__main__":
    model = NN(3, 7)
    summary(model, (3, 160, 120), device='cpu')
