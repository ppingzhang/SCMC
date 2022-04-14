import torch
import torch.nn as nn


class ScaleLayer(nn.Module):
    def __init__(self, in_channel=3, out_channel=100):
        super(ScaleLayer, self).__init__()
        self.l1 = torch.nn.Linear(in_channel, 50)
        self.relu1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(50, out_channel)
        self.exp = torch.nn.Sigmoid()

    def forward(self, x):
        # x: [B, C]
        l1_x = self.l1(x)
        l1_r = self.relu1(l1_x)
        l2_x = self.l2(l1_r)
        scale = self.exp(l2_x)
        ss = scale.shape
        scale = scale.reshape([ss[0], ss[1], 1, 1])
        return scale


if __name__ == "__main__":
    sl = ScaleLayer(3, 32)
    x = torch.zeros([2, 3])

    b = torch.zeros([2, 32, 64, 64])
    out = sl(x)
    out = out.reshape([2, 32, 1, 1])
    print(b.shape, out.shape)
    out = out * b
    print(out.shape)