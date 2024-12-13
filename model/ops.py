import torch
import torch.nn as nn
import torch.nn.functional as F


def log_sum_exp(x, axis=1):
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def lrelu(x, alpha=0.2):
    return torch.maximum(x, alpha * x)


class Conv2dSN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, spectral_norm=False):
        super(Conv2dSN, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv = nn.utils.spectral_norm(conv) if spectral_norm else conv

    def forward(self, x):
        return self.conv(x)


class InstanceNorm2d(nn.Module):
    def __init__(self, num_features):
        super(InstanceNorm2d, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=True)

    def forward(self, x):
        return self.norm(x)


class DeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DeConv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.deconv(x)


def avgpool2d(x, k=2):
    return F.avg_pool2d(x, kernel_size=k, stride=k, padding=0)


def upscale(x, scale):
    _, _, h, w = x.size()
    return F.interpolate(x, scale_factor=scale, mode='nearest')


def fully_connect(input_, output_size, spectral_norm=True):
    fc = nn.Linear(input_.size(-1), output_size)
    return nn.utils.spectral_norm(fc) if spectral_norm else fc


def conv_cond_concat(x, y):
    y = y.unsqueeze(-1).unsqueeze(-1)
    return torch.cat([x, y.expand_as(x)], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride, padding=kernel // 2)
        self.instance_norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride, padding=kernel // 2)
        self.instance_norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = F.relu(self.instance_norm1(self.conv1(x)))
        out = self.instance_norm2(self.conv2(out))
        return F.relu(out + residual)


class SpectralNorm(nn.Module):
    def __init__(self, layer, power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.layer = nn.utils.spectral_norm(layer)

    def forward(self, x):
        return self.layer(x)


def batch_normal(input, epsilon=1e-5, decay=0.9):
    bn = nn.BatchNorm2d(input.size(1), eps=epsilon, momentum=1 - decay, affine=True)
    return bn(input)


def _l2normalize(v, eps=1e-12):
    return v / (torch.sum(v ** 2) ** 0.5 + eps)


def spectral_norm(w, power_iterations=1):
    w_shape = w.shape
    w = w.view(w.size(0), -1) 
    u = torch.randn(1, w.size(0)).to(w.device)
    for _ in range(power_iterations):
        v = _l2normalize(torch.matmul(u, w))
        u = _l2normalize(torch.matmul(v, w.t()))
    sigma = torch.matmul(torch.matmul(u, w), v.t())
    w_norm = w / sigma
    return w_norm.view(w_shape)
