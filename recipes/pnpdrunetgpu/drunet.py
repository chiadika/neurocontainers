#!/usr/bin/env python3
"""DRUNet (Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser
Prior", TPAMI 2021), vendored so the container needs no extra package.

The layer names below reproduce the released `drunet_gray.pth` state dict
exactly -- 64 tensors under m_head / m_down1-3 / m_body / m_up1-3 / m_tail, all
convolutions bias-free. The weights are loaded with strict=True so any
architecture drift fails loudly rather than silently producing noise.

The network takes a 2-channel input (image, noise-level map) and returns the
denoised image directly.
"""

import torch
import torch.nn as nn


def conv(cin, cout, k=3, s=1, p=1):
    return nn.Conv2d(cin, cout, k, s, p, bias=False)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.res = nn.Sequential(conv(cin, cout), nn.ReLU(inplace=True), conv(cout, cout))

    def forward(self, x):
        return x + self.res(x)


class UNetRes(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nc=(64, 128, 256, 512), nb=4):
        super().__init__()
        self.m_head = conv(in_nc, nc[0])
        self.m_down1 = nn.Sequential(*[ResBlock(nc[0], nc[0]) for _ in range(nb)],
                                     conv(nc[0], nc[1], 2, 2, 0))
        self.m_down2 = nn.Sequential(*[ResBlock(nc[1], nc[1]) for _ in range(nb)],
                                     conv(nc[1], nc[2], 2, 2, 0))
        self.m_down3 = nn.Sequential(*[ResBlock(nc[2], nc[2]) for _ in range(nb)],
                                     conv(nc[2], nc[3], 2, 2, 0))
        self.m_body = nn.Sequential(*[ResBlock(nc[3], nc[3]) for _ in range(nb)])
        self.m_up3 = nn.Sequential(nn.ConvTranspose2d(nc[3], nc[2], 2, 2, 0, bias=False),
                                   *[ResBlock(nc[2], nc[2]) for _ in range(nb)])
        self.m_up2 = nn.Sequential(nn.ConvTranspose2d(nc[2], nc[1], 2, 2, 0, bias=False),
                                   *[ResBlock(nc[1], nc[1]) for _ in range(nb)])
        self.m_up1 = nn.Sequential(nn.ConvTranspose2d(nc[1], nc[0], 2, 2, 0, bias=False),
                                   *[ResBlock(nc[0], nc[0]) for _ in range(nb)])
        self.m_tail = conv(nc[0], out_nc)

    def forward(self, x0):
        # three stride-2 stages, so both spatial dims must be multiples of 8
        h, w = x0.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            x0 = nn.functional.pad(x0, (0, pw, 0, ph), mode="replicate")
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x[..., :h, :w]


def load_drunet(path, device="cpu"):
    model = UNetRes()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)      # strict: catch any drift
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)
