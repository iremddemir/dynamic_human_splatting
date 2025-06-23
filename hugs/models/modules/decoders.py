#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
import torch
import torch.nn as nn
from loguru import logger
import torch.nn.functional as F

from .activation import SineActivation


act_fn_dict = {
    "softplus": torch.nn.Softplus(),
    "relu": torch.nn.ReLU(),
    "sine": SineActivation(omega_0=30),
    "gelu": torch.nn.GELU(),
    "tanh": torch.nn.Tanh(),
}


class MyAppearanceDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=64, act="gelu"):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.opacity = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        self.dc = nn.Linear(hidden_dim, 3)
        self.rest = nn.Linear(hidden_dim, 15 * 3)

    def forward(self, x):
        x = self.net(x)
        dc = self.dc(x).unsqueeze(1)
        rest = self.rest(x).reshape(-1, 15, 3)
        opacity = self.opacity(x)
        return {"dc_delta": dc, "opacity_delta": opacity, "rest_delta": rest}


class MyAppearanceRefiner(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=64, act="gelu"):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.opacity = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        self.dc = nn.Linear(hidden_dim, 3)
        self.rest = nn.Linear(hidden_dim, 15 * 3)

    def forward(self, x, dc, rest, opacity):
        x = self.net(x)
        dc_delta = self.dc(torch.cat()).unsqueeze(1)
        rest_delta = self.rest(x).reshape(-1, 15, 3)
        opacity_delta = self.opacity(x)
        return {
            "dc": dc + dc_delta,
            "opacity": opacity + opacity_delta,
            "rest": rest + rest_delta,
        }


class AppearanceDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=64, act="gelu"):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.opacity = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        self.shs = nn.Linear(hidden_dim, 16 * 3)

    def forward(self, x):
        x = self.net(x)
        shs = self.shs(x)
        opacity = self.opacity(x)
        return {"shs": shs, "opacity": opacity}


class DeformationDecoder(torch.nn.Module):
    def __init__(
        self,
        n_features,
        hidden_dim=128,
        weight_norm=True,
        act="gelu",
        disable_posedirs=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.sine = SineActivation(omega_0=30)
        self.disable_posedirs = disable_posedirs

        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.skinning_linear = nn.Linear(hidden_dim, hidden_dim)
        self.skinning = nn.Linear(hidden_dim, 24)

        if weight_norm:
            self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)

        # initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)
        if not disable_posedirs:
            self.blendshapes = nn.Linear(hidden_dim, 3 * 207)
            torch.nn.init.constant_(self.blendshapes.bias, 0.0)
            torch.nn.init.constant_(self.blendshapes.weight, 0.0)

    def forward(self, x):
        x = self.net(x)
        if not self.disable_posedirs:
            posedirs = self.blendshapes(x)
            posedirs = posedirs.reshape(207, -1)

        lbs_weights = self.skinning(F.gelu(self.skinning_linear(x)))
        lbs_weights = F.gelu(lbs_weights)

        return {
            "lbs_weights": lbs_weights,
            "posedirs": posedirs if not self.disable_posedirs else None,
        }


class MyGeometryDecoder(torch.nn.Module):
    def __init__(self, n_features, use_surface=False, hidden_dim=128, act="gelu"):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )

        # self.xyz = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 3))
        # self.rotations = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 3))
        # self.scales = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 2 if use_surface else 3))

        self.xyz_head = nn.Linear(hidden_dim, 3)
        self.rotations_head = nn.Linear(hidden_dim, 3)
        self.scales_head = nn.Linear(hidden_dim, 3)

        for head in (self.xyz_head, self.rotations_head, self.scales_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x):
        x = self.net(x)
        xyz = torch.tanh(self.xyz_head(x)) * 0.1
        rotations = torch.tanh(self.rotations_head(x)) * 0.5
        scales = torch.tanh(self.scales_head(x)) * 10

        return {
            "xyz_delta": xyz,
            "rotations_delta": rotations,
            "scales_delta": scales,
        }


class GeometryDecoder(torch.nn.Module):
    def __init__(self, n_features, use_surface=False, hidden_dim=128, act="gelu"):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.xyz = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 3))
        self.rotations = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 6))
        self.scales = nn.Sequential(
            self.net, nn.Linear(self.hidden_dim, 2 if use_surface else 3)
        )

    def forward(self, x):
        xyz = self.xyz(x)
        rotations = self.rotations(x)
        scales = F.gelu(self.scales(x))

        return {
            "xyz": xyz,
            "rotations": rotations,
            "scales": scales,
        }
