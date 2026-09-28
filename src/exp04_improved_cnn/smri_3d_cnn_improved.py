from pathlib import Path
import sys

import torch
import torch.nn as nn


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# RESIDUAL 3D BLOCK
# ============================================================

class Residual3DBlock(nn.Module):
    """
    Residual 3D convolutional block.

    Main path:
        Conv3d
        GroupNorm
        SiLU
        Conv3d
        GroupNorm

    Shortcut:
        Identity when channel dimensions match.
        1x1x1 convolution otherwise.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        groups=8,
    ):
        super().__init__()

        # Make sure GroupNorm has a valid number of groups.
        valid_groups = min(
            groups,
            out_channels,
        )

        while (
            out_channels % valid_groups != 0
            and valid_groups > 1
        ):
            valid_groups -= 1

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.norm1 = nn.GroupNorm(
            num_groups=valid_groups,
            num_channels=out_channels,
        )

        self.act = nn.SiLU(
            inplace=True
        )

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=valid_groups,
            num_channels=out_channels,
        )

        if (
            in_channels != out_channels
            or stride != 1
        ):

            shortcut_groups = min(
                groups,
                out_channels,
            )

            while (
                out_channels % shortcut_groups != 0
                and shortcut_groups > 1
            ):
                shortcut_groups -= 1

            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(
                    num_groups=shortcut_groups,
                    num_channels=out_channels,
                ),
            )

        else:

            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity

        out = self.act(out)

        return out


# ============================================================
# IMPROVED 3D CNN
# ============================================================

class SMRI3DImprovedCNN(nn.Module):
    """
    Improved 3D CNN for structural MRI schizophrenia
    classification.

    Input:
        (B, 1, 95, 117, 99)

    Output:
        (B, 1)

    Output is a raw logit. Apply sigmoid for probability.
    """

    def __init__(
        self,
        dropout=0.30,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        self.stem = nn.Sequential(
            nn.Conv3d(
                1,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=4,
                num_channels=16,
            ),
            nn.SiLU(
                inplace=True
            ),
        )

        # ----------------------------------------------------
        # Residual feature extractor
        # ----------------------------------------------------

        self.block1 = Residual3DBlock(
            16,
            16,
            stride=1,
            groups=4,
        )

        self.block2 = Residual3DBlock(
            16,
            32,
            stride=2,
            groups=8,
        )

        self.block3 = Residual3DBlock(
            32,
            64,
            stride=2,
            groups=8,
        )

        self.block4 = Residual3DBlock(
            64,
            128,
            stride=2,
            groups=16,
        )

        # Additional residual refinement
        # without further downsampling.

        self.block5 = Residual3DBlock(
            128,
            128,
            stride=1,
            groups=16,
        )

        # ----------------------------------------------------
        # Global feature aggregation
        # ----------------------------------------------------

        self.global_pool = (
            nn.AdaptiveAvgPool3d(1)
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                64,
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x


# ============================================================
# FACTORY
# ============================================================

def create_model(
    dropout=0.30,
):

    return SMRI3DImprovedCNN(
        dropout=dropout
    )


# ============================================================
# PROBABILITY HELPER
# ============================================================

def predict_probability(
    model,
    x,
):

    model.eval()

    with torch.no_grad():

        logits = model(x)

        probabilities = torch.sigmoid(
            logits
        )

    return probabilities


# ============================================================
# MODEL INFORMATION
# ============================================================

def count_parameters(
    model,
):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# SANITY TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "EXPERIMENT 04 — IMPROVED 3D CNN"
    )
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    model = create_model(
        dropout=0.30
    ).to(device)

    model.eval()

    test_input = torch.randn(
        2,
        1,
        95,
        117,
        99,
        device=device,
    )

    with torch.no_grad():

        output = model(
            test_input
        )

        probability = torch.sigmoid(
            output
        )

    print(
        f"Input shape : "
        f"{tuple(test_input.shape)}"
    )

    print(
        f"Output shape: "
        f"{tuple(output.shape)}"
    )

    print(
        f"Probability range: "
        f"{probability.min().item():.6f} "
        f"to "
        f"{probability.max().item():.6f}"
    )

    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    assert (
        tuple(output.shape)
        == (2, 1)
    )

    assert torch.isfinite(
        output
    ).all()

    assert (
        probability.min().item()
        >= 0.0
    )

    assert (
        probability.max().item()
        <= 1.0
    )

    print()
    print(
        "FINAL STATUS: MODEL SANITY PASS"
    )