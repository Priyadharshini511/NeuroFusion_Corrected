"""
NeuroFusion AI
Experiment 01 - GroupNorm 3D CNN

Purpose:
    Controlled replacement of BatchNorm3d with GroupNorm.

Baseline architecture is otherwise preserved.

Input:
    (B, 1, 95, 117, 99)

Output:
    (B, 1) classification logits
"""

import torch
import torch.nn as nn


# ============================================================
# GroupNorm 3D Convolution Block
# ============================================================

class GroupNorm3DBlock(nn.Module):
    """
    3D convolution block using GroupNorm instead of BatchNorm3d.

    Structure:
        Conv3D
        GroupNorm
        ReLU
        optional MaxPool3D
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_pool: bool = True,
    ):

        super().__init__()

        # Use a number of groups that divides the number
        # of channels exactly.
        #
        # For the current architecture:
        #   16 -> 4 groups
        #   32 -> 8 groups
        #   64 -> 8 groups
        #   128 -> 16 groups

        if out_channels == 16:
            num_groups = 4

        elif out_channels == 32:
            num_groups = 8

        elif out_channels == 64:
            num_groups = 8

        elif out_channels == 128:
            num_groups = 16

        else:
            # Safe fallback for any future channel count.
            num_groups = min(
                8,
                out_channels
            )

            while (
                out_channels % num_groups != 0
                and num_groups > 1
            ):
                num_groups -= 1

        layers = [

            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=out_channels,
            ),

            nn.ReLU(
                inplace=True
            ),
        ]

        if use_pool:

            layers.append(
                nn.MaxPool3d(
                    kernel_size=2,
                    stride=2,
                )
            )

        self.block = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(x)


# ============================================================
# Experiment 01 GroupNorm 3D CNN
# ============================================================

class SMRI3DCNNGroupNorm(nn.Module):
    """
    NeuroFusion AI Experiment 01.

    Same architecture as the baseline 3D CNN,
    with BatchNorm3d replaced by GroupNorm.

    Input:
        (B, 1, 95, 117, 99)

    Output:
        (B, 1) classification logits
    """

    def __init__(
        self,
        dropout: float = 0.30,
    ):

        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------

            GroupNorm3DBlock(
                in_channels=1,
                out_channels=16,
                use_pool=True,
            ),

            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------

            GroupNorm3DBlock(
                in_channels=16,
                out_channels=32,
                use_pool=True,
            ),

            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------

            GroupNorm3DBlock(
                in_channels=32,
                out_channels=64,
                use_pool=True,
            ),

            # ------------------------------------------------
            # Block 4
            # ------------------------------------------------

            GroupNorm3DBlock(
                in_channels=64,
                out_channels=128,
                use_pool=False,
            ),
        )

        self.global_pool = (
            nn.AdaptiveAvgPool3d(
                output_size=1
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                in_features=128,
                out_features=64,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                in_features=64,
                out_features=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.features(x)

        x = self.global_pool(x)

        logits = self.classifier(x)

        return logits


# ============================================================
# Factory Function
# ============================================================

def create_model(
    dropout: float = 0.30,
) -> SMRI3DCNNGroupNorm:

    return SMRI3DCNNGroupNorm(
        dropout=dropout
    )


# ============================================================
# Probability Helper
# ============================================================

def predict_probability(
    logits: torch.Tensor,
) -> torch.Tensor:

    return torch.sigmoid(
        logits
    )


# ============================================================
# Sanity Check
# ============================================================

if __name__ == "__main__":

    print("=" * 72)
    print("NeuroFusion AI - Experiment 01 GroupNorm 3D CNN")
    print("=" * 72)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    model = create_model()

    model = model.to(
        device
    )

    model.eval()

    x = torch.randn(
        2,
        1,
        95,
        117,
        99,
        device=device,
    )

    with torch.no_grad():

        logits = model(
            x
        )

        probabilities = predict_probability(
            logits
        )

    print(
        f"Input shape        : "
        f"{tuple(x.shape)}"
    )

    print(
        f"Output shape       : "
        f"{tuple(logits.shape)}"
    )

    print(
        f"Probability range  : "
        f"{probabilities.min().item():.6f} .. "
        f"{probabilities.max().item():.6f}"
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters   : "
        f"{parameter_count:,}"
    )

    print(
        f"Trainable params   : "
        f"{trainable_parameter_count:,}"
    )

    # Verify that the model actually contains GroupNorm
    # and does not contain BatchNorm3d.

    group_norm_count = sum(
        1
        for module in model.modules()
        if isinstance(
            module,
            nn.GroupNorm
        )
    )

    batch_norm_count = sum(
        1
        for module in model.modules()
        if isinstance(
            module,
            nn.BatchNorm3d
        )
    )

    print(
        f"GroupNorm layers   : "
        f"{group_norm_count}"
    )

    print(
        f"BatchNorm3d layers : "
        f"{batch_norm_count}"
    )

    assert (
        group_norm_count == 4
    ), (
        "Expected exactly 4 GroupNorm layers."
    )

    assert (
        batch_norm_count == 0
    ), (
        "BatchNorm3d must not be present "
        "in Experiment 01."
    )

    assert (
        tuple(logits.shape)
        == (2, 1)
    ), (
        "Unexpected model output shape."
    )

    assert (
        torch.isfinite(
            logits
        ).all()
    ), (
        "Non-finite logits detected."
    )

    assert (
        torch.isfinite(
            probabilities
        ).all()
    ), (
        "Non-finite probabilities detected."
    )

    print()
    print(
        "GROUPNORM MODEL SANITY CHECK: PASS"
    )