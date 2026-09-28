from __future__ import annotations

import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    """
    3D convolutional block for the sMRI branch.

    Conv3d -> BatchNorm3d -> ReLU -> optional MaxPool3d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = True,
    ):
        super().__init__()

        layers = [
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if pool:
            layers.append(
                nn.MaxPool3d(
                    kernel_size=2,
                    stride=2,
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SMRICNNBackbone(nn.Module):
    """
    Baseline 3D CNN backbone.

    Input:
        (B, 1, 95, 117, 99)

    Output:
        (B, 64)

    The architecture intentionally matches the established
    baseline CNN as closely as possible so that Experiment 05
    primarily tests the contribution of AAL features.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            Conv3DBlock(1, 16, pool=True),
            Conv3DBlock(16, 32, pool=True),
            Conv3DBlock(32, 64, pool=True),
            Conv3DBlock(64, 128, pool=False),
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.projection(x)
        return x


class AALBranch(nn.Module):
    """
    AAL structural-feature branch.

    Input:
        116 AAL ROI features

    Output:
        32-dimensional learned representation

    AAL scaling is intentionally NOT performed here.
    The training pipeline must fit the scaler using only
    the training portion of each fold.
    """

    def __init__(
        self,
        input_features: int = 116,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SMRICNNAALFusion(nn.Module):
    """
    Experiment 05:
    3D sMRI CNN + AAL feature fusion.

    sMRI branch:
        3D MRI volume
        -> CNN
        -> 64-dimensional representation

    AAL branch:
        116 ROI features
        -> MLP
        -> 32-dimensional representation

    Fusion:
        64 + 32
        -> 96
        -> 32
        -> dropout
        -> binary logit

    Output:
        One logit per subject.
        Apply sigmoid externally when probability is required.
    """

    def __init__(
        self,
        aal_features: int = 116,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.smri_branch = SMRICNNBackbone()

        self.aal_branch = AALBranch(
            input_features=aal_features,
            dropout=dropout,
        )

        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
        )

    def forward(
        self,
        smri: torch.Tensor,
        aal: torch.Tensor,
    ) -> torch.Tensor:

        smri_features = self.smri_branch(smri)
        aal_features = self.aal_branch(aal)

        fused_features = torch.cat(
            [smri_features, aal_features],
            dim=1,
        )

        logits = self.fusion(fused_features)

        return logits


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():
    """
    Model sanity test.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SMRICNNAALFusion(
        aal_features=116,
        dropout=0.30,
    ).to(device)

    model.eval()

    batch_size = 2

    smri = torch.randn(
        batch_size,
        1,
        95,
        117,
        99,
        device=device,
    )

    aal = torch.randn(
        batch_size,
        116,
        device=device,
    )

    with torch.no_grad():
        logits = model(smri, aal)
        probabilities = torch.sigmoid(logits)

    print("=" * 60)
    print("EXPERIMENT 05 — CNN + AAL FUSION MODEL SANITY")
    print("=" * 60)

    print("Device:", device)
    print("sMRI input shape:", tuple(smri.shape))
    print("AAL input shape:", tuple(aal.shape))
    print("Output shape:", tuple(logits.shape))

    print(
        "Probability range:",
        f"{probabilities.min().item():.6f}",
        "to",
        f"{probabilities.max().item():.6f}",
    )

    print(
        "Trainable parameters:",
        count_parameters(model),
    )

    expected_shape = (batch_size, 1)

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape: {tuple(logits.shape)}"
        )

    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "Model produced non-finite logits."
        )

    if not torch.isfinite(probabilities).all():
        raise RuntimeError(
            "Model produced non-finite probabilities."
        )

    if not (
        torch.all(probabilities >= 0)
        and torch.all(probabilities <= 1)
    ):
        raise RuntimeError(
            "Probabilities are outside [0, 1]."
        )

    print("FINAL STATUS: MODEL SANITY PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()