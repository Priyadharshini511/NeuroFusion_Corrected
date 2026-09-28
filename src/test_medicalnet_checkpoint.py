import sys
from pathlib import Path

import torch
import torch.nn as nn


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MEDICALNET_ROOT = PROJECT_ROOT / "MedicalNet"

CHECKPOINT_PATH = (
    MEDICALNET_ROOT
    / "pretrain"
    / "resnet_18_23dataset.pth"
)

sys.path.insert(
    0,
    str(MEDICALNET_ROOT)
)

from models.resnet import resnet18


# ================================================================
# CREATE MEDICALNET RESNET-18
# ================================================================

print("=" * 70)
print("MEDICALNET RESNET-18 CHECKPOINT COMPATIBILITY TEST")
print("=" * 70)

print(f"Checkpoint: {CHECKPOINT_PATH}")

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )


model = resnet18(
    sample_input_D=95,
    sample_input_H=117,
    sample_input_W=99,
    num_seg_classes=2,
    shortcut_type="A",
    no_cuda=True,
)

print("MedicalNet model created.")


# ================================================================
# LOAD CHECKPOINT
# ================================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)

if "state_dict" not in checkpoint:
    raise RuntimeError(
        "Checkpoint does not contain 'state_dict'."
    )

state_dict = checkpoint["state_dict"]


# ================================================================
# REMOVE DATA-PARALLEL PREFIX
# ================================================================

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        clean_key = key[len("module."):]
    else:
        clean_key = key

    clean_state_dict[clean_key] = value


print(
    f"Checkpoint tensors: "
    f"{len(clean_state_dict)}"
)


# ================================================================
# ENCODER-ONLY LOADING
# ================================================================

model_state = model.state_dict()

encoder_state = {
    key: value
    for key, value in clean_state_dict.items()
    if key in model_state
    and not key.startswith("conv_seg.")
}

missing_before = [
    key
    for key in model_state
    if key not in encoder_state
    and not key.startswith("conv_seg.")
]

shape_mismatch = []

for key, value in encoder_state.items():

    if tuple(value.shape) != tuple(model_state[key].shape):
        shape_mismatch.append(
            (
                key,
                tuple(value.shape),
                tuple(model_state[key].shape),
            )
        )


print(
    f"Matching encoder tensors: "
    f"{len(encoder_state)}"
)

print(
    f"Missing encoder tensors: "
    f"{len(missing_before)}"
)

print(
    f"Shape mismatches: "
    f"{len(shape_mismatch)}"
)


if shape_mismatch:

    print("\nShape mismatches:")

    for item in shape_mismatch[:10]:
        print(item)

    raise RuntimeError(
        "Checkpoint/model shape mismatch detected."
    )


if missing_before:

    print("\nMissing encoder keys:")

    for key in missing_before[:20]:
        print(key)

    raise RuntimeError(
        "Missing MedicalNet encoder weights detected."
    )


# ================================================================
# LOAD WEIGHTS
# ================================================================

load_result = model.load_state_dict(
    encoder_state,
    strict=False,
)

print()
print(
    "load_state_dict missing keys:",
    len(load_result.missing_keys),
)

print(
    "load_state_dict unexpected keys:",
    len(load_result.unexpected_keys),
)


# ================================================================
# REPLACE SEGMENTATION HEAD WITH CLASSIFIER
# ================================================================

class MedicalNetClassifier(nn.Module):

    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Linear(
            512,
            1,
        )

    def forward(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        x = self.classifier(x)

        return x


classifier_model = MedicalNetClassifier(
    model
)


# ================================================================
# FORWARD-PASS TEST
# ================================================================

classifier_model.eval()

dummy_input = torch.randn(
    2,
    1,
    95,
    117,
    99,
)

with torch.no_grad():

    output = classifier_model(
        dummy_input
    )

probability = torch.sigmoid(
    output
)


print()
print(
    f"Dummy input shape: "
    f"{tuple(dummy_input.shape)}"
)

print(
    f"Output shape: "
    f"{tuple(output.shape)}"
)

print(
    f"Probability range: "
    f"{probability.min().item():.6f} - "
    f"{probability.max().item():.6f}"
)


# ================================================================
# PARAMETER COUNT
# ================================================================

total_params = sum(
    p.numel()
    for p in classifier_model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in classifier_model.parameters()
    if p.requires_grad
)

print(
    f"Total parameters: "
    f"{total_params:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_params:,}"
)


# ================================================================
# FINAL STATUS
# ================================================================

print()
print("=" * 70)
print("MEDICALNET CHECKPOINT COMPATIBILITY: PASS")
print("=" * 70)