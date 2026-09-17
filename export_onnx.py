from pathlib import Path
import torch
from model import ResNet


CHECKPOINT_PATH = Path("resnet_best.pt")
OUTPUT_PATH = Path(
    "models/resnet_emnist_letters.onnx"
)


def load_state_dict(checkpoint):
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        return checkpoint["state_dict"]

    if isinstance(checkpoint, dict):
        return checkpoint

    raise ValueError(
        "checkpoint の形式を確認してください。"
    )


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH} が見つかりません。"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    state_dict = load_state_dict(checkpoint)

    model = ResNet(num_classes=26)
    model.load_state_dict(state_dict)
    model.eval()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dummy_input = torch.zeros(
        1,
        1,
        28,
        28,
        dtype=torch.float32
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_PATH),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False
    )

    print(
        f"保存しました: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
