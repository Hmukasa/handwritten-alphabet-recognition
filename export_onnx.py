"""
export_onnx.py
--------------

学習済み PyTorch checkpoint ``resnet_best.pt`` を
GitHub Pages / ONNX Runtime Web で使用できる ONNX 形式へ変換する。

入力:
    resnet_best.pt

出力:
    models/resnet_emnist_letters.onnx

公開用ONNXには推論グラフと学習済み重みだけが入る。
Best epoch / validation accuracy / test accuracy などの
評価表示用メタデータはWebアプリでは使用しない。
"""

from pathlib import Path

import torch

from model import ResNet, parameter_count


CHECKPOINT_PATH = Path("resnet_best.pt")
OUTPUT_PATH = Path("models/resnet_emnist_letters.onnx")


def load_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if isinstance(checkpoint, dict):
        # state_dictそのものを保存した形式にも対応
        tensor_values = [
            value
            for value in checkpoint.values()
            if isinstance(value, torch.Tensor)
        ]

        if tensor_values:
            return checkpoint

    raise ValueError(
        "checkpoint から state_dict を取得できませんでした。"
    )


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH} がありません。\n"
            "学習済みモデルをこのフォルダへコピーしてください。"
        )

    print(f"Checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = load_state_dict(checkpoint)

    model = ResNet(num_classes=26)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Parameters: {parameter_count(model):,}")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dummy_input = torch.zeros(
        1, 1, 28, 28,
        dtype=torch.float32,
    )

    print(f"Exporting: {OUTPUT_PATH}")

    # dynamo=False を明示し、広いPyTorch環境で扱いやすい
    # 従来型ONNX exporterを使用する。
    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_PATH),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        export_params=True,
        do_constant_folding=True,
        dynamo=False,
    )

    size_mb = OUTPUT_PATH.stat().st_size / (1024 ** 2)

    print()
    print("ONNX export completed.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Size: {size_mb:.2f} MB")
    print("Input shape:  [1, 1, 28, 28]")
    print("Output shape: [1, 26]")
    print()
    print(
        "次は python3 -m http.server 8000 を実行し、"
        "ブラウザで動作確認してください。"
    )


if __name__ == "__main__":
    main()
