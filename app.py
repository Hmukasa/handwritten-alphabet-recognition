import base64
import io
import os
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

from model import build_model


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        APP_DIR / "models" / "resnet_best.pt",
    )
)

app = Flask(__name__)

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
model = None
model_info = {}


def load_model():
    global model, model_info

    if not MODEL_PATH.exists():
        model = None
        model_info = {}
        return

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model_type = checkpoint.get(
        "model_type",
        "cnn",
    )

    loaded = build_model(model_type)
    loaded.load_state_dict(
        checkpoint["state_dict"]
    )
    loaded.eval()

    model = loaded
    model_info = {
        "model_type": model_type.upper(),
    }


load_model()


def decode_data_url(data_url: str) -> Image.Image:
    if not data_url or "," not in data_url:
        raise ValueError("Invalid image data.")

    _, encoded = data_url.split(",", 1)
    raw = base64.b64decode(encoded)

    return Image.open(
        io.BytesIO(raw)
    ).convert("L")


def preprocess_canvas_image(image: Image.Image) -> torch.Tensor:
    """
    Canvas is black with white handwriting.

    1. crop the written region
    2. preserve aspect ratio
    3. resize to fit within 20x20
    4. center in a 28x28 image
    5. normalize to [-1, 1]
    """
    arr = np.asarray(
        image,
        dtype=np.uint8,
    )

    mask = arr > 20
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError(
            "文字が書かれていません。"
        )

    left = xs.min()
    right = xs.max() + 1
    top = ys.min()
    bottom = ys.max() + 1

    cropped = image.crop(
        (left, top, right, bottom)
    )

    w, h = cropped.size

    target = 20
    scale = min(
        target / w,
        target / h,
    )

    new_w = max(
        1,
        round(w * scale),
    )
    new_h = max(
        1,
        round(h * scale),
    )

    resized = cropped.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "L",
        (28, 28),
        color=0,
    )

    x = (28 - new_w) // 2
    y = (28 - new_h) // 2

    canvas.paste(
        resized,
        (x, y),
    )

    x_np = (
        np.asarray(
            canvas,
            dtype=np.float32,
        )
        / 255.0
    )

    x_np = (
        x_np - 0.5
    ) / 0.5

    return torch.from_numpy(
        x_np
    ).unsqueeze(0).unsqueeze(0)


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_info=model_info,
    )


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        **model_info,
    })


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({
            "error":
                "models/resnet_best.pt がありません。"
                "学習済み resnet_best.pt を models フォルダにコピーしてください。"
        }), 503

    payload = request.get_json(
        silent=True
    ) or {}

    try:
        image = decode_data_url(
            payload.get("image")
        )

        x = preprocess_canvas_image(
            image
        )

        with torch.no_grad():
            logits = model(x)
            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0].cpu().numpy()

        best_idx = int(
            np.argmax(probabilities)
        )

        order = np.argsort(
            probabilities
        )[::-1]

        return jsonify({
            "prediction": LETTERS[best_idx],
            "confidence": float(
                probabilities[best_idx]
            ),
            "top5": [
                {
                    "letter": LETTERS[int(i)],
                    "probability": float(
                        probabilities[int(i)]
                    ),
                }
                for i in order[:5]
            ],
            "probabilities": [
                {
                    "letter": LETTERS[i],
                    "probability": float(
                        probabilities[i]
                    ),
                }
                for i in range(26)
            ],
            "model": model_info,
        })

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 400


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
