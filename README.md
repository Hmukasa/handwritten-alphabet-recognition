# 手書きアルファベット認識

ブラウザ上で手書きした A〜Z の英字を、EMNIST Letters で学習した ResNet を用いて認識するWebアプリです。

推論には ONNX Runtime Web を使用しています。

## デモ

GitHub Pages:

https://hmukasa.github.io/handwritten-alphabet-recognition/

## 使い方

1. キャンバスに A〜Z の英字を1文字書きます。
2. 「認識する」を押します。
3. 予測文字、信頼度、上位5候補、A〜Z の予測確率が表示されます。
4. 「消去」を押すと書き直せます。

マウス、タッチ、ペン入力に対応しています。

## モデル

データセットには EMNIST Letters を使用しています。

入力は 28×28 のグレースケール画像で、出力は A〜Z の26クラスです。

モデルには ResNet を使用しています。

```text
Input (1 × 28 × 28)
        ↓
Conv (1 → 32)
        ↓
Residual Blocks × 2
        ↓
Max Pooling
        ↓
Residual Blocks × 2
        ↓
Max Pooling
        ↓
Residual Blocks × 2
        ↓
Max Pooling
        ↓
Fully Connected
        ↓
26 classes (A–Z)
```

## 前処理

ブラウザ上で入力された文字に対して、次の前処理を行います。

- 描画部分の検出
- アスペクト比を保った縮小
- 28×28画像の中央への配置
- 学習時と同じ正規化

入力された文字は、描画領域を切り出して20×20以内に縮小し、28×28の画像中央に配置してからモデルへ入力します。

## 学習結果

| Metric | Result |
| --- | ---: |
| Validation Accuracy | 96.15% |
| Test Accuracy | 95.77% |
| Macro F1 | 95.77% |

詳細は `training_results.md` に記載しています。

## ファイル構成

```text
.
├── index.html
├── style.css
├── script.js
├── model.py
├── export_onnx.py
├── requirements-export.txt
├── training_results.md
├── README.md
└── models/
    └── resnet_emnist_letters.onnx
```

主なファイルの役割は次のとおりです。

- `index.html`: Webページの構成
- `style.css`: Webページのデザイン
- `script.js`: 描画処理、前処理、文字認識
- `model.py`: ResNetの定義
- `export_onnx.py`: PyTorchモデルをONNX形式へ変換
- `models/resnet_emnist_letters.onnx`: Webアプリで使用する学習済みモデル
- `training_results.md`: 学習結果

## ローカルで実行する

リポジトリを取得します。

```bash
git clone https://github.com/Hmukasa/handwritten-alphabet-recognition.git
cd handwritten-alphabet-recognition
```

ローカルサーバーを起動します。

```bash
python3 -m http.server 8000
```

Windowsで `python3` が使用できない場合は、次のコマンドでも実行できます。

```bash
python -m http.server 8000
```

ブラウザで次のURLを開きます。

```text
http://localhost:8000/
```

## ONNXモデルの作成

学習済みのPyTorchモデル `resnet_best.pt` から、Webアプリで使用するONNXモデルを作成できます。

### 1. 仮想環境を作成する

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. 必要なライブラリをインストールする

```bash
pip install -r requirements-export.txt
```

### 3. ONNX形式に変換する

```bash
python export_onnx.py
```

変換したモデルは次の場所に保存されます。

```text
models/resnet_emnist_letters.onnx
```

## 認識の流れ

Webアプリでは次の流れで文字認識を行います。

```text
手書き入力
    ↓
描画領域を検出
    ↓
28×28に変換
    ↓
正規化
    ↓
ResNet
    ↓
A〜Zの予測確率
    ↓
認識結果を表示
```

## 使用技術

- EMNIST Letters
- PyTorch
- ResNet
- ONNX
- ONNX Runtime Web
- HTML
- CSS
- JavaScript
- GitHub Pages
