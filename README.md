# 手書きアルファベット認識

ブラウザ上に A〜Z の英字を手書きすると、EMNIST Letters で学習した ResNet が文字を認識するデモアプリです。

推論はブラウザ内で行うため、Flask などのサーバーは必要ありません。学習済み PyTorch モデルを ONNX 形式に変換し、ONNX Runtime Web を用いて実行しています。

## デモ

GitHub Pages:

https://mukahira.github.io/handwritten-alphabet-recognition/

## 使い方

1. キャンバスに A〜Z の英字を1文字書きます。
2. ResNet が入力された文字を認識します。
3. 予測文字、信頼度、上位5候補、A〜Z の予測確率が表示されます。
4. 「消去」を押すと書き直せます。

マウス、タッチ、ペン入力に対応しています。

## モデル

データセットには **EMNIST Letters** を使用しています。

入力は 28×28 のグレースケール画像、出力は A〜Z の26クラスです。モデルは3段階の残差ブロックからなる ResNet で、チャネル数を 32 → 64 → 128 と増やした後、全結合層で分類します。

```text
Input (1 × 28 × 28)
  ↓
Conv 1 → 32
  ↓
Residual Blocks × 2
  ↓
Residual Blocks × 2
  ↓
Residual Blocks × 2
  ↓
Fully Connected
  ↓
26 classes (A–Z)
```

学習済み PyTorch モデルは ONNX 形式へ変換し、ブラウザから読み込んでいます。

## Preprocessing

手書き入力はブラウザ側で次の処理を行います。

- 描画部分の検出
- アスペクト比を保ったリサイズ
- 28×28 画像の中央への配置
- EMNIST の学習時に合わせた正規化

ブラウザ上の文字は正しい向きで入力されるため、EMNIST の元画像を読み込む際に行う回転・反転補正は適用していません。

## 学習結果

| Metric | Result |
|---|---:|
| Validation Accuracy | 96.15% |
| Test Accuracy | 95.77% |
| Macro F1 | 95.77% |

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
└── models/
    └── resnet_emnist_letters.onnx
```

Webアプリの実行に必要なのは、主に `index.html`、`style.css`、`script.js`、および ONNX モデルです。

## ローカルで実行する

リポジトリを取得して、簡易HTTPサーバーを起動します。

```bash
git clone https://github.com/mukahira/handwritten-alphabet-recognition.git
cd handwritten-alphabet-recognition
python3 -m http.server 8000
```

ブラウザで次を開きます。

```text
http://localhost:8000/
```

## ONNX モデルを作り直す

学習済みの `resnet_best.pt` がある場合は、次の手順でブラウザ用モデルを生成できます。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-export.txt
python export_onnx.py
```

生成されたモデルは次の場所に保存されます。

```text
models/resnet_emnist_letters.onnx
```

## GitHub ページ

このアプリは静的ファイルのみで動作するため、GitHub Pages で公開できます。

Repository の

`Settings` → `Pages` → `Deploy from a branch`

から `main` ブランチの `/ (root)` を指定します。

## 技術

- EMNIST Letters
- PyTorch
- ResNet
- ONNX
- ONNX Runtime Web
- HTML / CSS / JavaScript
- GitHub Pages
