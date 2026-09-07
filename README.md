# 手書きアルファベット認識

ブラウザ上でマウス・タッチ・ペンを使って **A〜Z** の英字を描き、
EMNIST Letters で学習した **ResNet** でリアルタイムに認識するデモアプリです。

このリポジトリは、`MNIST_recognition` の

- Python でモデルを学習する
- 学習済みモデルをブラウザ用形式へ書き出す
- `index.html` / `style.css` / `script.js` だけで公開側を動かす

という構成を参考にしています。

今回のモデルは畳み込み層を多数含む ResNet なので、
MLP のように JavaScript で行列積を直接実装する代わりに、
**PyTorch → ONNX → ONNX Runtime Web** でブラウザ推論を行います。

公開時には Flask や Python サーバーは不要です。
そのため **GitHub Pages だけで常時公開できます**。

---

## ディレクトリ構成

```text
emnist_resnet_github_pages/
├── index.html                       # メインページ
├── style.css                       # Webアプリのデザイン
├── script.js                       # Canvas前処理・ONNX推論・結果表示
├── model.py                        # ResNet のPyTorch定義
├── export_onnx.py                  # .pt -> .onnx 変換
├── requirements-export.txt         # ONNX変換時だけ必要なPythonライブラリ
├── training_results.md             # 学習・評価結果の記録（画面には表示しない）
├── .gitignore
├── .nojekyll
└── models/
    ├── README.md
    └── resnet_emnist_letters.onnx  # ★ export_onnx.py 実行後に生成
```

`resnet_emnist_letters.onnx` は学習済み `resnet_best.pt` から生成します。

---

## モデル

入力:

```text
1 × 28 × 28 grayscale image
```

出力:

```text
A, B, C, ..., Z の26クラス
```

ResNet の構造:

```text
Conv 1->32

ResidualBlock(32->32)
ResidualBlock(32->32)
MaxPool

ResidualBlock(32->64)
ResidualBlock(64->64)
MaxPool

ResidualBlock(64->128)
ResidualBlock(128->128)
MaxPool

Flatten
Linear 1152->256
BatchNorm
ReLU
Dropout
Linear 256->26
```

---

## Step 1: 学習済みResNetをコピー

このアプリ用フォルダへ移動します。

```bash
cd ~/emnist_resnet_github_pages
```

学習済みモデルをコピーします。

```bash
cp ~/resnet_emnist_letters/models/resnet_best.pt \
   ./resnet_best.pt
```

`resnet_best.pt` は GitHub へ公開する必要はありません。
`.gitignore` で除外しています。

---

## Step 2: ONNXへ変換

初回だけ仮想環境を作成します。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-export.txt
```

次に、

```bash
python export_onnx.py
```

を実行します。

成功すると、

```text
models/resnet_emnist_letters.onnx
```

が生成されます。

このONNXファイルには、ブラウザ推論に必要な
**ResNetの計算グラフと学習済み重み**が保存されます。

Web画面では Best epoch、Validation Accuracy、Test Accuracy などは表示しません。

---

## Step 3: ローカルで動作確認

`index.html` を直接ダブルクリックするのではなく、
ローカルHTTPサーバーから開くことを推奨します。

```bash
cd ~/emnist_resnet_github_pages
python3 -m http.server 8000
```

ブラウザで次を開きます。

```text
http://localhost:8000/
```

終了するときはターミナルで `Control + C` を押します。

---

## 使い方

1. ブラウザでアプリを開く
2. Canvas に A〜Z の英字を1文字書く
3. ストローク終了後、自動でResNetが認識する
4. 必要なら「認識する」を押して再度推論する
5. 予測文字、信頼度、上位5候補、A〜Zの確率を確認する
6. 「消去」で書き直す

---

## Canvas前処理

Canvas は黒背景・白文字です。

ブラウザ側で以下の処理を行います。

1. **Bounding box検出**  
   白い描画ピクセルの範囲を取得

2. **アスペクト比を保持して縮小**  
   描画部分が最大20×20に収まるようにリサイズ

3. **28×28へ中央配置**  
   黒背景の28×28 Canvas中央に配置

4. **正規化**

```text
pixel: 0 ... 255
    ↓
0 ... 1
    ↓
(x - 0.5) / 0.5
    ↓
-1 ... 1
```

ブラウザに書く文字は最初から正立しているため、
EMNIST生データ読み込み時に使用する回転・反転補正は適用しません。

---

## ブラウザ推論の仕組み

公開用モデル:

```text
models/resnet_emnist_letters.onnx
```

JavaScriptでは ONNX Runtime Web を使います。

概略:

```text
Canvas
  ↓
28×28 前処理
  ↓
Float32 [1, 1, 28, 28]
  ↓
ResNet (ONNX)
  ↓
26 logits
  ↓
Softmax
  ↓
A〜Z の予測確率
```

推論は利用者のブラウザ内で実行されます。
画像をFlaskサーバーへ送信する必要はありません。

---

## Step 4: GitHubへアップロード

ONNX生成後、以下のファイルが揃っていることを確認してください。

```bash
ls
ls models
```

`models` の中に

```text
resnet_emnist_letters.onnx
```

があればOKです。

GitHubへ反映します。

```bash
git add .
git commit -m "Add browser-based ResNet alphabet recognition"
git push
```

ONNXファイルもGitHubへpushしてください。

---

## Step 5: GitHub Pagesで常時公開

GitHubの対象リポジトリを開きます。

```text
Settings
  ↓
Pages
  ↓
Build and deployment
  ↓
Source: Deploy from a branch
  ↓
Branch: main
  ↓
Folder: / (root)
  ↓
Save
```

しばらく待つと、例えば

```text
https://<GitHubユーザー名>.github.io/<リポジトリ名>/
```

という公開URLが発行されます。

この方式では推論がブラウザ内で完結するため、

- Macを起動したままにする必要がない
- Flaskを常時起動する必要がない
- Renderを使う必要がない
- GitHub Pagesから常時アクセスできる

という利点があります。

---

## GitHubへ公開するもの・しないもの

公開する:

```text
index.html
style.css
script.js
models/resnet_emnist_letters.onnx
```

通常は公開しなくてよい:

```text
resnet_best.pt
.venv/
__pycache__/
```

`resnet_best.pt` を公開しなくても、
ONNXファイルだけでWebアプリは動作します。

---

## 学習・評価結果について

学習結果は `training_results.md` に記録しています。

ただし、Webアプリの目的は文字認識デモであるため、
**Best score、Validation Accuracy、Test Accuracyなどはアプリ画面には表示しません。**

---

## 技術

- Dataset: EMNIST Letters
- Training: PyTorch
- Model: ResNet
- Browser model format: ONNX
- Browser inference: ONNX Runtime Web
- Frontend: HTML / CSS / JavaScript
- Hosting: GitHub Pages
