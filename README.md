# ResNet 手書きアルファベット認識アプリ

以前の完成版アプリのデザインをそのまま使用しています。

変更点は、画面から学習・テスト評価の表示だけを削除したことです。

削除した表示:
- Best validation accuracy
- Final test accuracy
- その他の学習・テスト評価値

残している表示:
- モデル名
- 予測文字
- 信頼度
- 上位5候補
- A〜Z の予測確率
- 元のCanvas/UIデザイン

## 学習済みResNetをコピー

```bash
cp ~/resnet_emnist_letters/models/resnet_best.pt \
   ~/resnet_letters_app_original_design/models/resnet_best.pt
```

## 初回セットアップ

```bash
cd ~/resnet_letters_app_original_design
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 起動

```bash
python app.py
```

ブラウザ:
http://127.0.0.1:5000/

終了:
Control + C
