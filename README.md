# ResNet 手書きアルファベット認識アプリ
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
