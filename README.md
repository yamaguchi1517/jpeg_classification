# JPEGデータの機械学習

画像に対する機械学習は通常CNNを用いる。その際、圧縮された状態(JPEG, PNG, GIFなど)で格納されている画像データを一度展開する必要がある。
圧縮データから直接、学習可能であれば、展開するステップを踏まずに済み、メモリの削減に繋がる。
この技術をプライバシー保護に応用した研究を行っていた。

# プログラム概要

JPEGデータから直接学習・分類可能。
JPEG形式のデータセットを入手できなかったため、mnistなどの一般的なデータセットをJPEG形式に圧縮する前処理を施す。

# 必要環境

```bash
python3
torch == 1.6.0
torchvision == 0.7.0
```

# 実行方法

```bash
# 実行
$ python3 main_classification.py

# ヘルプを表示
$ python3 main_classification.py --help

# データセットとエポック数を指定し実行
$ python3 main_classification.py --dataset mnist --epochs 500
```

その他のコマンドライン引数はヘルプを参照。

# 参考
## Webページ

https://qiita.com/Hi-king/items/7c61f54986d9a940208f
