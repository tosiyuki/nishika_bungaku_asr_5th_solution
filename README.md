# nishika_bungaku_asr_5th solution
Nishikaの[【音声認識コンペ】文学作品の音声を 文字起こししよう！📘🎧](https://competition.nishika.com/competitions/audio_book_transcription/summary)の5位のコードです。

## 環境構築
```
poetry install
```

学習データは[Nishika](https://competition.nishika.com/competitions/audio_book_transcription/data)からダウンロードして全て./dataの中にいれてください。

## 実行
### 1. 前処理
preprocess.ipynbを実行してください。

### 2. 訓練
```
poetry run python train_whisper.py
```

### 3. 推論
```
poetry run python inference_whisper.py
```

## 解法解説
以下に簡単にですが解法を共有しています

https://competition.nishika.com/competitions/audio_book_transcription/topics/726
