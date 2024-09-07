import os
import glob
import librosa
import pandas as pd

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm


BASE_MODEL = "kotoba-tech/kotoba-whisper-v1.0"
MODEL_NAME = "./whisper_finetune"
INPUT_PATH = "./data"
EXP = "001"
OUTOUT_PATH = f"output/exp_{EXP}"


if __name__ == "__main__":
    # 保存用のディレクトリの作成
    os.makedirs(OUTOUT_PATH, exist_ok=True)

    # モデルとプロセッサをロード:
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model.to("cuda")

    # モデルの設定
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task="transcribe")

    # テストデータの読み込み
    df_test = pd.read_csv(f"{INPUT_PATH}/test.csv")

    list_transcription = [] #文字起こし結果
    list_transcription_vad = []
    list_audio_file_names = []

    for audio_path in tqdm(df_test["audio_path"].to_list(), total=len(df_test["audio_path"].to_list())):
        # 拡張子を取り除いたファイル名を取り出す
        audio_file_name = os.path.basename(audio_path)
        file_pattern = os.path.join("test_vad_rm_noise", f"{audio_file_name}_*.mp3")
        audio_file_names = glob.glob(file_pattern)

        transcription = ""
        for i in range(len(audio_file_names)):
            ## 音声データ読み込み
            audio, sr = librosa.load(f"test_vad_rm_noise/{audio_file_name}_{i+1}.mp3", sr=16000)
            list_audio_file_names.append(f"test_vad_rm_noise/{audio_file_name}_{i+1}.mp3")

            ##音声データを、テンソルに変換：Whisper入力用
            input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to("cuda")

            ##推論(文字起こし)の実行
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription_vad = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcription += transcription_vad

            list_transcription_vad.append(transcription_vad)

        list_transcription.append(transcription)

    # データの出力
    vad_result = pd.DataFrame({
        "audio_path": list_audio_file_names,
        "target": list_transcription_vad,
    })
    vad_result.to_csv(f"{OUTOUT_PATH}/{EXP}_vad_output.csv", index=False)

    result = pd.DataFrame({
        "ID": df_test["ID"].to_list(),
        "target": list_transcription,
    })
    result.to_csv(f"{OUTOUT_PATH}/{EXP}_submission.csv", index=False)
