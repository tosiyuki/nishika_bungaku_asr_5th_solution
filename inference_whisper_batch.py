import os
import glob
import librosa
import pandas as pd
import torch
import multiprocessing as mp

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


BASE_MODEL = "openai/whisper-large-v3"
MODEL_NAME = "./whisper_finetune"
INPUT_PATH = "./data"
EXP = "001"
OUTOUT_PATH = f"output/exp_{EXP}"
BATCH_SIZE = 32  # バッチサイズを設定


class AudioDataset(Dataset):
    def __init__(self, audio_paths, processor):
        self.audio_paths = audio_paths
        self.processor = processor

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        audio, sr = librosa.load(audio_path, sr=16000)
        input_features = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_features.squeeze(0)
        return input_features


def collate_fn(batch):
    return torch.stack(batch)


if __name__ == "__main__":
    mp.set_start_method('spawn')

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
    list_audio_file_names = []

    for audio_path in tqdm(df_test["audio_path"].to_list(), total=len(df_test["audio_path"].to_list())):
        # 拡張子を取り除いたファイル名を取り出す
        audio_file_name = os.path.basename(audio_path)
        file_pattern = os.path.join("test_vad_rm_noise", f"{audio_file_name}_*.mp3")
        audio_file_names = glob.glob(file_pattern)

        test_audio = [f"test_vad_rm_noise/{audio_file_name}_{i+1}.mp3" for i in range(len(audio_file_names))]

        # データセットとデータローダーを作成
        dataset = AudioDataset(test_audio, processor)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, drop_last=False, num_workers=16)

        list_transcription_vad = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            ## 推論(文字起こし)の実行
            batch = batch.to("cuda")
            predicted_ids = model.generate(batch, forced_decoder_ids=forced_decoder_ids)
            transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            list_transcription_vad.extend(transcriptions)

        transcription = ""
        for i in list_transcription_vad:
            transcription += i

        list_transcription.append(transcription)

    result = pd.DataFrame({
        "ID": df_test["ID"].to_list(),
        "target": list_transcription,
    })
    result.to_csv(f"{OUTOUT_PATH}/{EXP}_submission.csv", index=False)
