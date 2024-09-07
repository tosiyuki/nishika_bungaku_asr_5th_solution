import evaluate
import pandas as pd
import torch
import transformers
import soundfile as sf
import librosa

from datasets import DatasetDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    set_seed
)

INPUT_PATH = "./data"
MODEL_NAME = "kotoba-tech/kotoba-whisper-v1.0"
SEED = 42
VAL_RATE = 0.0
MODEL_DIR = "./whisper_finetune"

class LazySupervisedDataset(torch.utils.data.Dataset):
    """Dataset for supervised fine-tuning.
    data_dict example:
        {
            audio: ['fine_tune/022CmvRb8aM5pKF_1.mp3', 'fine_tune/022CmvRb8aM5pKF_2.mp3', 'fine_tune/022CmvRb8aM5pKF_3.mp3', 'fine_tune/022CmvRb8aM5pKF_4.mp3', 'fine_tune/022CmvRb8aM5pKF_5.mp3']
            sentence: ['幸ありて', '普通の人ならたいして問題にすまいこのことが、', '私の心を暗くした。', 'もし耳がこのまま聞こえなくなったら、', 'その時は自殺するよりほかはないと思った。']    
        }
    """
    def __init__(
        self, 
        data_dict: dict[str, list[str]],
        tokenizer: transformers.PreTrainedTokenizer,
        feature_extractor,
    ):
        super(LazySupervisedDataset, self).__init__()

        print("Formatting inputs...Skip in lazy mode")
        self.data_dict = data_dict
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.data_dict["audio"])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        audio = self.data_dict["audio"][i]
        sentence = self.data_dict["sentence"][i]

        # audioの前処理
        #audio_array, sampling_rate = sf.read(audio, samplerate=16000)
        audio_array, sampling_rate = librosa.load(audio, sr=16000)
        input_features = feature_extractor(audio_array, sampling_rate=sampling_rate).input_features[0]

        # sentenceの前処理
        labels = self.tokenizer(sentence).input_ids

        data_dict = dict(
            input_features=input_features,
            labels=labels
        )

        return data_dict


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        #音声データのパディング
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        # print(f"音声データ🎺")
        # print(f"パディング前:{input_features}")
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt") #パディングを行い、pytorchテンソル(pt)形式として受け取る
        # print(f"パディング後:{input_features}")


        # テキストデータのパディング & filling(置き換え))処理
        # print(f"テキストデータ📗")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # print(f"パディング前:{label_features}")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")#パディングを行い、pytorchテンソル(pt)形式

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)#不適切な値を検出し、-100でfillする（置き換える)
        # print(f"パディング後:{labels}")


        # if bos token is appended in previous tokenization step,
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


if __name__ == "__main__":
    set_seed(SEED)

    # 訓練データの読み込み
    train = pd.read_csv(f"{INPUT_PATH}/train.csv")
    train_details = pd.read_csv(f"{INPUT_PATH}/train_details.csv")
    df_train = pd.merge(train, train_details,on="ID")

    # sentenseがnanの行は削除
    df_train = df_train.dropna(subset=["target_slice"])

    # datasetの作成
    train_audio = [f"train_vad_rm_noise/{i}.mp3" for i in df_train["DETAIL_ID"].to_list()]
    train_sentence = df_train["target_slice"].tolist()
    n_train = int(len(train_sentence)* (1-VAL_RATE))

    print(f"n_train: {n_train}")

    train_dataset = {
        "audio":train_audio[0:n_train],
        "sentence":train_sentence[0:n_train]
    }

    #validation data
    if VAL_RATE != 0.0:
        val_dataset = {
            "audio":train_audio[n_train:],
            "sentence":train_sentence[n_train:]
        }

    #確認
    print(f"audio sample:{train_dataset['audio'][0:5]}")
    print(f"sentence sample:{train_dataset['sentence'][0:5]}")

    #モデル
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.generation_config.language = "ja" #日本語
    model.generation_config.task = "transcribe" #音声認識タスクの一種である、文字起こし(transcribe)を行う
    model.generation_config.forced_decoder_ids = None

    #プロセッサ
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="ja", task="transcribe")

    #feature_extractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)

    #tokenizer
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="ja", task="transcribe")

    # データセットを作成
    novel_dict = DatasetDict()

    novel_dict["train"] = LazySupervisedDataset(train_dataset, tokenizer, feature_extractor)
    if VAL_RATE != 0.0:
        novel_dict["val"] = LazySupervisedDataset(val_dataset, tokenizer, feature_extractor)

     # Data Collatorを定義
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # 評価関数の定義(CER)
    metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions #文字起こし結果
        label_ids = pred.label_ids #正解

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id


        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True) #文字起こし結果
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True) #正解

        cer = metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_DIR,  #モデルの保存先
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,#学習率
        warmup_ratio=0.03,#学習率のウォームアップ期間
        #max_steps=10,
        num_train_epochs=3,
        seed=SEED,
        gradient_checkpointing=True,
        fp16=True, #GPUで学習を実行する
        # fp16=False, #CPUで学習を実行する
        evaluation_strategy= "no" if VAL_RATE == 0.0 else "steps",
        per_device_eval_batch_size=32,
        predict_with_generate=True,
        generation_max_length=448,
        save_steps= 500, #指定したステップごとに、学習途中のモデルを保存する
        eval_steps = None if VAL_RATE == 0.0 else 5, #指定したステップごとに、validationデータでモデルを評価する
        logging_steps=1, #指定したステップごとに、学習経過のログを出力する
        load_best_model_at_end=VAL_RATE != 0.0,
        metric_for_best_model="cer",
        greater_is_better=False,
        push_to_hub=False, #Falseにすると、HuggingFaceへのログイン無しで学習可能
        dataloader_num_workers=16,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        weight_decay=0.,
    )

    # use_cacheをFalseに設定
    model.config.use_cache = False

    trainer = Seq2SeqTrainer(
        args=training_args, #ハイパーパラメータ
        model=model,#ベースとなる"whisper-small"モデル
        train_dataset=novel_dict["train"],
        eval_dataset=None if VAL_RATE == 0.0 else novel_dict["val"],
        data_collator=data_collator, #data_collator (パディング処理を行う)
        compute_metrics=compute_metrics, #評価関数
        tokenizer = tokenizer,
    )

    if list(Path(MODEL_DIR).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(MODEL_DIR) #モデル
    processor.save_pretrained(MODEL_DIR) #プロセッサ