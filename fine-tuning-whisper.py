from transformers import WhisperTokenizer, WhisperProcessor, WhisperFeatureExtractor, WhisperForConditionalGeneration
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_dataset, load_from_disk, concatenate_datasets
import re
import librosa
import evaluate
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, get_scheduler, TrainerCallback

data_nb_samtale =  load_from_disk('')
rundkast = load_from_disk('')
BigBrother = load_from_disk('')


rundkast_val = rundkast["validation"]
data_nb_samtale_val = data_nb_samtale["validation"]
BigBrother_val = BigBrother["validation"]


training_set = [BigBrother["train"], data_nb_samtale["train"], rundkast["train"]]
validation_set = [BigBrother_val, data_nb_samtale_val, rundkast_val]

# Remove the 'gender' column from each dataset if it exists
datasets_to_combine =[dataset for dataset in training_set]
validation_to_combine = [dataset for dataset in validation_set]

# Combine the datasets
combined_training = concatenate_datasets(datasets_to_combine)
combined_validaton = concatenate_datasets(validation_to_combine)

combined_training = combined_training.shuffle()
combined_validaton = combined_validaton.shuffle()

tokenizer = WhisperTokenizer.from_pretrained("model_checkpoint", language="Norwegian", task="transcribe")
processor = WhisperProcessor.from_pretrained("model_checkpoint", language="Norwegian", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained("model_checkpoint")

repo_name = ""


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

model = WhisperForConditionalGeneration.from_pretrained("model_checkpoint")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Calculate steps per epoch dynamically
train_batch_size = 48
num_train_examples = len(combined_training)
steps_per_epoch = num_train_examples // train_batch_size
quarter_epoch_steps = steps_per_epoch // 4

training_args = Seq2SeqTrainingArguments(
    output_dir=repo_name,
    per_device_train_batch_size=train_batch_size,  # Increase if possible
    per_device_eval_batch_size=64,
    learning_rate=1e-5,
    weight_decay=0.005,
    warmup_steps=1000,
    num_train_epochs=15,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    predict_with_generate=True,
    save_strategy="steps",
    logging_steps=100,
    eval_steps=quarter_epoch_steps,
    save_steps=quarter_epoch_steps,
    metric_for_best_model="eval_BB-NB-RUND_wer",
    save_total_limit=2,
    load_best_model_at_end=True,
    greater_is_better=False,
    report_to=["tensorboard"],
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=combined_training,
    eval_dataset={'rundkast': rundkast_val, 'nb_samtale': data_nb_samtale_val, 'bigbrother': BigBrother_val, "BB-NB-RUND": combined_validaton},
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
trainer.save_model(repo_name)
tokenizer.save_pretrained(repo_name)

