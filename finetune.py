import os
import numpy as np

from torch import nn

import torch
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader

from haversine import haversine_vector


SEED_VAL = 42
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)

DATA_PATH = 'data'
MODEL_PATH = 'twitter-xlm-roberta-base'

EPOCHS = 1
BATCH_SIZE = 128


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_pattern_encodings, file_pattern_labels):
        self.encodings = {
            'attention_mask': torch.load(file_pattern_encodings % 'attention_mask'),
            'input_ids': torch.load(file_pattern_encodings % 'input_ids')
        }
        self.labels = self._load_labels(file_pattern_labels)

    def _load_labels(self, file_pattern_labels):
        with open(file_pattern_labels % 'longitude') as f:
            longitude_labels = list(map(float, map(str.strip, f.readlines())))
        with open(file_pattern_labels % 'latitude') as f:
            latitude_labels = list(map(float, map(str.strip, f.readlines())))
        with open(file_pattern_labels % 'country') as f:
            countries_labels = list(map(int, map(str.strip, f.readlines())))

        countries_len = len(set(countries_labels))
        countries = []
        for c in countries_labels:
            countries.append(countries_len * [0])
            countries[-1][c] = 1

        return [[l1, l2] + c for l1, l2, c in zip(latitude_labels, longitude_labels, countries)]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        labels = inputs.get("labels")
        logits = outputs.get("logits")

        loss_fn_MAE = nn.L1Loss()
        loss_MAE = loss_fn_MAE(logits[:, :2], labels[:, :2])

        loss_fn_CE = nn.CrossEntropyLoss()
        loss_CE = loss_fn_CE(logits[:, 2:], labels[:, 2:])
        loss = loss_MAE + 0.5 * loss_CE

        return (loss, outputs) if return_outputs else loss


class GeoBertModel:
    def __init__(self, model_path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                    num_labels=21).to(self.device)

    def train(self, train_dataset, val_dataset, training_args):
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.custom_metric
        )

        trainer.train()
        trainer.save_model("results/best_model_new")  # save best model

    @staticmethod
    def custom_metric(eval_pred):
        predictions, labels = eval_pred
        hd = haversine_vector(labels, predictions)
        return {"hd_median": np.median(hd), "hd_mean": np.mean(hd)}


if __name__ == "__main__":
    model = GeoBertModel()

    full_train_dataset = Dataset(os.path.join(DATA_PATH, 'data_%s_tensor.pt'),
                           os.path.join(DATA_PATH, 'data_labels_%s.txt'))

    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset,
                                                               [round(0.9 * len(full_train_dataset)),
                                                                round(0.1 * len(full_train_dataset))])
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=False,
        evaluation_strategy="steps",
        eval_steps=50
    )

    model.train(train_dataset, val_dataset)
