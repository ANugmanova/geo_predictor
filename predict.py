import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader

BATCH_SIZE = 64
MAX_LENGTH = 50
MODEL_PATH = 'geo-twitter-xlm'


class GeoBertModel:
    def __init__(self, model_path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                    num_labels=21).to(self.device)

    def predict(self, texts):
        encoded_input = self.tokenizer(texts, return_tensors='pt', truncation=True,
                                       padding='max_length', max_length=MAX_LENGTH)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        outputs = self.model(**encoded_input)
        logits = outputs.get("logits").cpu().detach()[:, :2].numpy()
        return logits


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.texts = dataset['text']
        self.labels = [np.array([float(latitude), float(longitude)])
                       for latitude, longitude in zip(dataset['latitude'], dataset['longitude'])]

    def __getitem__(self, idx):
        item = {
            'labels': self.labels[idx],
            'texts': self.texts[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)


def haversine_vector(c1, c2, r=6371):
    c1 = np.radians(c1)
    c2 = np.radians(c2)
    d = c2 - c1

    a = np.sin(d[:, 0] / 2) ** 2 + np.cos(c1[:, 0]) * np.cos(c2[:, 0]) * np.sin(d[:, 1] / 2) ** 2
    d = 2 * np.arcsin(np.sqrt(a))  

    return r * d


def evaluate(model, eval_dataloader):
    haversine_distances = []
    model.model.eval()
    for batch in tqdm(eval_dataloader):
        pred = model.predict(batch.get("texts"))
        target = np.array((batch.get("labels")))
        haversine_distances.append(haversine_vector(pred, target))
    haversine_distances = np.concatenate(haversine_distances)
    return np.mean(haversine_distances), np.median(haversine_distances)


def load_data(file_name):
    dataset = pd.read_csv(file_name)[:2000]
    dataset = Dataset(dataset)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    return dataloader


if __name__ == '__main__':
    test_file_name = 'test.csv'
    print('Loading data...')
    eval_dataloader = load_data(test_file_name)
    print('Loading model...')
    model = GeoBertModel(model_path=MODEL_PATH)
    print('Evaluating...')
    haversine_mean, haversine_medial = evaluate(model, eval_dataloader)
    print(f'Haversine Distance: \n\tMean:{haversine_mean}\n\tMedian:{haversine_medial}')


