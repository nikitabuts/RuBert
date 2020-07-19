import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import transformers
from transformers import BertModel, BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

from collections import defaultdict
import pickle

import logging
logging.basicConfig(level=logging.ERROR)


def f1(outputs, targets):
    return f1_score(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy(), average='macro')


def custom_oversampling(comments, targets):
    counter = targets.value_counts()
    num_repeat = (counter.max() / counter).map(int)
    repeats = targets.copy().map({0: num_repeat[0], 1: num_repeat[1], 2: num_repeat[2]})
    return comments.repeat(repeats), targets.repeat(repeats)


class SentimentDataset(Dataset):
  
  def __init__(self, comments, targets, tokenizer, max_len):
    self.comments = comments
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.comments)

  def __getitem__(self, sample):
    comment = self.comments[sample]
    target = self.targets[sample]

    encoding = self.tokenizer.encode_plus(
      comment,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
      truncation=True
    )

    return {
      'comment_text': comment,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


class SentimentClassifier(nn.Module):
  
  def __init__(self, n_classes, dropout, tokens_length, 
                PRE_TRAINED_MODEL_NAME, PRE_TRAINED_MODEL_CONFIG):
    super(SentimentClassifier, self).__init__()
    config = BertConfig.from_json_file(PRE_TRAINED_MODEL_CONFIG)
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, config=config)
    self.bert.resize_token_embeddings(tokens_length)
    self.drop = nn.Dropout(p=dropout)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

  def return_bert_model(self):
    return self.bert


def create_data_loader(comments, targets, tokenizer, max_len, batch_size, sampler=None):
      ds = SentimentDataset(
        comments = comments,
        targets = targets,
        tokenizer = tokenizer,
        max_len = max_len,
      )
    
      if sampler:
        return DataLoader(
          ds,
          batch_size=batch_size,
          sampler=sampler
        )
      else:
        return DataLoader(
          ds,
          batch_size=batch_size,
        )

def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
    ):
    model = model.train()
    model.bert = model.bert.train()
    losses = []
    correct_predictions = 0
    f_measure = 0
    for d in tqdm(data_loader, desc='training_iteration'):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        f_measure += f1(preds, targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / len(data_loader), f_measure / len(data_loader), np.mean(losses)


def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    model.bert = model.bert.eval()
    losses = []
    correct_predictions = 0
    f_measure = 0
    with torch.no_grad():
        for d in tqdm(data_loader, desc='eval_iteration'):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            f_measure += f1(preds, targets)
            losses.append(loss.item())
    return correct_predictions.double() / len(data_loader), f_measure / len(data_loader), np.mean(losses)


def full_train(model, epochs, train_data_loader, val_data_loader,
              loss_fn, optimizer, scheduler, device):
    history = defaultdict(list)
    best_accuracy = 0
    best_f1 = 0
    for epoch in tqdm(range(epochs), desc='training'):
        train_acc, train_f1, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
        )
        print(f'Train loss {train_loss} accuracy {train_acc} f1 {train_f1}')
        val_acc, val_f1, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
        )
        print(f'Val   loss {val_loss} accuracy {val_acc} f1 {val_f1}')
        print('-----------')

        if best_f1 < val_f1:
          best_f1 = val_f1
          bert = model.return_bert_model()
          bert.save_pretrained('training/bert_trained')
          torch.save(model, 'training/bert_trained_with_linear_layer.pth')

        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_loss'].append(val_loss)

    with open('training/logs_for_bert', 'wb') as f:
      pickle.dump(history, f)  

def main():
    paths = {'vocab': 'rubert_files/vocab.txt',
            'model_name': 'rubert_files',
            'config': 'rubert_files/bert_config.json'}

    params = {'epochs': 9, 'n_classes': 3,
              'max_len': 64, 'batch_size': 64,
              'lr': 5e-4, 'dropout': 0.3}

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(0)

    tokenizer = BertTokenizer.from_pretrained(paths['vocab'])
    model = SentimentClassifier(params['n_classes'], params['dropout'], len(tokenizer), 
                                    PRE_TRAINED_MODEL_NAME=paths['model_name'], PRE_TRAINED_MODEL_CONFIG=paths['config']).to(device)

    dataset = pd.read_csv('Preprocessing/for_pipeline/preprocessed_comments.csv')
    comments, targets = pd.Series([' '.join(eval(comment)) for comment in dataset['comments']]), dataset['score'].map({1:0, 2:0, 3:1, 4:1, 5:2})
    ros = RandomOverSampler()
    train_com, test_com, train_tar, test_tar = train_test_split(comments, targets, test_size=0.1, random_state=17)

    train_com, train_tar = ros.fit_resample(train_com.values.reshape(-1, 1), train_tar)
    train_com = pd.Series(train_com.reshape(-1, ))
        
    train_data_loader = create_data_loader(train_com.tolist(), train_tar.tolist(), tokenizer, params['max_len'], params['batch_size'])
    val_data_loader = create_data_loader(test_com.tolist(), test_tar.tolist(), tokenizer, params['max_len'], params['batch_size'])

    del dataset, comments, targets
    del train_com, train_tar
    del test_com, test_tar
    
    optimizer = AdamW(model.parameters(), lr=params['lr'], correct_bias=False)
    total_steps = len(train_data_loader) * params['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    full_train(model, params['epochs'], train_data_loader, val_data_loader,
                loss_fn, optimizer, scheduler, device)
    

main()
