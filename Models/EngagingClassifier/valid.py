import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification
from utils import EngagingDataset, create_mini_batch, get_predictions

EPOCHS = 10
BATCH_SIZE = 128
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower=True)

title_data = pd.read_csv('engaging_data.csv')
title_data = title_data.dropna()
train, test = train_test_split(title_data, test_size=0.1, random_state=412)
valid, test = train_test_split(test, test_size=0.5, random_state=412)
print(title_data.shape, train.shape, valid.shape, test.shape)

trainset = EngagingDataset("train", train, tokenizer=tokenizer)
validset = EngagingDataset("valid", valid, tokenizer=tokenizer)
testset = EngagingDataset("valid", test, tokenizer=tokenizer)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch)
validloader = DataLoader(validset, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch)

title, label = trainset.data.iloc[5].values
tokens_tensor, segments_tensor, label_tensor = trainset[5]
tokens = tokenizer.convert_ids_to_tokens(tokens_tensor.tolist())
combined_text = " ".join(tokens)
print(combined_text)

#model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

model = torch.load('engaging_predictor.pkl')
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)

model.eval()

_, test_acc, test_loss = get_predictions(model, testloader, compute_acc=True)

print(test_acc)
print(test_loss)