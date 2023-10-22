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
testset = EngagingDataset("test", test, tokenizer=tokenizer)

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

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

#model = torch.load('engaging_predictor.pkl')
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)


min_valid_loss = 10000
for epoch in range(EPOCHS):
    running_loss = 0.0
    for data in trainloader:
        
        tokens_tensors, segments_tensors, \
        masks_tensors, labels = [t.to(device) for t in data]

        optimizer.zero_grad()
        
        outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors, 
                        labels=labels)

        loss = outputs[0]
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    _, train_acc, _ = get_predictions(model, trainloader, compute_acc=True)
    _, valid_acc, valid_loss = get_predictions(model, validloader, compute_acc=True)
    #_, test_acc, test_loss = get_predictions(model, testloader, compute_acc=True)
    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        torch.save(model, 'engaging_predictor_collin.pkl')
    print('[epoch %d] train loss: %.3f, train acc: %.3f, valid loss: %.3f, valid acc: %.3f' %
          (epoch + 1, running_loss, train_acc, valid_loss, valid_acc))