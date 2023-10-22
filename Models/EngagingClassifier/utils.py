import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class EngagingDataset(Dataset):
    def __init__(self, mode, data, tokenizer):
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        self.data = data
        self.len = len(self.data)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        if self.mode == "test":
            title = self.data.iloc[idx, 0]
            label_tensor = None
        else:
            title, label = self.data.iloc[idx, :].values
            label_tensor = torch.tensor(label, dtype=torch.long)
            
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(title)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)
        
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        segments_tensor = torch.tensor([0] * len_a, dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)

    # label_ids = torch.nn.functional.one_hot(label_ids, 7).squeeze(1) 
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids

def get_predictions(model, dataloader, compute_acc=False, output_logits=False):
    predictions = None
    out_logits = None
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        for data in dataloader:
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            if not compute_acc:
                tokens_tensors, segments_tensors, masks_tensors = data[:3]
                outputs = model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors)
                logits = outputs[0]
                _, pred = torch.max(logits.data, 1)
            else:
                tokens_tensors, segments_tensors, masks_tensors, labels = data[:4]
                outputs = model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors,
                                labels=labels)
                loss = outputs[0]
                logits = outputs[1]
                _, pred = torch.max(logits.data, 1)
                running_loss += loss.item()
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            if predictions is None:
                predictions = pred
                if output_logits:
                    out_logits = logits.data
            else:
                predictions = torch.cat((predictions, pred))
                if output_logits:
                    out_logits = torch.cat((out_logits, logits.data))
    
    if compute_acc:
        acc = correct / total
        return predictions, acc, running_loss
    
    if output_logits:
        return predictions, out_logits
    return predictions