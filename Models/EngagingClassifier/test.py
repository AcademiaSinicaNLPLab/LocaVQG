import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification
from utils import EngagingDataset, create_mini_batch, get_predictions

BATCH_SIZE = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower=True)

model = torch.load('engaging_predictor.pkl')
model.to(device)

batch_text = [
        "Did you know that Pittsburgh is known as the City of Bridges? With over 400 bridges, this city has more bridges than any other city in the world, including Venice, Italy. Can you spot any of the famous bridges from our current location on PPG Street?",
        "The glass buildings you see around us are an example of Pittsburgh's thriving modern architecture. Have any of you visited or seen other cities with impressive glass buildings like these? If so, which ones were your favorites?",
        "As we drive along this city street, you might notice the diverse mix of people walking and engaging in various activities. Pittsburgh is known for its friendly and welcoming atmosphere. Have you had any memorable interactions with locals during your visit here?",
        "Pittsburgh has a rich history in the steel industry, which played a significant role in the city's development. Can you see any remnants or influences of the steel industry in the architecture or infrastructure around us?",
        "The transportation in Pittsburgh is quite diverse, as you can see with the taxi and truck driving by. Have you had a chance to try any of the public transportation options in the city, such as the buses or the historic funiculars known as the Duquesne and Monongahela Inclines?"
]

def inference(model, batch_text):
    result = []
    d = {"question": batch_text, "label": None}
    infer_sample = pd.DataFrame(d)
    # print(infer_sample)
    testset = EngagingDataset("test", infer_sample, tokenizer=tokenizer)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, 
                            collate_fn=create_mini_batch)
    pred = get_predictions(model, testloader, compute_acc=False, output_logits=False)
    result.extend(pred.tolist())
    # for i, j in zip(batch_text, pred):
        # print(f'{i} | pred: {j} \n ============================= \n')
    return result

res = inference(model, batch_text)
print(res)
