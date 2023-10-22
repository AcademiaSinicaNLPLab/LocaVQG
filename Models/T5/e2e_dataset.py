import json
from PIL import Image
import clip
import torch
import pickle
import os

def load_img_feat(img_feat_dir):
    img_feat_paths = os.listdir(img_feat_dir)
    img_feat_paths = [f"{img_feat_dir}/{p}" for p in img_feat_paths]
    
    valid_img_id_list, verbs, objects = [], [], []
    for path in img_feat_paths:
        with open(path, 'rb') as f:
            info = pickle.load(f)
        valid_img_id_list.append(path.split('/')[-1][:-4])
        verbs.append(info['verb'])
        objects.append(info['objects'])

    return valid_img_id_list, verbs, objects


class Dataset():
    def __init__(self, args, tokenizer, split):
        self.max_desc_length = args.max_desc_length
        self.max_question_length = args.max_question_length
        self.tokenizer = tokenizer
        self.split = split
        _, self.clip_preprocess = clip.load("RN50", device="cuda", jit=False)

        self._load_img_feat()

        self.load_data()

    def _load_img_feat(self):
        valid_img_id_list, verbs, objects = [], [], []
        for folder in ["train", "val", "test"]:
            img_feat_dir = f"/home/VIST/data/swig/global_features/{folder}/vist"
            id_list, v, o = load_img_feat(img_feat_dir)
            valid_img_id_list += id_list
            verbs += v
            objects += o

        valid_img_id_dict = dict(zip(valid_img_id_list, range(len(valid_img_id_list))))

        self.image_feat = {
            "valid_img_id_list": valid_img_id_list, 
            "valid_img_id_dict": valid_img_id_dict,
            "verbs":   verbs, 
            "objects": objects, 
        }

    def _img_to_token(self, verb, nouns, roles):
        text_instance = [self.tokenizer.begin_verb] + self.tokenizer.tokenize(verb) + [self.tokenizer.end_verb]
        
        # <role> noun <role>
        for i, noun in enumerate(nouns):
            if noun != '':
                tag = self.tokenizer.situ_role_tokens[roles[i]]
                text_instance += [tag[0]] + self.tokenizer.tokenize(noun) + [tag[1]]

        return text_instance

    def _pad_text_token(self, tokens, max_seq_len):
        pad_len = max_seq_len - len(tokens)
        assert pad_len >= 0, print(len(tokens))

        mask = [1] * len(tokens) + [0] * pad_len

        tokens += [self.tokenizer.pad_token] * pad_len
        
        return tokens, mask

    def load_data(self):

        with open(f'/home/VIST/data/swig/SWiG_jsons/imsitu_space.json', 'r') as f:
            mapping = json.load(f)
            noun_mapping = mapping['nouns']
            verb_role_mapping = mapping['verbs']

        data = json.load(open(f"/home/VIST/data/VQGPlus/{self.split}.json", "r"))
            
        task_token = self.tokenizer.tokenize(f'generate question:')

        self.ids, self.description, self.description_mask, self.question = [], [], [], []
        for imgs, questions in data.items():
            text_tokens = []
            img_ids = imgs.split("_")
            for i, img_id in enumerate(img_ids):
                img_index = self.image_feat["valid_img_id_dict"][img_id]
                verb      = self.image_feat["verbs"][img_index]
                roles     = verb_role_mapping[verb]['order']

                noun_id_list = self.image_feat["objects"][img_index]
                nouns = []
                for noun_id in noun_id_list:
                    nouns.append(noun_mapping[noun_id]['gloss'][0] if noun_id != '' and noun_id != 'Pad' and noun_id != 'oov' else '')

                sub_text_tokens = \
                    self._img_to_token(
                        verb, 
                        nouns, 
                        roles
                    )
                
                text_tokens += sub_text_tokens

            text_tokens = task_token + text_tokens + [self.tokenizer.eos_token]
            padded_tokens, padded_mask = self._pad_text_token(text_tokens, self.max_desc_length)
            padded_tokens = self.tokenizer.convert_tokens_to_ids(padded_tokens)

            _description, _description_mask, _question = [], [], []
            
            if self.split == 'test':
                questions = questions[:1]
                
            for question in questions:
                question = self.tokenizer([question['Question']], padding='max_length', truncation=True, max_length=self.max_question_length, return_tensors='pt')

                _description.append(torch.tensor(padded_tokens))
                _description_mask.append(torch.tensor(padded_mask))
                _question.append(question["input_ids"][0])

            self.ids.append(imgs)
            self.description.append(torch.stack(_description))
            self.description_mask.append(torch.stack(_description_mask))
            self.question.append(torch.stack(_question))
                

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        images = self.ids[idx].split("_")
        images = [self.clip_preprocess(Image.open(f"/home/VIST/data/vist_images/{i}.jpg")).unsqueeze(0).cuda() for i in images]
        images = torch.cat(images, dim=0)

        
        return {
            "ids": self.ids[idx],
            "images": images.repeat(self.description[idx].shape[0], 1, 1, 1, 1),
            "input_ids": self.description[idx],
            "attention_mask": self.description_mask[idx],
            "labels": self.question[idx],
        }

    def collate_fn(self, data):
        return_dict = {
            'images':         torch.cat([d["images"] for d in data]),
            'input_ids':      torch.cat([d["input_ids"] for d in data]),
            'attention_mask': torch.cat([d["attention_mask"] for d in data]),
            'labels':         torch.cat([d["labels"] for d in data]),
        }


        return_dict['ids'] = []
        for d in data:
            for i in range(d['input_ids'].shape[0]):
                return_dict['ids'].append(d['ids'])

        return return_dict