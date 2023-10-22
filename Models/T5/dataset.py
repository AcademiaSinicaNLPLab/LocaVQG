import json
from PIL import Image
import clip
import torch
import pickle
from count_vocab import *

class Dataset():
    def __init__(self, args, tokenizer, split):
        self.dataset = args.dataset
        self.ablation = args.ablation
        self.max_desc_length = args.max_desc_length
        self.max_question_length = args.max_question_length
        self.tokenizer = tokenizer
        self.split = split
        self.clip = args.clip
        if self.clip:
            _, self.clip_preprocess = clip.load("RN50", device="cuda", jit=False)

        self.load_data()

    def load_data(self):
        img_sent_map = {}

        if self.dataset == "VIST":
            for split in ["train", "val", "test"]:
                data = json.load(open(f"/home/VIST/data/VIST/VIST_coref_nos_mapped_frame_noun_{split}_list.json", "r"))
                for story in data:
                    imgs = "_".join([s["photo_flickr_id"] for s in story])
                    story = " ".join([s["text"] for s in story])
                    if imgs not in img_sent_map:
                        img_sent_map[imgs] = []
                    img_sent_map[imgs].append(story)

        data = json.load(open(f"/home/VIST/data/MVQG/{self.split}.json", "r"))
            
        if self.dataset == "DII":
            img_caption_list = []
            for split in ["train", "val", "test"]:
                cap_data = json.load(open(f"/home/VIST/data/DII/{split}.description-in-isolation.json", "r"))
                img_caption_list += cap_data["annotations"]

            img_id_index_map = {}
            for i, cap in enumerate(img_caption_list):
                img_id_index_map[cap[0]["photo_flickr_id"]] = i

            for k in data.keys():
                imgs = k.split("_")
                try:
                    cap = " ".join([img_caption_list[img_id_index_map[img]][0]["text"] for img in imgs])
                except:
                    continue

                if k not in img_sent_map:
                    img_sent_map[k] = []
                img_sent_map[k].append(cap)

        self.ids, self.description, self.description_mask, self.question = [], [], [], []
        if self.dataset == "Desc":
            for k, v in data.items():
                for x in v:
                    description = f"generate questions: {x['Summary']} </s>"
                    description = self.tokenizer([description], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')
                    question = self.tokenizer([x['Question']], padding='max_length', truncation=True, max_length=self.max_question_length, return_tensors='pt')
                    self.ids.append(k)
                    self.description.append(description["input_ids"])
                    self.description_mask.append(description["attention_mask"])
                    self.question.append(question["input_ids"])
        elif self.dataset == "Desc_wiki":
            data_question = json.load(open(f"/home/VIST/projects/MVQG_group3/crawl_wiki/attractions_en_clean_questions_{self.split}.json", 'r'))
            data_wiki = json.load(open("/home/VIST/projects/MVQG_group3/crawl_wiki/attractions_en_clean.json", 'r'))
            if self.split != "test":
                for k, v in data_question.items():
                    for x in v:
                        description = f"generate questions: {data_wiki[k]} </s>"
                        description = self.tokenizer([description], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')
                        question = self.tokenizer([x], padding='max_length', truncation=True, max_length=self.max_question_length, return_tensors='pt')
                        self.ids.append(k)
                        self.description.append(description["input_ids"])
                        self.description_mask.append(description["attention_mask"])
                        self.question.append(question["input_ids"])
            else:
                for k, v in data_question.items():
                    description = f"generate questions: {data_wiki[k]} </s>"
                    description = self.tokenizer([description], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')
                    question = self.tokenizer([v[0]], padding='max_length', truncation=True, max_length=self.max_question_length, return_tensors='pt')
                    self.ids.append(k)
                    self.description.append(description["input_ids"])
                    self.description_mask.append(description["attention_mask"])
                    self.question.append(question["input_ids"])

        elif self.dataset == "STORY":
            #input frame
            #output is story, input is golden path
            img_storyline_story = {}
            if self.split != "test":
                data = json.load(open(f"/home/VIST/data/VIST/VIST_coref_nos_mapped_frame_noun_{self.split}_list.json", "r"))

                for story in data:
                    imgs = "_".join([s["photo_flickr_id"] for s in story])
                    story_combined = " ".join([s["text"] for s in story])
                    imgs = imgs + "_" + story[0]["story_id"]
                    #golden_path = " ".join([" ".join(s["coref_mapped_seq"]) for s in story_ethan])
                    golden_path = " ".join([" ".join(s["coref_mapped_seq"]) for s in story])
                    if imgs not in img_storyline_story:
                        img_storyline_story[imgs] = []
                    img_storyline_story[imgs].append([golden_path, story_combined])
            else:
                data_pred = json.load(open(f"/home/VIST/data/VIST/VIST_coref_nos_mapped_frame_noun_{self.split}_list_pred.json", "r"))
                #data_ethan = json.load(open(f"/home/EthanHsu/commen-sense-storytelling/data/added_path_terms_EMNLP/vist_scored_terms_5_path.5term.newobj.json", "r"))
                data_ethan = json.load(open(f"/home/VIST/projects/MVQG_group3/PRVIST_allen/story_plotting/generated_storylines/pred_terms_HR_BiLSTM_plus_36_is_image_abs_position_only_6to7_repetitve_penalty_0.9_is_restrict_vocab_test_new_obj_test_reproduce.json", "r"))
                #data = json.load(open(f"/home/VIST/data/VIST/VIST_coref_nos_mapped_frame_noun_{self.split}_list.json", "r"))

                for (story_pred, story_ethan) in zip(data_pred, data_ethan):
                    imgs = "_".join([s["photo_flickr_id"] for s in story_pred])
                    story_combined = " ".join([s["text"] for s in story_pred])
                    imgs = imgs + "_" + story_ethan[0]["story_id"]
                    #golden_path = " ".join([" ".join(s["coref_mapped_seq"]) for s in story_ethan])
                    golden_path = " ".join([" ".join(s["predicted_term_seq"]) for s in story_ethan])
                    if imgs not in img_storyline_story:
                        img_storyline_story[imgs] = []
                    img_storyline_story[imgs].append([golden_path, story_combined])

            for imgs, gold_story in img_storyline_story.items():
                for golden_p, stor in gold_story:
                    description = f"generate story: {golden_p} </s>"
                    description = self.tokenizer([description], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')
                    question = self.tokenizer([stor], padding='max_length', truncation=True, max_length=self.max_question_length, return_tensors='pt')

                    self.ids.append(imgs)
                    self.description.append(description["input_ids"])
                    self.description_mask.append(description["attention_mask"])
                    self.question.append(question["input_ids"])

        elif self.dataset == "ROC":
            #input frame
            #output is story, input is golden path
            img_storyline_story = {}
            data = json.load(open(f"/home/VIST/projects/MVQG_group3/PRVIST_allen/story_reworking_work/data/ROC/ROC_{self.split}.json", "r"))
            for story in data:
                story_combined = " ".join([s for s in story["storys"]])
                golden_path = " ".join([" ".join(s) for s in story["coref_mapped_seq"]])

                description = f"generate story: {golden_path} </s>"
                description = self.tokenizer([description], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')
                question = self.tokenizer([story_combined], padding='max_length', truncation=True, max_length=self.max_question_length, return_tensors='pt')

                #self.ids.append(imgs)
                self.ids.append(1)
                self.description.append(description["input_ids"])
                self.description_mask.append(description["attention_mask"])
                self.question.append(question["input_ids"])

        elif self.dataset == "GS":
            img_storyline_story = {}
            data = json.load(open(f"/home/VIST/projects/MVQG_group3/PRVIST_allen/story_plotting/generated_storylines_gs/pred_terms_HR_BiLSTM_plus_36_is_image_abs_position_only_5to7_repetitve_penalty_0.9_is_restrict_vocab_test_new_obj_test_combined-graph_bigdetection_image2terms_area_no_duplicates_30_nicholas_5_dec_1000req.json", "r"))
            #data = json.load(open(f"/home/VIST/projects/MVQG_group3/PRVIST_allen/story_plotting/generated_storylines_gs/pred_terms_HR_BiLSTM_plus_36_is_image_abs_position_only_6to7_repetitve_penalty_0.9_is_restrict_vocab_test_new_obj_test_gs_multi_graph_bigdetection_thr_0.5_max_term_40.json", "r"))
            #data = json.load(open(f"/home/VIST/projects/MVQG_group3/PRVIST_allen/story_plotting/generated_storylines_gs/pred_terms_HR_BiLSTM_plus_36_is_image_abs_position_only_5to7_repetitve_penalty_0.9_is_restrict_vocab_test_new_obj_test_bigdetection_image2terms_area_no_duplicates_30.json", "r"))
            for story in data:
                #story_combined = " ".join([s for s in story["storys"]])
                golden_path = " ".join([" ".join(s["predicted_term_seq"]) for s in story])

                description = f"generate story: {golden_path} </s>"
                description = self.tokenizer([description], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')
                #question = self.tokenizer([story_combined], padding='max_length', truncation=True, max_length=self.max_question_length, return_tensors='pt')

                #self.ids.append(imgs)
                self.ids.append(story[0]["story_id"])
                self.description.append(description["input_ids"])
                self.description_mask.append(description["attention_mask"])
                self.question.append(description["input_ids"])
            
        elif self.dataset == "GS_image2terms":

            img_storyline_story = {}
            story_id_2_image_id = json.load(open(f"/home/VIST/projects/MVQG_group3/PRVIST_allen/story_plotting/data/Google_Street/Golden/VIST_coref_nos_mapped_frame_noun_test_list.json", "r"))
            data = json.load(open(f"/home/VIST/justin/street_view_test_terms_100.json", "r"))

            for story in story_id_2_image_id:
                #story_combined = " ".join([s for s in story["storys"]])
                flicker_list = []
                for x in story:
                    flicker_list.append(x['photo_flickr_id'])
                golden_path = " ".join([data[ids] for ids in flicker_list])

                description = f"generate story: {golden_path} </s>"
                description = self.tokenizer([description], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')
                #question = self.tokenizer([story_combined], padding='max_length', truncation=True, max_length=self.max_question_length, return_tensors='pt')

                #self.ids.append(imgs)
                self.ids.append("_".join(flicker_list))
                self.description.append(description["input_ids"])
                self.description_mask.append(description["attention_mask"])
                self.question.append(description["input_ids"])

        elif self.dataset == "GS_question":
            data = json.load(open(f"./results/GS_bigdetection_image2terms_no_dup_30_nicholas_5_dec_1000req.json"))
            #data = json.load(open(f"./results/GS_bigdetection_image2terms_no_dup_30.json"))
            for key, story in data.items():
                for i in range(5):
                    description = f"generate question: {story[i]} </s>"
                    description = self.tokenizer([description], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')

                    self.ids.append(f"{key}_{i}")
                    self.description.append(description["input_ids"])
                    self.description_mask.append(description["attention_mask"])
                    self.question.append(description["input_ids"])

        elif self.dataset == "StreetviewFull":
            data = json.load(open(f"/home/VIST/projects/MVQG_GS/streetview_data/streetview_caption/Streetview_CaptionFull_{self.split}.json"))
            #data = json.load(open(f"./results/GS_bigdetection_image2terms_no_dup_30.json"))
            for key, story in data.items():
                description = f"generate 5 question: {story['Caption']} </s>"
                description = self.tokenizer([description], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')
                question = self.tokenizer([story["Question"]], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')

                self.ids.append(f"{key}")
                self.description.append(description["input_ids"])
                self.description_mask.append(description["attention_mask"])
                self.question.append(question["input_ids"])
        
        elif self.dataset == "StreetviewFilter":
            if self.ablation == "0":
                data = json.load(open(f"/home/VIST/projects/MVQG_GS/streetview_data/streetview_caption/StreetviewFilter_Caption_{self.split}_important.json"))
            else:
                data = json.load(open(f"/home/VIST/projects/MVQG_GS/streetview_data/streetview_caption_ablation/StreetviewFilter_Caption{self.ablation}_{self.split}.json"))
            #data = json.load(open(f"/home/VIST/projects/MVQG_GS/streetview_data/streetview_caption/StreetviewFilter_Caption_pick.json"))
            for key, story in data.items():
                description = "generate question: "
                for j in range(5):
                    description += f"{story['Caption'][j]} </s>"
                description = self.tokenizer([description], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')
                
                for i in range(5):
                    question = self.tokenizer([story["Question"][i]], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')

                    self.ids.append(f"{key}_{i}")
                    self.description.append(description["input_ids"])
                    self.description_mask.append(description["attention_mask"])
                    self.question.append(question["input_ids"])
        
        elif self.dataset == "StreetviewFilterFull":
            data = json.load(open(f"/home/VIST/projects/MVQG_GS/streetview_data/streetview_caption/StreetviewFilter_CaptionFull_{self.split}.json"))
            #data = json.load(open(f"./results/GS_bigdetection_image2terms_no_dup_30.json"))
            for key, story in data.items():
                description = f"generate 5 question: {story['Caption']} </s>"
                description = self.tokenizer([description], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')
                question = self.tokenizer([story["Question"]], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')

                self.ids.append(f"{key}")
                self.description.append(description["input_ids"])
                self.description_mask.append(description["attention_mask"])
                self.question.append(question["input_ids"])
        
        elif self.dataset == "Streetview":
            data = json.load(open(f"/home/VIST/projects/MVQG_GS/streetview_data/streetview_caption/Streetview_Caption_{self.split}.json"))
            #data = json.load(open(f"/home/VIST/projects/MVQG_GS/streetview_data/streetview_caption/StreetviewFilter_Caption_pick.json"))
            for key, story in data.items():
                description = "generate question: "
                for j in range(5):
                    description += f"{story['Caption'][j]} </s>"
                description = self.tokenizer([description], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')
                
                for i in range(5):
                    question = self.tokenizer([story["Question"][i]], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')

                    self.ids.append(f"{key}_{i}")
                    self.description.append(description["input_ids"])
                    self.description_mask.append(description["attention_mask"])
                    self.question.append(question["input_ids"])

        else:
            for imgs, questions in data.items():
                if self.dataset == "None":
                    descriptions = [""]
                else:
                    try:
                        descriptions = img_sent_map[imgs]
                    except:
                        continue

                for description in descriptions:
                    description = f"generate questions: {description} </s>"
                    description = self.tokenizer([description], padding='max_length', truncation=True, max_length=self.max_desc_length, return_tensors='pt')

                    _description, _description_mask, _question = [], [], []
                    for question in questions:
                        question = self.tokenizer([question['Question']], padding='max_length', truncation=True, max_length=self.max_question_length, return_tensors='pt')

                        _description.append(description["input_ids"][0])
                        _description_mask.append(description["attention_mask"][0])
                        _question.append(question["input_ids"][0])

                    self.ids.append(imgs)
                    self.description.append(torch.stack(_description))
                    self.description_mask.append(torch.stack(_description_mask))
                    self.question.append(torch.stack(_question))
            


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.clip:
            images = self.ids[idx].split("_")
            images = [self.clip_preprocess(Image.open(f"/home/VIST/data/vist_images/{i}.jpg")).unsqueeze(0).cuda() for i in images]
            images = torch.cat(images, dim=0)

        
        return_dict = {
            "ids": self.ids[idx],
            "input_ids": self.description[idx],
            "attention_mask": self.description_mask[idx],
            "labels": self.question[idx],
        }

        if self.clip:
            return_dict['images'] = images.repeat(self.description[idx].shape[0], 1, 1, 1, 1)

        return return_dict

    def collate_fn(self, data):
        return_dict = {
            'input_ids':      torch.cat([d["input_ids"] for d in data]),
            'attention_mask': torch.cat([d["attention_mask"] for d in data]),
            'labels':         torch.cat([d["labels"] for d in data]),
        }

        if self.clip:
            return_dict['images'] = torch.cat([d["images"] for d in data]),

        return_dict['ids'] = []
        for d in data:
            for i in range(d['input_ids'].shape[0]):
                return_dict['ids'].append(d['ids'])

        return return_dict
