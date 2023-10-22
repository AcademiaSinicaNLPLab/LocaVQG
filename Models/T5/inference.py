from transformers import AutoConfig, AutoTokenizer, AutoModelWithLMHead, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, HammingDiversityLogitsProcessor
import argparse
from dataset import Dataset
from torch.utils.data import DataLoader, SequentialSampler

import os
import json
import pickle
import dotenv
import logging
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from transformers import T5Config, T5Tokenizer


from dataset import Dataset
from model import VLT5
from count_vocab import *

# from model import ImageSentenceEmbeddingNetwork

dotenv.load_dotenv()

RESULT_FOLDER = "results"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in
                #   (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

MODEL_LIST = {
    "Description2Q": [AutoConfig, AutoTokenizer, AutoModelWithLMHead],
    "Description2Q_CLIP": [T5Config, T5Tokenizer, VLT5]
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        probs = F.softmax(logits, dim=1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)

        _cumsum = sorted_probs.cumsum(1)
        mask = _cumsum < top_p
        mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], 1)
        sorted_probs = sorted_probs * mask.float()
        sorted_probs = sorted_probs / sorted_probs.sum(1, keepdim=True)

        logits.scatter_(1, sorted_indices, sorted_probs.log())

    return logits

def sample_sequence(model,
                    length, 
                    context, 
                    images=None, 
                    attention_mask=None,
                    num_samples=1, 
                    temperature=1, 
                    top_k=0, 
                    top_p=0.0, 
                    device=None, 
                    encoder=None, 
                    decode_input_ids=None,
                    penalty=None):
                    
    if device is None:
        device = torch.device("cpu")

    inputs = {
        "input_ids": context[0].repeat(num_samples, 1),
        "attention_mask": attention_mask[0].repeat(num_samples, 1),
    }
    if images is not None:
        inputs[ "images"] = images.repeat(num_samples, 1, 1, 1, 1)

    encoder_outputs = encoder(**inputs, return_dict=True)
    generated = decode_input_ids

    with torch.no_grad():
        for tok_idx in range(length):
            outputs = model(
                decoder_input_ids=generated,
                encoder_outputs=encoder_outputs,
                attention_mask=inputs['attention_mask'],
            )

            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

            if penalty != None:

                exist_tokens = F.one_hot(generated, num_classes = filtered_logits.size(1)).to(model.device)
                exist_tokens = exist_tokens.sum(dim = 1)
                exist_tokens[:, 5] = 0 # . index
                exist_tokens[:, 6] = 0 # , index
                filtered_logits = filtered_logits - penalty * exist_tokens
            
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            generated = torch.cat((generated, next_token), dim=-1)
            
    return generated


def main():
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument("--clip", action='store_true')
    parser.add_argument("--dataset", type=str, default="VIST")
    parser.add_argument("--max_desc_length", type=int, default=128)
    parser.add_argument("--max_question_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--save_name", type=str, default="t5-small-e2e-qg")
    parser.add_argument("--ablation", type=str, default="0")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--inference_type", default='all', type=str, choices=['all', 'intent', 'need', 'react'],
                        help="inference type to generate")
    parser.add_argument("--num_samples", default=5, type=int, help="No. of samples to obtain.")
    parser.add_argument("--remain_samples", default=5, type=int, help="No. of samples to obtain.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--penalty", type=float)
    parser.add_argument("--pick", action='store_true')


    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    args.device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    MODEL = MODEL_LIST["Description2Q_CLIP"] if args.clip else MODEL_LIST["Description2Q"]

    config = MODEL[0].from_pretrained(args.save_dir)

    tokenizer = MODEL[1].from_pretrained("t5-base" if args.clip else "t5-base")
    model = MODEL[2].from_pretrained(args.save_dir, config=config)

    if args.dataset == "STORY" or args.dataset == "ROC" or args.dataset == "GS":
        with open('term_vocab.pkl', 'rb') as f:
            frame_vocab = pickle.load(f)
        frame_list = [k for k in frame_vocab.word2idx.keys()]
        tokenizer.add_tokens(frame_list)
        model.resize_token_embeddings(len(tokenizer))


    model.cuda()
    model.eval()

    print(args)

    # def _prompt_to_gen(context_orig, img_feats):
    def _prompt_to_gen(
        context_orig, 
        attention_mask=None, 
        images=None,
        encoder=None,
        decode_input_ids=None,
    ):
        text_gen = [{} for _ in range(args.num_samples)]

        # set the start token to signal when to start generation
        end_token = tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]


        # begin sampling sequence starting from context_input
        out = sample_sequence(
            model=model,
            context=context_orig,
            attention_mask=attention_mask,
            images=images,
            length=args.max_question_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
            num_samples=args.num_samples,
            encoder=encoder,
            decode_input_ids=decode_input_ids,
            penalty=args.penalty,
        )

        # ensure to end the sequence with end token, and pad the rest of the sequence.
        
        out[:,-1] = end_token
        ending_idx = (out == end_token).nonzero()
        processed = []
        for i in range(ending_idx.size(0)):
            sample_idx = ending_idx[i][0].item()
            if sample_idx not in processed:
                processed.append(sample_idx)
                end_idx = ending_idx[i][1].item()
                if end_idx < out.size(1) - 1:
                    out[sample_idx,end_idx+1:] = pad_token

        end_token = tokenizer.eos_token

        # decode the sequence to text
        text_gen = [tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True, end_token=end_token) for o in out.tolist()]
        return text_gen

    # output file to store the generations

    # Get Dataset Loader
    eval_dataset = Dataset(args, tokenizer, 'test')
    # all_records = eval_dataset.records # TODO: replace this line to img ids

    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset),
                                 batch_size=args.batch_size)

    encoder = model.get_encoder()
    decode_input_ids = torch.ones((args.num_samples, 1), device=model.device, dtype=torch.long)
    decode_input_ids = decode_input_ids * model.config.decoder_start_token_id
        
    results = {}
    for data_input in tqdm.tqdm(eval_dataloader):
        data = {
            "images": data_input["images"] if args.clip else None,
            "input_ids": data_input["input_ids"].cuda(),
            "attention_mask": data_input["attention_mask"].cuda(),
        }

        # print(data['input_ids'].shape)
        # print(data['attention_mask'].shape)
        # Skip if we have processed this image, event, and inference type.
        # context = input_record['img_fn'] + input_record['event'] + input_record['inference_relation']
        # if context not in context_inputs:
        #     context_inputs.add(context)
        # else:
        #     continue

        # Now, generate the inferences and decode using original ids.
        generations = _prompt_to_gen(
            context_orig=data['input_ids'], 
            attention_mask=data['attention_mask'], 
            images=data['images'], 
            encoder=encoder,
            decode_input_ids=decode_input_ids
        )
        # for i in range(len(generations)):
        #     generations[i] = replace_names(generations[i], input_record['name2person'])
        # output_record = {k: input_record[k] for k in output_keys}
        key = data_input["ids"][0]
        if key not in results:
            results[key] = []

        results[key] += random.sample(generations, args.remain_samples)
        print(generations[:10])

    # for key in results.keys():
    #     results[key] = random.sample(results[key], args.remain_samples)

    json.dump(results, open(os.path.join(args.result_dir, f"{args.save_name}.json"),'w'), indent=2)
    print('Saved to', args.save_name)

if __name__ == '__main__':
    main()
