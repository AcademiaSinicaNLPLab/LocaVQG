from transformers import AutoConfig, AutoTokenizer, AutoModelWithLMHead, AdamW, get_linear_schedule_with_warmup, T5Tokenizer, T5Config
import argparse
from tqdm import tqdm, trange
import os
import json
from dataset import Dataset
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import random
import numpy as np
import logging
from count_vocab import *

from model import VLT5, freeze_model_parameters

logger = logging.getLogger(__name__)

MODEL_LIST = {
    "Description2Q": [AutoConfig, AutoTokenizer, AutoModelWithLMHead],
    "Description2Q_CLIP": [T5Config, T5Tokenizer, VLT5]
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def train(args, model, teacher, train_set, eval_set):
    train_sampler = RandomSampler(train_set)
    train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True, collate_fn=train_set.collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_set)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    temperature = 2.0
    #alpha = 0.5
    alpha = 0.00005

    criterion = torch.nn.KLDivLoss(reduction='batchmean')

    for param in teacher.parameters():
        param.requires_grad = False

    train_iterator = trange(int(args.num_epochs), desc="Epoch")
    for epoch in train_iterator:
        logging.info(f"\n\n*** Starting Epoch: {epoch} ***\n\n")
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, data_input in enumerate(epoch_iterator):
            data = {
                "input_ids": data_input["input_ids"].cuda(),
                "attention_mask": data_input["attention_mask"].cuda(),
                "labels": data_input["labels"].cuda(),
            }
            if args.clip:
                data["images"] = data_input["images"]
            
            model.train()

            with torch.no_grad():
                teacher_logits = teacher(**data).logits
                teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
                #teacher_probs = torch.nn.functional.softmax(teacher_logits.view(-1, teacher_logits.size(-1)) / temperature, dim=1)

            outputs = model(**data)
            
            student_probs = torch.nn.functional.softmax(outputs.logits / temperature, dim=1)

            #print(f"KLDIV LOSS: {criterion(torch.log(student_probs), teacher_probs)} | NNLOSS: {outputs.loss}")
            
            loss = criterion(torch.log(student_probs), teacher_probs) * alpha + outputs.loss * (1 - alpha)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1

                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                if args.save_freq > 0 and global_step % args.save_freq == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.save_dir, f'checkpoint-{global_step}')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info(f"Saving model checkpoint to {output_dir}")

                    logging.info(f"Evaluate epoch ... {epoch}; iter ... {global_step}")
                    results = eval(args, model, eval_set, postfix=f'{epoch}_{global_step}')


def eval(args, model, eval_set, postfix=""):
    eval_sampler = SequentialSampler(eval_set)
    eval_dataloader = DataLoader(eval_set, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True, collate_fn=eval_set.collate_fn)

    logger.info(f"***** Running evaluation {postfix} *****")
    logger.info(f"  Num examples = {len(eval_set)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for data_input in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():

            data = {
                "input_ids": data_input["input_ids"].cuda(),
                "attention_mask": data_input["attention_mask"].cuda(),
                "labels": data_input["labels"].cuda()
            }
            if args.clip:
                data["images"] = data_input["images"]

            outputs = model(**data)
            
            lm_loss = outputs.loss
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    output_eval_file = os.path.join(args.log_dir, f"{args.save_name}.json")

    results = json.load(open(output_eval_file, "r")) if os.path.exists(output_eval_file) else {}

    if len(postfix) == 0:
        results.update({
            "perplexity": perplexity.item(),
            "eval_loss": eval_loss
        })
    else:
        results.update({
            "perplexity_{}".format(postfix): perplexity.item(),
            "loss_{}".format(postfix): eval_loss
        })

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(postfix))
        writer.write(json.dumps(results))
        writer.close()
    
    logging.info(f"saving model to {args.save_dir}/{postfix}_{eval_loss}.model")

    return results

def main(args):
    set_seed(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    TEACHER = MODEL_LIST["Description2Q"]
    MODEL = MODEL_LIST["Description2Q_CLIP"] if args.clip else MODEL_LIST["Description2Q"]

    teacher_dir = args.teacher_dir

    if args.clip:
        config = MODEL[0].from_pretrained("t5-base")

        config.use_adapter = args.use_adapter
        config.skip_adapter_layer = not args.freeze_lm_params
        config.skip_vqg_adapter_layer = True
    elif args.load_dir != "NONE":
        config = MODEL[0].from_pretrained(args.load_dir)
    else:
        config_teacher = TEACHER[0].from_pretrained(teacher_dir)
        config = MODEL[0].from_pretrained("google/t5-efficient-tiny")

    tokenizer_teacher = TEACHER[1].from_pretrained("t5-base")
    tokenizer = MODEL[1].from_pretrained("t5-base" if args.clip else "google/t5-efficient-tiny")

    sample = 'This is just a sample of a lovely text'

    if args.load_dir != "NONE":
        model = MODEL[2].from_pretrained(args.load_dir)
    else:
        teacher = TEACHER[2].from_pretrained("t5-base" if args.clip else teacher_dir, config=config_teacher)
        model = MODEL[2].from_pretrained("t5-base" if args.clip else "google/t5-efficient-tiny", config=config)

    teacher.cuda()
    model.cuda()
    
    if args.dataset == "STORY" or args.dataset == "ROC":
        with open('term_vocab.pkl', 'rb') as f:
            frame_vocab = pickle.load(f)
        frame_list = [k for k in frame_vocab.word2idx.keys()]
        tokenizer.add_tokens(frame_list)

        #model.resize_token_embeddings(len(tokenizer))

    train_set = Dataset(args, tokenizer, "train")
    eval_set = Dataset(args, tokenizer, "val")

    torch.save(args, os.path.join(args.save_dir, 'training_args.bin'))
    global_step, tr_loss = train(args, model, teacher, train_set, eval_set)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.save_dir)
    model = AutoModelWithLMHead.from_pretrained(args.save_dir)
    model.cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", action='store_true')
    parser.add_argument("--dataset", type=str, default="VIST")
    parser.add_argument("--max_desc_length", type=int, default=128)
    parser.add_argument("--max_question_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--ablation", type=str, default="0")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--load_dir", type=str, default="checkpoints")
    parser.add_argument("--teacher_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--save_name", type=str, default="t5-small-e2e-qg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", type=str, default="O1")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--use_adapter", action='store_true')


    args = parser.parse_args()

    main(args)