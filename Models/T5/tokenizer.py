import os
import json
import dotenv
from transformers import GPT2Tokenizer, T5Tokenizer, PreTrainedTokenizer
import sentencepiece as spm
import re

dotenv.load_dotenv()

IMSITU_ROLES_LIST = f'/home/VIST/data/swig/SWiG_jsons/imsitu_space.json'

class VQGTokenizer():
    def __init__(self, 
                 begin_img="<|b_img|>",
                 end_img="<|e_img|>",
                 begin_question="<|b_qn|>",
                 end_question="<|e_qn|>",
                 begin_story="<|b_st|>",
                 end_story="<|e_st|>",
                 begin_situation="<|b_situ|>",
                 end_situation="<|e_situ|>",
                 begin_verb="<|b_verb|>",
                 end_verb="<|e_verb|>",
                 begin_role="<|b_{}|>",
                 end_role="<|e_{}|>"):
        self.begin_img = begin_img
        self.end_img = end_img
        self.begin_question = begin_question
        self.end_question = end_question
        self.begin_story = begin_story
        self.end_story = end_story
        self.begin_situation = begin_situation
        self.end_situation = end_situation
        self.begin_verb = begin_verb
        self.end_verb = end_verb
        self.begin_role = begin_role
        self.end_role = end_role

        self.task_tokens = ["<vqg>", "<vist>"]
        self.tasks = {
            'vqg': "<vqg>",
            'vist': "<vist>"
        }

        with open(IMSITU_ROLES_LIST, 'r') as json_file:
            imsitu_roles = json.load(json_file)
            roles = set()
            for item in imsitu_roles['verbs']:
                if 'order' in imsitu_roles['verbs'][item]:
                    for o in imsitu_roles['verbs'][item]['order']:
                        roles.add(o)
            self.special_situ_roles_tokens = [begin_situation, end_situation]
            self.situ_role_tokens = {}
            for role in roles:
                self.special_situ_roles_tokens.append(self.begin_role.format(role))
                self.special_situ_roles_tokens.append(self.end_role.format(role))
                self.situ_role_tokens[role] = [self.begin_role.format(role), self.end_role.format(role)]

        self.special_tokens = [self.begin_question, self.end_question,
                          self.begin_img, self.end_img,
                          self.begin_situation, self.end_situation,
                          self.begin_verb, self.end_verb,
                          self.begin_story, self.end_story
                          ] + self.special_situ_roles_tokens + self.task_tokens

        self.tokens2remove = [self.begin_img, self.end_img, self.begin_situation, self.end_situation, 
                        self.begin_verb, self.end_verb, self.begin_story,
                        self.begin_question, self.unk_token, self.end_story]
        self.tokens2remove += self.special_situ_roles_tokens + self.task_tokens   
        self.tokens2remove = set(self.tokens2remove)  

    def decode(self, text, skip_special_tokens=False, end_token=None):
        if skip_special_tokens:
            for t in self.tokens2remove:
                text = text.replace(t, ' ')

        idx = text.find(end_token)
        if idx != -1:
            text = text[:idx]
        return text.strip()

class VQGGpt2Tokenizer(VQGTokenizer, GPT2Tokenizer):
    def __init__(self,
                 vocab_file,
                 merges_file,
                 errors='replace',
                 unk_token="<|endoftext|>",
                 bos_token="<|endoftext|>",
                 eos_token="<|endoftext|>",
                 begin_img="<|b_img|>",
                 end_img="<|e_img|>",
                 begin_question="<|b_qn|>",
                 end_question="<|e_qn|>",
                 begin_story="<|b_st|>",
                 end_story="<|e_st|>",
                 begin_situation="<|b_situ|>",
                 end_situation="<|e_situ|>",
                 begin_verb="<|b_verb|>",
                 end_verb="<|e_verb|>",
                 begin_role="<|b_{}|>",
                 end_role="<|e_{}|>",
                 **kwargs):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        GPT2Tokenizer.__init__(
            self,
            vocab_file,
            merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs
        )
        VQGTokenizer.__init__(
            self,
            begin_img=begin_img,
            end_img=end_img,
            begin_question=begin_question,
            end_question=end_question,
            begin_story=begin_story,
            end_story=end_story,
            begin_situation=begin_situation,
            end_situation=end_situation,
            begin_verb=begin_verb,
            end_verb=end_verb,
            begin_role=begin_role,
            end_role=end_role
        )


        # self.add_special_tokens({
        #     "additional_special_tokens": self.special_tokens
        # })


    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True, end_token=None):
        text = GPT2Tokenizer.decode(self, token_ids, False, clean_up_tokenization_spaces)
        return VQGTokenizer.decode(self, text, skip_special_tokens=skip_special_tokens, end_token=end_token)

class VLT5Tokenizer(T5Tokenizer):

    # vocab_files_names = VOCAB_FILES_NAMES
    # pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # model_input_names = ["attention_mask"]

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        vis_extra_ids=100,
        additional_special_tokens=None,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in x), additional_special_tokens)))
            if extra_tokens < extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to T5Tokenizer. "
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )

        if vis_extra_ids > 0:
            additional_special_tokens.extend(["<vis_extra_id_{}>".format(i) for i in range(vis_extra_ids)])

        PreTrainedTokenizer.__init__(
            self,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._vis_extra_ids = vis_extra_ids

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size() + self._extra_ids + self._vis_extra_ids

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(
            i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._vis_extra_ids
        elif token.startswith("<vis_extra_id_"):
            match = re.match(r"<vis_extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            if index > self.sp_model.get_piece_size() + self._extra_ids - 1:
                token = "<vis_extra_id_{}>".format(self.vocab_size - 1 - index)
            else:
                token = "<extra_id_{}>".format(self.vocab_size - self._vis_extra_ids - 1 - index)
        return token

class VQGT5Tokenizer(VQGTokenizer, VLT5Tokenizer):
    def __init__(self,
                 vocab_file,
                 unk_token="<unk>",
                 pad_token="<pad>",
                 eos_token="</s>",
                 extra_ids=100,
                 vis_extra_ids=100,
                 additional_special_tokens=None,
                 begin_img="</b_img>",
                 end_img="</e_img>",
                 begin_question="</b_qn>",
                 end_question="</e_qn>",
                 begin_story="</b_st>",
                 end_story="</e_st>",
                 begin_situation="</b_situ>",
                 end_situation="</e_situ>",
                 begin_verb="</b_verb>",
                 end_verb="</e_verb>",
                 begin_role="</b_{}>",
                 end_role="</e_{}>",
                 **kwargs):
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token

        VLT5Tokenizer.__init__(
            self,
            vocab_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            vis_extra_ids=vis_extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs
        )
        VQGTokenizer.__init__(
            self,
            begin_img=begin_img,
            end_img=end_img,
            begin_question=begin_question,
            end_question=end_question,
            begin_story=begin_story,
            end_story=end_story,
            begin_situation=begin_situation,
            end_situation=end_situation,
            begin_verb=begin_verb,
            end_verb=end_verb,
            begin_role=begin_role,
            end_role=end_role
        )

        self.tokens2remove |= set([self.pad_token])

        # self.add_special_tokens({
        #     "additional_special_tokens": self.special_tokens
        # })


    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True, end_token=None):
        text = VLT5Tokenizer.decode(self, token_ids, False, clean_up_tokenization_spaces)
        return VQGTokenizer.decode(self, text, skip_special_tokens=skip_special_tokens, end_token=end_token)