import torch 
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer

def pairwise_euclidean_dist(a:torch.Tensor, b:torch.Tensor):
    pdist = nn.PairwiseDistance(p=2)

    output = pdist(a, b)
    return output


def euclidean_dist(a: torch.Tensor, b:torch.Tensor):
    if len(a.shape) == 1:
        a = a.view((1,)+a.shape)
    if len(b.shape) == 1:
        b = b.view((1,)+b.shape)
    dist_matrix = torch.cdist(a, b, p=2)    
    return dist_matrix



def tokenize_sentence(sentence, tokenizer):
    sentence_tokens = []
    start = None
    end = None
    flag = False

    for item in sentence:
        splits = item.split('\t')
        word = splits[0]
        word_label = splits[1]

        if 'b' in word_label:
            sentence_tokens += ['[START]']
            start = len(sentence_tokens)
            flag = True
        elif 'o' in word_label and flag == True:
            end = len(sentence_tokens)
            sentence_tokens += ['[END]']
            flag = False

        tokens = tokenizer.tokenize(word)
        for token in tokens:
            sentence_tokens += [token]
    if end is None:
        end = len(sentence_tokens)
        sentence_tokens += ['[END]']
    sentence_token_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)

    return sentence_token_ids, [start, end]


def encode_mention_input(tokenizer: BertTokenizer,
                         context_token_ids: list,
                         mention_position: list,
                         max_len: int,
                         device):
    start_pos, end_pos = mention_position

    if len(context_token_ids) > max_len - 2:
        if end_pos < max_len - 2:
            context_token_ids = context_token_ids[:max_len - 2]
        else:  # truncate context_tokens on the left
            context_len = len(context_token_ids)
            # calculate number of tokens need to be cut
            n_cut = context_len - (max_len - 2)
            # carefully truncate
            if start_pos - n_cut < 0:  # case mention lies on the cut part
                # I still want to keep some context on the left
                if n_cut - 20 < 0:
                    n_cut = start_pos
                else:
                    n_cut = n_cut - 20
            start_pos -= n_cut
            end_pos -= n_cut
            context_token_ids = context_token_ids[n_cut:(max_len - 2) + n_cut]

    start_pos += 1  # add cls token
    end_pos += 1

    input_ids = [tokenizer.cls_token_id] + context_token_ids + [tokenizer.sep_token_id]
    input_len = len(input_ids)
    input_ids = input_ids[:max_len] + [tokenizer.pad_token_id] * (max_len - input_len)
    token_type_ids = [0] * max_len
    attention_mask = [1] * input_len + [0] * (max_len - input_len)

    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        token_type_ids=torch.tensor(token_type_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    ), [start_pos, end_pos]


def encode_entity_input(tokenizer: BertTokenizer,
                        entity_string: str,
                        max_len: int,
                        device):
    entity_tokens = tokenizer.tokenize(entity_string)
    entity_token_ids = tokenizer.convert_tokens_to_ids(entity_tokens)

    if len(entity_token_ids) > max_len - 2:
        entity_token_ids = entity_token_ids[:max_len - 2]

    input_ids = [tokenizer.cls_token_id] + entity_token_ids + [tokenizer.sep_token_id]
    input_len = len(input_ids)
    input_ids = input_ids[:max_len] + [0] * (max_len - input_len)
    token_type_ids = [0] * max_len
    attention_mask = [1] * input_len + [0] * (max_len - input_len)
    attention_mask = attention_mask[:max_len]

    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        token_type_ids=torch.tensor(token_type_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device)
    )