import glob
import os
import torch
import random
import logging
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from torch.utils.data.dataloader import default_collate
from lightning.pytorch import Trainer, seed_everything
from src.refactor_utils import (
    encode_entity_input,
    encode_mention_input,
    tokenize_sentence
)

class ValidationDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 tokenizer: BertTokenizer,
                 dictionary_file: str,
                 candidate_file: str,
                 device: torch.device,
                 logger,
                 max_mention_len=128,
                 max_entity_len=128):

        self.logger = logger
        self.tokenizer = tokenizer
        self.device = device
        self.max_mention_len = max_mention_len
        self.max_entity_len = max_entity_len

        all_context_data = []
        all_mentions = []
        all_labels = []
        candidate_dict = defaultdict(list)
        self.entity_dict = defaultdict(str)
        # load context data
        logger.info('Load Query Mentions...')
        context_files = glob.glob(os.path.join(data_dir, "*.context"))

        for context_file in tqdm(context_files):
            with open(context_file, "r", encoding='utf-8') as f:
                list_sents = f.read().split('\n\n')
                list_sents = list_sents[:-1]
                for sent in list_sents:
                    sent_token_ids, mention_pos = tokenize_sentence(sent.lower().split('\n'), tokenizer)
                    all_context_data.append((sent_token_ids, mention_pos))

            mention_file = os.path.join(data_dir, os.path.basename(context_file).replace(".context", ".txt"))
            with open(mention_file, "r", encoding='utf-8') as f:
                lines = f.read().split('\n')
                for line in lines:
                    line = line.split('||')
                    mention = line[1].lower()
                    label = line[0]
                    all_mentions.append(mention)
                    all_labels.append(label)

        # Load dictionary file.
        logger.info('Load Entity Dictionary...')
        with open(dictionary_file, "r", encoding='utf-8') as f:
            lines = f.read().split('\n')
            for line in lines:
                line = line.split('||')
                cui = line[0]
                description = line[1]
                self.entity_dict[cui] = description

        # Load candidates
        logger.info('Load Candidates...')
        with open(candidate_file, "r", encoding='utf-8') as f:
            lines = f.read().split('\n')
            for line in lines:
                line = line.split('||')
                mention = line[0]
                candidates = line[1].split(' ')
                candidate_dict[mention] = candidates

        logger.info('Finish Initializing Dataset!')

        self.data = []

        for idx in range(len(all_labels)):
            self.data.append(
                {
                    'context_data': all_context_data[idx],
                    'label': all_labels[idx],
                    'mention': all_mentions[idx],
                    'candidates': candidate_dict[all_mentions[idx]]
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        mention_input, mention_position = encode_mention_input(tokenizer=self.tokenizer,
                                                               context_token_ids=item['context_data'][0],
                                                               mention_position=item['context_data'][1],
                                                               max_len=self.max_mention_len,
                                                               device=self.device)

        mention_position = torch.tensor(mention_position, device=self.device)

        label = item['label']
        candidates = item['candidates'][:64]
        candidate_inputs = []

        for candidate in candidates:
            candidate_input = encode_entity_input(tokenizer=self.tokenizer,
                                                  entity_string=self.entity_dict[candidate],
                                                  max_len=self.max_entity_len,
                                                  device=self.device)

            candidate_inputs.append(candidate_input)

        candidate_inputs = default_collate(candidate_inputs)

        return dict(
            mention_input=mention_input,
            mention_position=mention_position,
            label=label,
            candidate_inputs=candidate_inputs,
            candidate_labels=candidates
        )