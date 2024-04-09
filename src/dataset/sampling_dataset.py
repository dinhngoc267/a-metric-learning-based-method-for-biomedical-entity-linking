import glob
import os
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from torch.utils.data.dataloader import default_collate
from src.refactor_utils import (
    encode_entity_input,
    encode_mention_input,
    tokenize_sentence
)


class SamplingDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 dictionary_file: str,
                 candidate_file: str,
                 tokenizer: BertTokenizer,
                 device: torch.device,
                 logger,
                 max_mention_len=128,
                 max_entity_len=128):

        super().__init__()

        logger.info('Initialize Dataset...')
        self.tokenizer = tokenizer
        self.device = device

        self.max_mention_len = max_mention_len
        self.max_entity_len = max_entity_len

        self.all_context_data = []
        self.all_mentions = []
        self.all_labels = []
        all_docids = []

        self.entity_dict = defaultdict()
        self.candidate_dict = defaultdict()

        # load context data
        logger.info('Load Query Mentions...')
        context_files = glob.glob(os.path.join(data_dir, "*.context"))

        for context_file in tqdm(context_files):
            with open(context_file, "r", encoding='utf-8') as f:
                list_sents = f.read().split('\n\n')
                list_sents = list_sents[:-1]
                for sent in list_sents:
                    sent_token_ids, mention_pos = tokenize_sentence(sent.lower().split('\n'), tokenizer)
                    self.all_context_data.append((sent_token_ids, mention_pos))

            mention_file = os.path.join(data_dir, os.path.basename(context_file).replace(".context", ".txt"))
            with open(mention_file, "r", encoding='utf-8') as f:
                lines = f.read().split('\n')
                for line in lines:
                    line = line.split('||')
                    mention = line[1].lower()
                    label = line[0]
                    self.all_mentions.append(mention)
                    self.all_labels.append(label)
                    id = os.path.basename(mention_file).replace(".txt", "")
                    all_docids.append(id)

        deleted_indices = []
        for idx, (context, _) in enumerate(self.all_context_data):
            if len(context) < 10:
                deleted_indices.append(idx)
        logger.info('Deleted {} mention with context less than 10 tokens'.format(len(deleted_indices)))

        self.all_context_data = [x for i, x in enumerate(self.all_context_data) if i not in deleted_indices]
        self.all_mentions = [x for i, x in enumerate(self.all_mentions) if i not in deleted_indices]
        self.all_labels = [x for i, x in enumerate(self.all_labels) if i not in deleted_indices]
        docids = [x for i, x in enumerate(all_docids) if i not in deleted_indices]

        # Down-sampling based on the number of labels appear across the documents instead of the number of mentions.
        # E.g: multiple mentions have the same label in a doc will be counted as one.
        labels_docid_dict = defaultdict(set)  # key: label; value: list of documents which have that label
        for idx, label in enumerate(self.all_labels):
            labels_docid_dict[label].add(docids[idx])

        labels_docid_dict = defaultdict(set)  # key: label; value: list of documents which have that label
        for idx, label in enumerate(self.all_labels):
            labels_docid_dict[label].add(all_docids[idx])

        # down-sampling to max 200 docs of each label.
        for label, list_docids in labels_docid_dict.items():
            if len(list_docids) > 200:
                labels_docid_dict[label] = random.sample(list(list_docids), 200)  # sample without replacement
            else:
                labels_docid_dict[label] = list(list_docids)

        self.data = defaultdict(list)
        for idx in range(len(self.all_mentions)):
            if all_docids[idx] in labels_docid_dict[self.all_labels[idx]]:
                key = self.all_labels[idx] + '-' + all_docids[idx]
                self.data[key].append(idx)

        # get the max possible of a cluster size
        self.max_cluster_size = 1
        for value in self.data.values():
            if len(value) > self.max_cluster_size:
                self.max_cluster_size = len(value)
        # print(self.max_cluster_size)
        self.max_cluster_size = 20
        self.labels = list(self.data.keys())

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
                self.candidate_dict[mention] = candidates

        logger.info('Finish Initializing Dataset!')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item_label = self.labels[idx]
        label = item_label.split('-')[0]

        # get all indices of mentions have the same label in a document
        mention_indices = self.data[item_label]
        if len(mention_indices) > self.max_cluster_size: # choose randomly
            mention_indices = random.sample(mention_indices, self.max_cluster_size)

        # random choose the anchor point
        anchor_index = random.choice(mention_indices)
        anchor_mention_input, anchor_mention_position = encode_mention_input(tokenizer=self.tokenizer,
                                                                             context_token_ids=self.all_context_data[anchor_index][0],
                                                                             mention_position=self.all_context_data[anchor_index][1],
                                                                             max_len=self.max_mention_len,
                                                                             device=self.device)

        mention_positions = []
        mention_inputs = []
        for idx in mention_indices:
            mention_input, mention_position = encode_mention_input(tokenizer=self.tokenizer,
                                                                   context_token_ids=self.all_context_data[idx][0],
                                                                   mention_position=self.all_context_data[idx][1],
                                                                   max_len=self.max_mention_len,
                                                                   device=self.device)
            mention_positions.append(mention_position)
            mention_inputs.append(mention_input)

        mention_inputs = default_collate(mention_inputs)
        mention_positions = torch.tensor(mention_positions).to(self.device)

        # masking input model for equal cluster size
        for key, value in mention_inputs.items():
            mask = torch.zeros((self.max_cluster_size, self.max_mention_len), device=self.device, dtype=torch.long)
            if mask.size(0) > value.size(0):
                mask[:value.size(0)] = value
            else:
                mask = value[:mask.size(0)]
            mention_inputs[key] = mask
        # mask mention position
        mask = torch.zeros((self.max_cluster_size, 2), device=self.device, dtype=torch.long)
        if mention_positions.size(0) < mask.size(0):
            mask[:mention_positions.size(0)] = mention_positions
        else:
            mask = mention_positions[:mask.size(0)]
        mention_positions = mask

        # get all the candidates of the

        anchor_mention_string = self.all_mentions[anchor_index]
        candidates = [candidate for candidate in self.candidate_dict[anchor_mention_string] if candidate != label][:64]

        negative_candidate_inputs = []
        for candidate in candidates:
            candidate_input = encode_entity_input(tokenizer=self.tokenizer,
                                                  entity_string=self.entity_dict[candidate],
                                                  max_len=self.max_entity_len,
                                                  device=self.device)
            negative_candidate_inputs.append(candidate_input)

        negative_candidate_inputs = default_collate(negative_candidate_inputs)

        positive_entity_input = encode_entity_input(tokenizer=self.tokenizer,
                                                    entity_string=self.entity_dict[label],
                                                    max_len=self.max_entity_len,
                                                    device=self.device)

        return dict(
            anchor_mention_input=anchor_mention_input,
            anchor_mention_position=torch.tensor(anchor_mention_position),
            mention_inputs=mention_inputs,
            mention_positions=mention_positions,
            negative_candidate_inputs=negative_candidate_inputs,
            positive_entity_input=positive_entity_input,
            negative_candidates=candidates,
            positive_entity=label,
        )


