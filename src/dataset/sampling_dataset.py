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
        if len(mention_indices) > self.max_cluster_size:  # choose randomly
            mention_indices = random.sample(mention_indices, self.max_cluster_size)

        # random choose the anchor point
        anchor_index = random.choice(mention_indices)
        anchor_mention_input, anchor_mention_position = encode_mention_input(tokenizer=self.tokenizer,
                                                                             context_token_ids=
                                                                             self.all_context_data[anchor_index][0],
                                                                             mention_position=
                                                                             self.all_context_data[anchor_index][1],
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

        cluster_size = mention_positions.size(0)
        # masking input model for equal cluster size
        # for key, value in mention_inputs.items():
        #     mask = torch.zeros((self.max_cluster_size, self.max_mention_len), device=self.device, dtype=torch.long)
        #     if mask.size(0) > value.size(0):
        #         mask[:value.size(0)] = value
        #     else:
        #         mask = value[:mask.size(0)]
        #     mention_inputs[key] = mask
        # # mask mention position
        # mask = torch.zeros((self.max_cluster_size, 2), device=self.device, dtype=torch.long)
        # if mention_positions.size(0) < mask.size(0):
        #     mask[:mention_positions.size(0)] = mention_positions
        # else:
        #     mask = mention_positions[:mask.size(0)]
        # mention_positions = mask

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
            cluster_size=cluster_size,
            negative_candidate_inputs=negative_candidate_inputs,
            positive_entity_input=positive_entity_input,
            negative_candidates=candidates,
            positive_entity=label,
        )


class MyCollate:
    def __call__(self, batch):
        # sort the batch by source length in decreased order
        batch = sorted(batch, key=lambda x: x['cluster_size'], reverse=True)
        max_cluster_size = batch[0]['cluster_size']

        batch_dict = {'cluster_size': torch.tensor([x['cluster_size'] for x in batch]),
                      'anchor_mention_input': default_collate([x['anchor_mention_input'] for x in batch]),
                      'anchor_mention_position': torch.stack([x['anchor_mention_position'] for x in batch]),
                      'mention_inputs': [x['mention_inputs'] for x in batch],
                      'mention_positions': [x['mention_positions'] for x in batch],
                      'negative_candidate_inputs': default_collate([x['negative_candidate_inputs'] for x in batch]),
                      'positive_entity_input': default_collate([x['positive_entity_input'] for x in batch]),
                      'negative_candidates': [x['negative_candidates'] for x in batch],
                      'positive_entity': [x['positive_entity'] for x in batch]
                      }

        # print(batch_dict['mention_inputs'])

        # masking input model for equal cluster size
        for idx, item in enumerate(batch_dict['mention_inputs']):
            for key, value in item.items():
                mask = torch.zeros((max_cluster_size, value.size(-1)), device=value.device, dtype=torch.long)
                if mask.size(0) > value.size(0):
                    mask[:value.size(0)] = value
                else:
                    mask = value[:mask.size(0)]
                item[key] = mask
            batch_dict['mention_inputs'][idx] = item
        batch_dict['mention_inputs'] = default_collate(batch_dict['mention_inputs'])
        # mask mention position
        for idx, item in enumerate(batch_dict['mention_positions']):
            mask = torch.zeros((max_cluster_size, 2), device=item.device, dtype=torch.long)
            mask[:item.size(0)] = item
            batch_dict['mention_positions'][idx] = mask
        batch_dict['mention_positions'] = torch.stack(batch_dict['mention_positions'])
        return batch_dict
