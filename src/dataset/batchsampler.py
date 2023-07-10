from collections import defaultdict
import numpy as np
import math
import random
import copy
import torch
from .dataset import MentionEntityDataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import euclidean_dist


class MentionEntityBatchSampler(object):
    def __init__(self, model, device,
                 tokenizer,  
                 context_data,
                 pair_indices, 
                 entity_description_tokens_dict,
                 batch_size = 16,
                 max_len = 128,
                 max_doc_per_class = 250,
                 shuffle=False):

        super().__init__()
        self.shuffle = shuffle
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.context_data = context_data
        self.pair_indices = pair_indices
        self.entity_description_tokens_dict = entity_description_tokens_dict
        self.batch_size = batch_size
        self.max_doc_per_class = max_doc_per_class
         
        self.negative_pair_indices_dict = defaultdict(list)
        for idx, pair in enumerate(self.pair_indices):
            label = pair[0]
            if label == '0':
                self.negative_pair_indices_dict[pair[1]].append(idx)
        
        pair_indices = self.pair_indices
        positive_pairs = pair_indices[pair_indices[:,0] == '1']
        
        self.all_labels = list(set(positive_pairs[:,2]))        
        self.unique_cui_dict = defaultdict(list)
        self.indices_per_class = defaultdict(list)
        self.is_visited = defaultdict(list)
        self.samples_per_class = defaultdict(list)
        
        for label in self.all_labels:
            cui = label.split('-')[1]
            self.unique_cui_dict[cui].append(label) 
        
        for index, pair in enumerate(pair_indices):
            label = pair[2]
            if pair[0] == '1':
                self.indices_per_class[label].append(index)

        # using a dictionary of visited sample. 
        for label in self.indices_per_class.keys():
            self.is_visited[label] = np.array([0] * len(self.indices_per_class[label]))           
        
        n_samples = sum([len(x) if len(x) <= max_doc_per_class else max_doc_per_class for x in self.unique_cui_dict.values()])
        self.total_batch = math.ceil(n_samples/self.batch_size) 
        
        chosen_labels = []
        for cui in self.unique_cui_dict:
            list_labels = self.unique_cui_dict[cui]
            if len(list_labels) > self.max_doc_per_class:
                chosen_labels.extend(random.choices(list_labels, k = self.max_doc_per_class))
            else:
                chosen_labels.extend(list_labels)
                
        self.chosen_labels = chosen_labels
        random.shuffle(self.chosen_labels)
        self.samples_per_class = defaultdict(list)
        for label, indices in self.indices_per_class.items():
            # get array of indices unused of samples:
            is_used_array = self.is_visited[label]
            unused_indices = np.where(is_used_array == 0)[0]
            if len(unused_indices) < 1: 
                self.is_visited[label] = np.array([0] * len(self.indices_per_class[label]))
                is_used_array = self.is_visited[label]
                unused_indices = np.where(is_used_array == 0)[0]
            selected_sample_indices = np.random.choice(unused_indices,1, replace=False)
            self.samples_per_class[label] = np.array(self.indices_per_class[label])[selected_sample_indices]
            self.is_visited[label][selected_sample_indices] = 1
            
    def __iter__(self):
        # get list of positive pair indices and negative pair indices from pair indices labels
        # if self.shuffle:
        #     random.shuffle(self.all_labels)
        # random.shuffle(self.chosen_labels)
        random.shuffle(self.chosen_labels)
        current_all_labels = copy.deepcopy(self.chosen_labels)
        batch_data = []
        all_batches = []
        
        self.samples_per_class = defaultdict(list)
        for label, indices in self.indices_per_class.items():
            # get array of indices unused of samples:
            is_used_array = self.is_visited[label]
            unused_indices = np.where(is_used_array == 0)[0]
            if len(unused_indices) < 1: 
                self.is_visited[label] = np.array([0] * len(self.indices_per_class[label]))
                is_used_array = self.is_visited[label]
                unused_indices = np.where(is_used_array == 0)[0]
            selected_sample_indices = np.random.choice(unused_indices,1, replace=False)
            self.samples_per_class[label] = np.array(self.indices_per_class[label])[selected_sample_indices]
            self.is_visited[label][selected_sample_indices] = 1
        
        while len(current_all_labels) >0:
            previous_all_labels = copy.deepcopy(current_all_labels)

            for batch_index in range(0, len(previous_all_labels), self.batch_size):
                batch_labels = previous_all_labels[batch_index: batch_index + self.batch_size]  

                for label in batch_labels:
                    pair_index = self.samples_per_class[label][0]
                    anchor_index = int(self.pair_indices[pair_index][1])
                    batch_data.append(self.samples_per_class[label][0])
                    current_all_labels.remove(label)

                all_batches.append(batch_data)
                batch_data = []
        self.all_batches = all_batches
        

        for idx, pos_batch_indices in enumerate(all_batches):   #[self.start_batch_index:self.start_batch_index + self.max_num_batch]):
            batch_indices = [] 
            with torch.no_grad():
                self.model.eval()
                
                neg_batch_indices = []
                mention_input_buffer = []
                mention_position_buffer = []
                entity_input_buffer = []
                negative_indices_dict = defaultdict(list)
                all_negative_candidates = []
                mention_indices = []
                
                for i, pos_pair in enumerate(pos_batch_indices): 
                    mention_idx = self.pair_indices[pos_pair][1]
                    mention_indices.append(mention_idx)
                    mention_input, mention_position = MentionEntityDataset.generate_mention_input(int(mention_idx), self.context_data, self.max_len)
                    neg_candidates = np.array(self.negative_pair_indices_dict[mention_idx])
                    neg_candidates_pairs = self.pair_indices[neg_candidates]
                    
                    mention_input_buffer.append(mention_input)
                    mention_position_buffer.append(mention_position)
                    
                    for neg_pair in neg_candidates_pairs:
                        candidate_cui = neg_pair[2]
                        all_negative_candidates.append(candidate_cui)
                mention_input_buffer = torch.stack(mention_input_buffer).to(self.device)
                all_negative_candidates = list(set(all_negative_candidates))
                
                for i, pos_pair in enumerate(pos_batch_indices): 
                    mention_idx = self.pair_indices[pos_pair][1]
                    neg_candidates = np.array(self.negative_pair_indices_dict[mention_idx])
                    neg_candidates_pairs = self.pair_indices[neg_candidates]
                    
                    indices = []
                    for neg_pair in neg_candidates_pairs:
                        candidate_cui = neg_pair[2]
                        indices.append(all_negative_candidates.index(candidate_cui))
                    negative_indices_dict[i] = indices
                
                for candidate in all_negative_candidates:
                    entity_description_tokens = self.entity_description_tokens_dict[candidate] 
                    input_tokens= MentionEntityDataset.generate_entity_input(entity_description_tokens, self.max_len)
                    entity_input_buffer.append(input_tokens)
                entity_input_buffer = torch.stack(entity_input_buffer).to(self.device)
                
                mention_embeds = self.model.mention_encoder((mention_input_buffer,mention_position_buffer))
                entity_embeds = self.model.entity_encoder(entity_input_buffer)
                
                dists = euclidean_dist(mention_embeds, entity_embeds)
                for idx, row in enumerate(dists):
                    mention_idx = mention_indices[idx]
                    neg_candidates = np.array(self.negative_pair_indices_dict[mention_idx])
                    negative_scores = row[negative_indices_dict[idx]]
                    hardest_index = torch.topk(negative_scores, 1, dim=0, largest=False)[1].cpu()              
                    hardest_index = neg_candidates[hardest_index].flatten().tolist()
                    neg_batch_indices.extend(hardest_index)
                    
                batch_indices.extend(pos_batch_indices)
                batch_indices.extend(neg_batch_indices)

            yield batch_indices 
            
    def __len__(self):
        return self.total_batch
