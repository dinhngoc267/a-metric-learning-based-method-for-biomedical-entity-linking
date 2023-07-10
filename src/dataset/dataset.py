from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm
import glob
import os
import numpy as np
import torch

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
    sentence_tokens = tokenizer.convert_tokens_to_ids(sentence_tokens)

    return sentence_tokens, [start, end]
    

class MentionEntityDataset(Dataset):
    def __init__(self, data_dir, 
               dictionary_file,
               candidate_file, 
               tokenizer, max_len=128, min_len = 10):

        super().__init__()

        self.tokenizer = tokenizer
        self.context_data = []
        self.mentions = []
        self.labels = []
        self.entity_description_dict = {}
        self.candidates_dict = {}
        self.pair_indices = []
        
        mention_docid_dict = defaultdict(dict)
        docids = []
        # load context data
        files = glob.glob(os.path.join(data_dir, "*.context"))
        label_dictionary = defaultdict(list)
        mention_index = 0
        for file_name in tqdm(files):
            with open(file_name, "r", encoding='utf-8') as f:
                list_sents = f.read().split('\n\n')
                list_sents = list_sents[:-1]
                for sent in list_sents:
                    sent_token_ids, mention_pos = tokenize_sentence(sent.lower().split('\n'), tokenizer)
                    self.context_data.append((sent_token_ids, mention_pos))
            mention_file = os.path.join(data_dir, os.path.basename(file_name).replace(".context", ".txt"))
            with open(mention_file, "r", encoding='utf-8') as f:
                lines = f.read().split('\n')
                for line in lines: 
                    line = line.split('||')
                    mention = line[1].lower()
                    label = line[0]
                    self.mentions.append(mention)
                    self.labels.append(label)
                    id = os.path.basename(mention_file).replace(".txt", "")
                    docids.append(id)

        
        deleted_indices = []
        for idx, (context, _) in enumerate(self.context_data):
            if len(context) < min_len:
                deleted_indices.append(idx)
        print('Deleted {} mention with context less than 10 characters'.format(len(deleted_indices)))
        
        self.context_data = [x for i, x in enumerate(self.context_data) if i not in deleted_indices]
        self.mentions = [x for i, x in enumerate(self.mentions) if i not in deleted_indices]
        self.labels = [x for i, x in enumerate(self.labels) if i not in deleted_indices]
        docids = [x for i, x in enumerate(docids) if i not in deleted_indices]

        for idx, label in enumerate(self.labels):
            doc_id = docids[idx]     
            if label not in  mention_docid_dict[doc_id]:
                mention_docid_dict[doc_id][label] = [idx]
            else:
                mention_docid_dict[doc_id][label].append(idx)
                    
        mention_indices_dict = defaultdict(dict)
        
        for id, dictionary in mention_docid_dict.items():
            for label, list_indices in dictionary.items():
                for index in list_indices:
                    mention_indices_dict[index] = {'in': list_indices, 'out': []}
                    for label_2, list_indices_2 in dictionary.items():
                        if label_2 != label:
                            mention_indices_dict[index]['out'].append(list_indices_2)
        
        self.mention_indices_dict = mention_indices_dict
    
        
        for idx,label in enumerate(self.labels):
            doc_id = docids[idx]
            key = str(doc_id) + "-" + str(label) 
            label_dictionary[key].append(idx)            
        
        # load dictionary
        with open(dictionary_file, "r", encoding='utf-8') as f:
            lines = f.read().split('\n')

            for line in lines:
                line = line.split('||')
                cui = line[0]
                description = line[1]
                self.entity_description_dict[cui] = description

        # load candidates
        all_cuis = []
        with open(candidate_file, "r", encoding='utf-8') as f:
            lines = f.read().split('\n')
            for line in lines:
                line = line.split('||')
                self.candidates_dict[line[0]] = list(set(line[1].split(' ')))
                all_cuis += self.candidates_dict[line[0]] 
        
        for label in self.labels:
            all_cuis.append(label)
            
        all_cuis = list(set(all_cuis))
        self.entity_description_tokens_dict = {}
        for cui in tqdm(all_cuis):
            entity_description_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.entity_description_dict[cui]))
            self.entity_description_tokens_dict[cui] = entity_description_token


        # create list pair of index for training 
        anchor_negative_pairs = []
        for i in range(len(self.mentions)):
            hardest_negative_cuis = [cui for cui in self.candidates_dict[self.mentions[i].lower()] if cui != self.labels[i]]
            for cui in hardest_negative_cuis:
                anchor_negative_pairs.append((0, i, cui))

        anchor_positive_pairs = []
        for label, mention_indices in label_dictionary.items():
            for mention_index in mention_indices:
                anchor_positive_pairs.append((1,mention_index, label))

        self.pair_indices = anchor_negative_pairs + anchor_positive_pairs
        self.pair_indices = np.array(self.pair_indices)
        self.max_len = max_len

        print('There are {} positive pairs and {} negative pairs\n'.format(len(anchor_positive_pairs), len(anchor_negative_pairs)))

    def __len__(self):
        return len(self.pair_indices)
    
    def __getitem__(self, idx):
        pair = self.pair_indices[idx]
        
        label = int(pair[0])
        mention_index = int(pair[1])
        
        if '-' in pair[2]:
            cui = pair[2].split('-')[1]
        else:
            cui = pair[2]
        entity_description_tokens = self.entity_description_tokens_dict[cui] #self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(entity_description))
        mention_input, mention_position  = MentionEntityDataset.generate_mention_input(mention_index, self.context_data, self.max_len)
        entity_input = MentionEntityDataset.generate_entity_input(entity_description_tokens, self.max_len)
        
        
        return (mention_input, mention_position), entity_input, mention_index # [positive_cluster_input], [negative_clusters_input]
    


    @staticmethod
    def generate_mention_input(data_index,  
                               context_data,
                               max_len=128):
        """

        """
        sentence_tokens,  [start_mention_token, end_mention_token]  = context_data[data_index]
        
        if len(sentence_tokens) > max_len - 2:
            if end_mention_token < max_len - 2:
                sentence_tokens = sentence_tokens[:max_len -2]
            else:
                length_before = len(sentence_tokens)
                #sentence_tokens = sentence_tokens[length_before - (max_len-2):]
                n_cut = length_before - (max_len-2) # len(sentence_tokens)
                # as I cut tokens before mention tokens so position of mention token change. 
                if start_mention_token - n_cut < 0:
                    if n_cut -20 < 0:
                        n_cut = n_cut - (n_cut - start_mention_token)
                    else:
                        n_cut -= 20    
                start_mention_token  -= n_cut
                end_mention_token -= n_cut
                sentence_tokens = sentence_tokens[n_cut: (max_len-2) + n_cut]
                
        start_mention_token += 1 # add cls token 
        end_mention_token += 1
        
        input_ids = tokenizer.convert_tokens_to_ids(['[CLS]']) + sentence_tokens + tokenizer.convert_tokens_to_ids(['[SEP]'])

        input_len = len(input_ids)
        input_ids = input_ids[:max_len] + [0]*(max_len-input_len)
        token_type_ids = [0]*max_len
        attention_mask = [1]*input_len + [0]*(max_len-input_len)
        attention_mask = attention_mask[:max_len]       
        
        return torch.tensor([input_ids, token_type_ids, attention_mask]), np.array([start_mention_token, end_mention_token])

    def generate_entity_input(entity_description_tokens: list, 
                              max_len=128):
        """

        """

        if len(entity_description_tokens) > max_len-2:
            entity_description_tokens = entity_description_tokens[:max_len-2]


        input_ids = tokenizer.convert_tokens_to_ids(['[CLS]']) + entity_description_tokens + tokenizer.convert_tokens_to_ids(['[SEP]'])

        input_len = len(input_ids)
        input_ids = input_ids[:max_len] + [0]*(max_len-input_len)
        token_type_ids = [0]*max_len
        attention_mask = [1]*input_len + [0]*(max_len-input_len)
        attention_mask = attention_mask[:max_len]

        entity_input = torch.tensor([input_ids, token_type_ids, attention_mask])

        return entity_input
