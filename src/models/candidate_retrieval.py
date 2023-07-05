import numpy as np
import nltk
nltk.download('stopwords')
import math
import glob
import os
import logging
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
from tqdm import tqdm 
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from collections import defaultdict

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
LOGGER.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                        '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
LOGGER.addHandler(console)

class SparseRetrieval:
    def __init__(self, data_dir, dictionary_file, char_ngram_range = (2,5)):
        """
        Params:
            data_dir: directory of data files
            dictionary_file: file path of dictionary
        """
        super().__init__()
        self.char_tfidf = TfidfVectorizer(analyzer='char',
                                    lowercase=True,
                                    ngram_range=char_ngram_range,
                                    dtype=np.float16)

        self.word_tfidf = TfidfVectorizer(analyzer='word', 
                                     lowercase =True, 
                                     ngram_range=(1, 1),
                                     dtype=np.float16, 
                                     stop_words = stopwords.words('english'), 
                                     token_pattern='[a-zA-Z0-9_]{1,}')
        
        all_synonyms = []
        all_cuis = []
        mentions = []
        mention_cuis = []
        self.synonym_cui_dict = defaultdict(list)

        # load mentions
        data_files = glob.glob(os.path.join(data_dir, "*.txt"))

        for file in tqdm(data_files):
            with open(file, "r", encoding="utf8") as f:
                lines = f.read().split('\n')
                
                for line in lines:
                    cui = line.split('||')[0]
                    name = line.split('||')[1].lower()
                    mentions.append(name)
                    mention_cuis.append(cui)

        # load entities
        with open(dictionary_file, "r", encoding="utf8") as f:
            lines = f.read().split('\n')
            
            for line in tqdm(lines):
                names = line.split('||')[1].split(' [SEP] ')[1:]
                names = list(set([x.lower() for x in names]))
                cui = line.split('||')[0]
                all_synonyms.extend(names)
                all_cuis += len(names)*[cui]
                
                for name in names:
                    self.synonym_cui_dict[name].append(cui)

        corpus = list(set(mentions + all_synonyms))

        self.all_synonyms = all_synonyms
        self.all_cuis = all_cuis
        self.fit(corpus)
        
    def fit(self, corpus):
        LOGGER.info('Training generator...')
        self.char_tfidf.fit(tqdm(corpus))
        self.word_tfidf.fit(tqdm(corpus))
        LOGGER.info('Finish training.')

    def generate_candidates(self, data_dir, output_file, batch_size = 256, top_k_char = 56, topk = 64):
        """
        Return candidate dictionary where key is mention string and value is list of its candidate 
        """   

        # load mentions
        data_files = glob.glob(os.path.join(data_dir, "*.txt"))
        mentions = []

        for file in tqdm(data_files):
            with open(file, "r", encoding="utf8") as f:
                lines = f.read().split('\n')
                
                for line in lines:
                    cui = line.split('||')[0]
                    name = line.split('||')[1].lower()
                    mentions.append(name)

        unique_mentions = list(set(mentions))


        mention_char_sparse_matrix = self.char_tfidf.transform(unique_mentions)
        synonyms_char_sparse_matrix = self.char_tfidf.transform(self.all_synonyms)
        
        mention_word_sparse_matrix = self.word_tfidf.transform(unique_mentions)
        synonyms_word_sparse_matrix = self.word_tfidf.transform(self.all_synonyms)       

        candidates_dict =  defaultdict(list)
        max_get = 512
        count = 0

        for i in tqdm(range(0, len(unique_mentions), batch_size)):
            # compute cosine similarity
            cosine_sim = linear_kernel(mention_char_sparse_matrix[i:i+batch_size], synonyms_char_sparse_matrix)
            for row in cosine_sim:
                    
                # get index of the k highest elements
                top_k_ind = np.argpartition(row, -max_get)[-max_get:]
                top_k_ind = top_k_ind[np.argsort(row[top_k_ind])][::-1]

                list_name_candidates = [self.all_synonyms[idx] for idx in top_k_ind]
                list_cui_candidates = []
                
                for name in list_name_candidates:
                    cuis = self.synonym_cui_dict[name]
                    for cui in cuis:
                        if cui not in list_cui_candidates:
                            list_cui_candidates.append(cui)
                        if len(list_cui_candidates) == top_k_char: 
                            break
                
                candidates_dict[unique_mentions[count]] = list_cui_candidates
                count += 1

        count = 0
        for i in tqdm(range(0, len(unique_mentions), batch_size)):
            # compute cosine similarity
            cosine_sim = linear_kernel(mention_word_sparse_matrix[i:i+batch_size], synonyms_word_sparse_matrix)
            for row in cosine_sim:
                mention_string = unique_mentions[count]

                # get index of the k highest elements
                top_k_ind = np.argpartition(row, -max_get)[-max_get:]
                top_k_ind = top_k_ind[np.argsort(row[top_k_ind])][::-1]

                list_name_candidates = [self.all_synonyms[idx] for idx in top_k_ind]
                
                for name in list_name_candidates:
                    cuis = self.synonym_cui_dict[name]
                    flag = False
                    for cui in cuis:
                        if cui not in candidates_dict[mention_string]:
                            candidates_dict[mention_string].append(cui)
                        if len(candidates_dict[mention_string]) == topk: 
                            flag = True
                            break
                    if flag == True: 
                        break
                count += 1

        written_data = []
        with open(output_file, "w", encoding="utf8") as f:
            for name, list_candidates in candidates_dict.items():
                written_data.append(name + '||' + ' '.join(list_candidates))
            f.write("\n".join(written_data))

        return candidates_dict

class DenseRetrieval:
    def __init__(self, model: AutoModel, tokenizer:AutoTokenizer , dictionary_file, data_dir, max_length = 50, batch_size = 128) -> None:
        
        self.model = model
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = tokenizer

        # get all synonyms in KB
        all_cuis = []
        all_synonyms = []

        with open(dictionary_file, "r") as f:
            lines = f.read().split('\n')

            for line in tqdm(lines):
                names = line.split("||")[1].split("[SEP]")[1:]
                names = [x.lower().strip() for x in names]

                cui = line.split("||")[0]
                all_synonyms += names
                all_cuis += len(names)*[cui]

        self.all_cuis = np.array(all_cuis)

        # encode the synonyms and mention strings
        all_synonym_reps = []
        with torch.no_grad():
            for i in tqdm(np.arange(0, len(all_synonyms, batch_size))):
                toks = tokenizer.batch_encode_plus(all_synonyms[i:i+batch_size], 
                                                   padding="max_length", 
                                                   max_length=max_length, 
                                                   truncation=True,
                                                   return_tensors="pt")
                
                toks_cuda = {}
                for k, v in toks.items():
                    toks_cuda[k] = v.cuda()

                cls_rep = model(**toks_cuda)[0][:,0,:] # cls token
                all_synonym_reps.append(cls_rep.cpu().detach().numpy())
        
        all_synonym_reps = np.concatenate(all_synonym_reps, axis=0)
        
        self.all_synonyms_reps = all_synonym_reps

    def retrieve_candidate(self, data_dir, output_file, topk = 64) -> list:

        # get all mention strings in data directory
        all_mentions = []
        files = glob.glob(os.path.join(data_dir, "*.txt"))

        for file in tqdm(files):
            with open(file, "r", encoding="utf8") as f:
                lines = f.read().split("\n")
                
                for line in lines:
                    name = line.split("||").lower()
                    all_mentions.append(name)
        
        unique_mentions = list(set(all_mentions))

        all_mention_reps = []
        with torch.no_grad():
            for i in tqdm(np.arange(0, len(unique_mentions), self.batch_size)):
                toks = self.tokenizer.batch_encode_plus(unique_mentions[i:i+self.batch_size], 
                                                   padding="max_length", 
                                                   max_length=self.max_length, 
                                                   truncation=True,
                                                   return_tensors="pt")
                toks_cuda = {}
                for k,v in toks.items():
                    toks_cuda[k] = v.cuda()
                cls_rep = self.model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
                all_mention_reps.append(cls_rep.cpu().detach().numpy())
        all_mention_reps = np.concatenate(all_mention_reps, axis=0)


        # use faiss for retrieval
        
        faiss_index = faiss.IndexFlatIP(all_mention_reps.shape[-1])
        faiss_index.add(self.all_synonyms_reps)

        k = 1024
        search_batch = 16
        all_I = []

        for i in tqdm(range(0, len(all_mention_reps), search_batch)):
            _, I = faiss_index.search(all_mention_reps[i:i+search_batch], k)
            all_I.append(I)
        
        all_I = np.concatenate(all_I, axis=0)
        candidate_dict = {}
        written_data = []

        for i in tqdm(range(len(unique_mentions))):
            mention = unique_mentions[i]

            list_candidates = []
            sorted_cuis = self.all_cuis[all_I[i]]

            for cui in sorted_cuis:
                if cui not in list_candidates:
                    list_candidates.append(cui)
                    if len(list_candidates) == topk:
                        break
            
            written_data.append(mention + '||' + ' '.join(list_candidates))
            candidate_dict[mention] = list_candidates

        # write file 
        with open(output_file, "w", encoding="utf8") as f:
            f.write("\n".join(written_data))

        return candidate_dict



def main(args):

    data_dir = args.data_dir
    dictionary_file = args.dictionary_file
    output_file = args.output_file
    model_type = args.model_type

    if model_type == "sparse":
        candidate_generator = SparseRetrieval(data_dir=data_dir, dictionary_file=dictionary_file)
        _ = candidate_generator.generate_candidates(output_dir = output_file)
    else:
        tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        model =  AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        candidate_generator = DenseRetrieval(data_dir=data_dir, dictionary_file=dictionary_file)
        _ = candidate_generator.generate_candidates(output_file = output_file)
      

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                    default="./data/st21pv/train_dev",
                    help='path of data directory')
    parser.add_argument('--dictionary_file', type=str,
                    default="./data/umls/dictionary.txt",
                    help='path of input file (train/test)')                
    parser.add_argument('--output_file', type=str,
                    default="../../output/sparse_candidate.txt", 
                    help='path of output directory')
    parser.add_argument('--model_type', type=str,
                        default="sparse",
                        help='sparse or dense')

    args = parser.parse_args()
    main(args)