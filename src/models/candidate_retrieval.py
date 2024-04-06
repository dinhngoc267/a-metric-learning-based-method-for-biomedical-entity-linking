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
import pickle


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
LOGGER.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                        '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
LOGGER.addHandler(console)


class SparseRetrieval:
    def __init__(self, data_dir, dictionary_file, char_ngram_range=(2, 5)):
        """
        Params:
            data_dir: directory of data files
            dictionary_file: file path of dictionary
        """
        super().__init__()
        self.char_tfidf = TfidfVectorizer(analyzer='char',
                                          lowercase=True,
                                          # max_features=2000000,
                                          ngram_range=char_ngram_range,
                                          dtype=np.float32)

        self.word_tfidf = TfidfVectorizer(analyzer='word',
                                          lowercase=True,
                                          # max_features=1000000,
                                          ngram_range=(1, 1),
                                          dtype=np.float32,
                                          stop_words=stopwords.words('english'),
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
                all_cuis += len(names) * [cui]

                for name in names:
                    self.synonym_cui_dict[name].append(cui)

        corpus = list(set(mentions + all_synonyms))
        self.all_synonyms = all_synonyms
        self.all_cuis = all_cuis
        self.fit(corpus)

    def fit(self, corpus):
        LOGGER.info('Training generator...')
        LOGGER.info('Training Character-based TFIDF')
        self.char_tfidf.fit(tqdm(corpus))
        LOGGER.info('Training Word-based TFIDF')
        self.word_tfidf.fit(tqdm(corpus))
        LOGGER.info('Finish training.')

    def generate_candidates(self, data_dir, output_file, batch_size=512, top_k_char=46, top_k=65):
        """
        Return candidate dictionary where key is mention string and value is list of its candidates
        """

        # load mentions
        data_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
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

        word_candidates_dict = defaultdict(list)
        char_candidates_dict = defaultdict(list)
        max_get = 512
        count = 0

        for i in tqdm(range(0, len(unique_mentions), batch_size)):
            # compute cosine similarity
            cosine_sim = linear_kernel(mention_char_sparse_matrix[i:i + batch_size], synonyms_char_sparse_matrix)
            for row in cosine_sim:

                # get index of the k highest elements
                top_k_ind = np.argpartition(row, -max_get)[-max_get:]
                top_k_ind = top_k_ind[np.argsort(row[top_k_ind])][::-1]

                list_name_candidates = [self.all_synonyms[idx] for idx in top_k_ind]
                list_cui_candidates = []

                for name in list_name_candidates:
                    cuis = self.synonym_cui_dict[name]
                    flag = False
                    for cui in cuis:
                        if cui not in list_cui_candidates:
                            list_cui_candidates.append(cui)
                        if len(list_cui_candidates) == top_k:
                            flag = True
                            break
                    if flag == True:
                        break

                char_candidates_dict[unique_mentions[count]] = list_cui_candidates
                count += 1

        count = 0
        for i in tqdm(range(0, len(unique_mentions), batch_size)):
            # compute cosine similarity
            cosine_sim = linear_kernel(mention_word_sparse_matrix[i:i + batch_size], synonyms_word_sparse_matrix)
            for row in cosine_sim:
                mention_string = unique_mentions[count]

                # get index of the k highest elements
                top_k_ind = np.argpartition(row, -max_get)[-max_get:]
                top_k_ind = top_k_ind[np.argsort(row[top_k_ind])][::-1]

                list_name_candidates = [self.all_synonyms[idx] for idx in top_k_ind]
                list_cui_candidates = []
                for name in list_name_candidates:
                    cuis = self.synonym_cui_dict[name]
                    flag = False
                    for cui in cuis:
                        if cui not in list_cui_candidates:  # candidates_dict[mention_string]:
                            list_cui_candidates.append(cui)
                        if len(list_cui_candidates) == top_k:
                            flag = True
                            break
                    if flag == True:
                        break

                word_candidates_dict[mention_string] = list_cui_candidates
                count += 1

        recall_by_k = {}
        for k in range(0, 64):

            final_candidate_dict = {}
            for mention, c_candidates in char_candidates_dict.items():
                candidates = c_candidates[:k] + word_candidates_dict[mention][k:64]
                final_candidate_dict[mention] = candidates

            recall = self.cal_recall(data_dir, final_candidate_dict)
            recall_by_k[k] = recall

        top_k_char = int(max(recall_by_k, key=recall_by_k.get))

        final_candidate_dict = {}
        for mention, c_candidates in char_candidates_dict.items():
            candidates = c_candidates[:top_k_char] + word_candidates_dict[mention][top_k_char:65]
            final_candidate_dict[mention] = candidates

        written_data = []
        with open(output_file, "w", encoding="utf8") as f:
            for name, list_candidates in final_candidate_dict.items():
                written_data.append(name + '||' + ' '.join(list_candidates))
            f.write("\n".join(written_data))

        with open("top_k_char.txt", "w") as f:
            f.write(str(top_k_char) + " " + str(recall_by_k[top_k_char]))

        pickle.dump(word_candidates_dict, open("word_candidates_dict.pk", "wb"))
        pickle.dump(char_candidates_dict, open("char_candidates_dict.pk", "wb"))

        return final_candidate_dict, word_candidates_dict, char_candidates_dict

    from collections import defaultdict
    def cal_recall(self, data_dir, candidate_dict):
        files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))

        n = 0
        correct = 0
        for file in files:
            with open(file, "r", encoding='utf-8') as f:
                lines = f.read().split('\n')
                for line in lines:
                    line = line.split('||')
                    mention = line[1].lower()
                    label = line[0]
                    n += 1
                    if label in candidate_dict[mention][:64]:
                        correct += 1

        return correct / n


class DenseRetrieval:
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer, dictionary_file, data_dir, max_length=50,
                 batch_size=128) -> None:

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
                all_cuis += len(names) * [cui]

        self.all_cuis = np.array(all_cuis)

        # encode the synonyms and mention strings
        all_synonym_reps = []
        with torch.no_grad():
            for i in tqdm(np.arange(0, len(all_synonyms, batch_size))):
                toks = tokenizer.batch_encode_plus(all_synonyms[i:i + batch_size],
                                                   padding="max_length",
                                                   max_length=max_length,
                                                   truncation=True,
                                                   return_tensors="pt")

                toks_cuda = {}
                for k, v in toks.items():
                    toks_cuda[k] = v.cuda()

                cls_rep = model(**toks_cuda)[0][:, 0, :]  # cls token
                all_synonym_reps.append(cls_rep.cpu().detach().numpy())

        all_synonym_reps = np.concatenate(all_synonym_reps, axis=0)

        self.all_synonyms_reps = all_synonym_reps

    def retrieve_candidate(self, data_dir, output_file, topk=64):

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
                toks = self.tokenizer.batch_encode_plus(unique_mentions[i:i + self.batch_size],
                                                        padding="max_length",
                                                        max_length=self.max_length,
                                                        truncation=True,
                                                        return_tensors="pt")
                toks_cuda = {}
                for k, v in toks.items():
                    toks_cuda[k] = v.cuda()
                cls_rep = self.model(**toks_cuda)[0][:, 0, :]  # use CLS representation as the embedding
                all_mention_reps.append(cls_rep.cpu().detach().numpy())
        all_mention_reps = np.concatenate(all_mention_reps, axis=0)

        # use faiss for retrieval

        faiss_index = faiss.IndexFlatIP(all_mention_reps.shape[-1])
        faiss_index.add(self.all_synonyms_reps)

        k = 1024
        search_batch = 16
        all_I = []

        for i in tqdm(range(0, len(all_mention_reps), search_batch)):
            _, I = faiss_index.search(all_mention_reps[i:i + search_batch], k)
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


if __name__ == "__main__":
    corpus_dir = 'data/processed/MedMention/st21pv/train_dev'
    data_dir = 'data/processed/MedMention/st21pv/train_dev'
    dictionary_file = 'data/processed/MedMention/umls/dictionary.txt'
    output_file = 'output/candidates/st21_train_dev/sparse_candidate.txt'
    model_type = 'sparse'

    if model_type == "sparse":
        candidate_generator = SparseRetrieval(data_dir=corpus_dir, dictionary_file=dictionary_file)
        candidates_dict, word_candidates, char_candidates = candidate_generator.generate_candidates(data_dir=data_dir,
                                                                                                    output_file=output_file)
    else:
        tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        candidate_generator = DenseRetrieval(data_dir=data_dir, dictionary_file=dictionary_file)
        _ = candidate_generator.generate_candidates(data_dir=data_dir, output_file=output_file)

