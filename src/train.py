import torch
from models.candidate_reranker import MentionEntityDualEncoder, EntityEncoder, MentionEncoder
from models.srn import Radius
from dataset.batchsampler import MentionEntityBatchSampler
from dataset.dataset import MentionEntityDataset
from torch.utils.data import DataLoader
from tqdm import tqdm 
from transformers import BertTokenizer
from utils import pairwise_euclidean_dist, euclidean_dist
import yaml
import argparse

def train(args):

    data_dir = args.data_dir
    dictionary_file = args.dictionary_file
    candidate_file = args.candidate_file
    bert_base = args.bert_base
    # batch_size = args.batch_size
    # epochs = args.epochs

    # read config file
    with open(".\configs\srn.yaml") as f:
        srn_config = yaml.safe_load(f)

    with open(".\configs\candidate_reranker.yaml") as f:
        reranker_config = yaml.safe_load(f)

    tokenizer = BertTokenizer.from_pretrained(bert_base)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[START]', '[END]']})

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    mention_entity_dataset = MentionEntityDataset(data_dir=data_dir,
                                                  dictionary_file=dictionary_file,
                                                  candidate_file=candidate_file,
                                                  tokenizer=tokenizer)
    


    mention_entity_dual_encoder = MentionEntityDualEncoder(entity_encoder = EntityEncoder(bert_base = bert_base, tokenizer=tokenizer) , mention_encoder = MentionEncoder(bert_base = bert_base, tokenizer=tokenizer))
    mention_entity_dual_encoder.to(device)

    optimizer = torch.optim.Adam(mention_entity_dual_encoder.parameters(), lr=reranker_config['optimizer']['learning_rate'])

    radius = Radius(init_value=10.0)
    radius.to(device)
    radius_optimizer = torch.optim.Adam(radius.parameters(), lr=srn_config['optimizer']['learning_rate'])

    mention_entity_dual_encoder.train()

    batch_size = reranker_config['train']['batch_size']
    epochs = reranker_config['train']['epochs']

    batch_sampler = MentionEntityBatchSampler(model=mention_entity_dual_encoder,
                                            device=device,
                                            batch_size = batch_size,
                                            tokenizer=tokenizer,
                                            context_data=mention_entity_dataset.context_data,
                                            pair_indices = mention_entity_dataset.pair_indices,
                                            entity_description_tokens_dict = mention_entity_dataset.entity_description_tokens_dict, shuffle=False)
    data_loader = DataLoader(dataset= mention_entity_dataset,batch_sampler= batch_sampler, shuffle=False)


    for i in range(epochs):
        print("-----EPOCH: {}-------".format(str(i)))


        with tqdm(data_loader, unit="batch") as tepoch:
            mention_entity_dual_encoder.train()
            for index, (mention_input, entity_input, mention_index) in enumerate(tepoch):
                mention_input, mention_position = mention_input

                if mention_input.shape[0] == batch_size*2:
                    n = batch_size
                else:
                    n = entity_input.shape[0]//2
                    
                mention_index = mention_index[:n] 
                mention_input = mention_input[:n].to(device)
                mention_position = mention_position[:n]
                positive_entity_inputs = entity_input[:n].to(device)
                negative_entity_inputs = entity_input[n:].to(device)

                anchor_mention_index = []
                with torch.no_grad():
                    mention_entity_dual_encoder.eval()
                    positive_mention_inputs_clusters = []

                    devide = "\n"
                    batch_dists = []
                    for i, index in enumerate(mention_index):
                        positive_mention_inputs = []
                        positive_mention_positions = []
                        dists = []
                        anchor_mention_embed = mention_entity_dual_encoder.mention_encoder((mention_input[i].view((1,) + mention_input[i].shape), mention_position[i].reshape((1,) + mention_position[i].shape)))

                        for positive_index in mention_entity_dataset.mention_indices_dict[index.item()]['in']:
                            positive_mention_input, positive_mention_position =  MentionEntityDataset.generate_mention_input(positive_index, mention_entity_dataset.context_data, mention_entity_dataset.max_len)
                            positive_mention_embed = mention_entity_dual_encoder.mention_encoder((positive_mention_input.view((1,) + positive_mention_input.shape).to(device), positive_mention_position.reshape((1,) + positive_mention_position.shape)))

                            dist = pairwise_euclidean_dist(anchor_mention_embed, positive_mention_embed)

                            if dist  < radius.radius + 1:                              
                                if dist > 1e-2:
                                    dists.append(dist)
                                positive_mention_inputs.append(positive_mention_input)
                                positive_mention_positions.append(positive_mention_position)
                        if len(dists) > 0:
                            dists = torch.max(torch.stack(dists))
                            batch_dists.append(dists)
                        positive_mention_inputs = torch.stack(positive_mention_inputs).to(device)
                        positive_mention_inputs_clusters.append((positive_mention_inputs, positive_mention_positions))
                        devide = devide+ str(positive_mention_inputs.shape[0]) + "/" +  str(len(mention_entity_dataset.mention_indices_dict[index.item()]['in'])) + " "


                mention_entity_dual_encoder.train()
                loss, dist_neg, dist_pos = mention_entity_dual_encoder.compute_loss((mention_input, mention_position), 
                                                                        positive_entity_inputs, 
                                                                        negative_entity_inputs,
                                                                        anchor_mention_index,
                                                                        positive_mention_inputs_clusters,
                                                                        None)

                if len(batch_dists) > 0:
                    batch_dists = torch.mean(torch.stack(batch_dists)).to(device)

                    radius_loss = radius.loss(batch_dists)     #torch.sum(r[r > 1e-2 ])/torch.sum(r > 1e-2))
                    radius_optimizer.zero_grad()
                    radius_loss.backward()
                    radius_optimizer.step()

                optimizer.zero_grad()
                loss.backward()
                sum_loss += loss.item()
                optimizer.step()
                current_batch_index += 1
                devide += "\n"
                tepoch.set_postfix(loss=round(sum_loss/current_batch_index, 3), dist_neg = round(dist_neg,3), dist_pos = round(dist_pos,3), 
                                radius = radius.radius.item(), dists =  batch_dists)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default="./data/processed/st21pv/train_dev")

    parser.add_argument('--dictionary_file', type=str, default='./data/umls/dictionary.txt')

    parser.add_argument('--candidate_file', type=str, default='./output/candidates/st21pv/train_dev_candidates.txt')

    parser.add_argument('--bert_base', type=str, default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

    args = parser.parse_args()

    train(args)

