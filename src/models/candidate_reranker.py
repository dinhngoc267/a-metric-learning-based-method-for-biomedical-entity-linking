import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel


class MentionEncoder(nn.Module):
    def __init__(self, bert_base, tokenizer):
        super().__init__()
        self.base_model = BertModel.from_pretrained(bert_base) 
        self.base_model.resize_token_embeddings(len(tokenizer))

    def forward(self, mentions):
        inputs, mention_positions = mentions
        input_ids, token_type_ids, attention_mask = inputs[:,0,:], inputs[:,1,:], inputs[:,2,:]

        outputs = self.base_model(input_ids = input_ids, token_type_ids= token_type_ids,attention_mask=attention_mask)
        outputs = outputs[0]
        
        
        mention_embeddings = []
        for idx, embedding in enumerate(outputs):
            start, end = mention_positions[idx]
            start_embedding = embedding[start]
            end_embedding = embedding[end]
            mention_embeddings.append(torch.mean(torch.stack([start_embedding, end_embedding]), dim=0))                                      
        mention_embeddings = torch.stack(mention_embeddings)
        
        return mention_embeddings

    
class EntityEncoder(nn.Module):
    def __init__(self, bert_base, tokenizer):

        super().__init__()
        self.base_model = BertModel.from_pretrained(bert_base)
        self.base_model.resize_token_embeddings(len(tokenizer))

    def forward(self, x):
        input_ids, token_type_ids, attention_mask = x[:,0,:], x[:,1,:], x[:,2,:]

        outputs = self.base_model(input_ids = input_ids, token_type_ids = token_type_ids,attention_mask = attention_mask)
        outputs = outputs[0]

        
        cls_embeddings = outputs[:,0,:]
        
        return cls_embeddings
    
class MentionEntityDualEncoder(nn.Module):
    def __init__(self, entity_encoder, mention_encoder, embed_dims= 768, margin = 1.2):
        super().__init__()

        self.entity_encoder = entity_encoder
        self.mention_encoder = mention_encoder
        self.margin = margin
        
    def compute_loss(self, anchor_mention_inputs, 
                     positive_entity_inputs, 
                     negative_entity_inputs,
                     anchor_mention_index,
                     positive_mention_inputs_clusters, 
                     all_negative_mention_inputs_clusters):
        
        positive_enity_embeds = self.entity_encoder(positive_entity_inputs)
        negative_entity_embeds = self.entity_encoder(negative_entity_inputs)

        positive_mention_cluster_prototypes = []

        for i, cluster in enumerate(positive_mention_inputs_clusters):
            mention_embeds = self.mention_encoder(cluster)
            prototype = torch.mean(mention_embeds, dim=0)
            positive_mention_cluster_prototypes.append(prototype)       
                
        positive_mention_cluster_prototypes = torch.stack(positive_mention_cluster_prototypes)

        positive_entity_scores  = self.pairwise_euclidean_squared_dist(positive_mention_cluster_prototypes,positive_enity_embeds)
        negative_entity_scores = self.pairwise_euclidean_squared_dist(positive_mention_cluster_prototypes, negative_entity_embeds)
        
        
        loss = F.relu(positive_entity_scores - negative_entity_scores + self.margin) 

        return torch.mean(loss), torch.mean(negative_entity_scores).item(), torch.mean(positive_entity_scores).item()#, radius
    
    def pairwise_euclidean_dist(self, a:torch.Tensor, b:torch.Tensor):
        pdist = nn.PairwiseDistance(p=2)

        output = pdist(a, b)
        return output  
    
    def euclidean_dist(self, a: torch.Tensor, b:torch.Tensor):
        if len(a.shape) == 1:
            a = a.view((1,)+a.shape)
        if len(b.shape) == 1:
            b = b.view((1,)+b.shape)        
        dist_matrix = torch.cdist(a, b, p=2)
        return dist_matrix
    
    
    def compute_cluster_prototype(self,mention_inputs):
        mention_embeds = self.mention_encoder(mention_inputs)
        prototype = torch.mean(mention_embeds, dim=0)
        
        return prototype
        



    