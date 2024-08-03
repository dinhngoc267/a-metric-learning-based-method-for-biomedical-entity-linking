import torch
import random
import logging
import pytorch_lightning as pl
import numpy as np
from transformers import BertTokenizer, BertModel
from lightning.pytorch import Trainer, seed_everything
from src.refactor_utils import euclidean_dist, pairwise_euclidean_dist
from src.metric import LinkingAccuracy


class MentionEncoder(pl.LightningModule):
    def __init__(self, bert_base, tokenizer):
        super().__init__()
        self.base_model = BertModel.from_pretrained(bert_base)
        self.base_model.resize_token_embeddings(len(tokenizer))

    def forward(self, mention_inputs, mention_positions):
        outputs = self.base_model(**mention_inputs)
        outputs = outputs[0]

        mention_embeddings = torch.mean(outputs[(torch.arange(0, outputs.size(0)).unsqueeze(1),
                                                 mention_positions)],
                                        dim=1)

        return mention_embeddings


class EntityEncoder(pl.LightningModule):
    def __init__(self, bert_base, tokenizer):
        super().__init__()
        self.base_model = BertModel.from_pretrained(bert_base)
        self.base_model.resize_token_embeddings(len(tokenizer))

    def forward(self, entity_inputs):
        outputs = self.base_model(**entity_inputs)
        outputs = outputs[0]

        cls_embeddings = outputs[:, 0, :]

        return cls_embeddings


class ReRanker(pl.LightningModule):
    def __init__(self, bert_base, tokenizer, init_cluster_radius=10.0):
        super().__init__()

        self.entity_encoder = EntityEncoder(bert_base=bert_base,
                                            tokenizer=tokenizer)

        self.mention_encoder = MentionEncoder(bert_base=bert_base,
                                              tokenizer=tokenizer)

        self.mean_radius = torch.nn.Parameter(torch.tensor(init_cluster_radius))
        self.std_radius = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("cluster_radius", self.mean_radius)

        self.automatic_optimization = False
        self.linking_metric = LinkingAccuracy()

    def __form_cluster(self, batch, batch_idx):
        # in the original proposal, the cluster is formed by radius neighbors

        batch_anchor_mention_input = batch['anchor_mention_input']
        batch_anchor_mention_position = batch['anchor_mention_position']
        batch_mention_inputs = batch['mention_inputs']
        batch_mention_positions = batch['mention_positions']

        batch_size = batch_mention_positions.size(0)
        max_cluster_size = batch_mention_positions.size(1)

        with torch.no_grad():
            self.eval()

            # get the embedding of anchor point
            batch_anchor_embedding = self.mention_encoder(batch_anchor_mention_input,
                                                          batch_anchor_mention_position)
            # [batch_size, embedding_dims]

            # get embedding of other mentions
            batch_mention_inputs = {key: value.view(-1, value.size(-1)) for key, value in batch_mention_inputs.items()}
            batch_mention_positions = batch_mention_positions.view(-1, batch_mention_positions.size(
                -1))  # [batch_size*max_cluster_size, 2]

            batch_mention_embeddings = self.mention_encoder(batch_mention_inputs,
                                                            batch_mention_positions)  # (batch_size*max_cluster_size, embedding_dim)

            # convert to original shape
            batch_mention_positions = batch_mention_positions.view(batch_size,
                                                                   -1,
                                                                   batch_mention_positions.size(-1))
            # (batch_size, max_cluster_size, 2)
            batch_mention_embeddings = batch_mention_embeddings.view(batch_size,
                                                                     -1,
                                                                     batch_mention_embeddings.size(-1))
            # (batch_size, max_cluster_size, embedding_dims)

            # calculate distance between anchor and other data points
            dist = torch.cdist(batch_anchor_embedding.unsqueeze(1),
                               batch_mention_embeddings,
                               p=2.0).squeeze(1)

            # mask out padded rows
            mask = ~batch_mention_positions.any(dim=-1)
            dist[mask] = 1e+3

            valid_mask = dist < (self.mean_radius + self.std_radius * 1.5)
            valid_dist = dist*valid_mask


            batch_cluster_mention_positions = batch_mention_positions.view(batch_size,
                                                                           -1,
                                                                           batch_mention_positions.size(-1)) * valid_mask.unsqueeze(-1)

            radius, _ = valid_dist.topk(1, largest=True, dim=-1)
            radius = radius.flatten().detach()
            radius = radius[radius > 1e-3]

            if len(radius) > 1:
                std_r, mean_r = torch.std_mean(radius)
                return batch_cluster_mention_positions, std_r, mean_r
            else:
                return batch_cluster_mention_positions, self.std_radius, self.mean_radius

    def __hard_negative_mining(self, batch):
        with torch.no_grad():
            self.eval()

            # sampling hard negative sample
            batch_anchor_mention_input = batch['anchor_mention_input']
            batch_anchor_mention_position = batch['anchor_mention_position']
            batch_negative_candidate_inputs = batch['negative_candidate_inputs']
            batch_size = batch_anchor_mention_position.size(0)

            batch_negative_candidates = batch['negative_candidates']
            # batch_negative_candidates = [[batch_negative_candidates[j][i] for j in range(64)] for i in range(batch_size)]
            all_negative_candidates = [batch_negative_candidates[i][j] for i in range(len(batch_negative_candidates))
                                       for j in range(len(batch_negative_candidates[0]))]
            all_negative_candidates = np.array(all_negative_candidates)
            batch_positive_entity = np.array(batch['positive_entity'])

            # compute anchor embedding
            batch_anchor_embedding = self.mention_encoder(batch_anchor_mention_input,
                                                          batch_anchor_mention_position)

            # flatten batch_negative_candidate_inputs into 2D -> [batch_size * max_candidates, input_dim]
            batch_negative_candidate_inputs = {key: value.view(-1, value.size(-1))
                                               for key, value in batch_negative_candidate_inputs.items()}

            # compute candidate embeddings
            batch_candidate_embeddings = self.entity_encoder(batch_negative_candidate_inputs)  # (batch_size * max_candidates, dim)


            # resize to original shape to get hard sample only from negative candidates
            # batch_candidate_embeddings = batch_candidate_embeddings.view(batch_size, -1, batch_candidate_embeddings.size(-1))
            # batch_negative_candidate_inputs = {key:value.view(batch_size, -1, value.size(-1)) for key, value in batch_negative_candidate_inputs.items()}
            # dist = torch.cdist(batch_anchor_embedding[:,None], batch_candidate_embeddings, p=2.0).squeeze(1) # batch_size, max_candidates
            # _, topi = dist.topk(1, largest=False, dim=-1)
            # topi = topi.detach()
            # batch_hard_negative_inputs = {key: value[(torch.arange(value.size(0)).unsqueeze(-1), topi)].squeeze(1) for key, value in batch_negative_c

            # calculate distance between anchor and in batch candidates (more negative candidates will be considered)
            dist = euclidean_dist(batch_anchor_embedding,
                                  batch_candidate_embeddings)  # [batch_size, batch_size*max_candidates]

            # mask out the case that negative candidates of other samples in batch is positive label of an sample
            mask = batch_positive_entity[:, None] == all_negative_candidates
            mask = torch.tensor(mask, device=self.device) * 1e+3
            dist = dist + mask

            _, topi = dist.topk(1, largest=False, dim=-1)
            topi = topi.squeeze(-1).detach()
            batch_hard_negative_inputs = {key: value[topi] for key, value in batch_negative_candidate_inputs.items()}
            return batch_hard_negative_inputs

    def training_step(self, batch, batch_idx):

        dual_encoder_opt, cluster_opt = self.optimizers()

        batch_mention_inputs = batch['mention_inputs']
        batch_positive_entity_input = batch['positive_entity_input']

        # form cluster
        batch_cluster_mention_positions, std_radius, mean_radius = self.__form_cluster(batch, batch_idx)
        # hard negative mining
        batch_hard_negative_inputs = self.__hard_negative_mining(batch)

        self.train()

        # compute cluster prototype
        batch_size = batch_cluster_mention_positions.size(0)

        batch_mention_inputs = {key: value.view(-1, value.size(-1)) for key, value in batch_mention_inputs.items()}
        batch_cluster_mention_positions = batch_cluster_mention_positions.view(-1,
                                                                               batch_cluster_mention_positions.size(-1))

        batch_cluster_embeddings = self.mention_encoder(batch_mention_inputs, batch_cluster_mention_positions)

        # mask out padded row
        mask = torch.sum(batch_cluster_mention_positions, dim=-1) # [batch_size * max_cluster_size,1]
        mask[mask != 0] = 1
        # batch_cluster_embeddings = mask.unsqueeze(-1) * batch_cluster_embeddings
        batch_cluster_embeddings[mask==0] = 0

        # calculate real size of cluster
        mask = mask.view(batch_size, -1)
        batch_cluster_size = torch.sum(mask, dim=-1, keepdim=True)

        batch_mention_embeddings = batch_cluster_embeddings.view(batch_size, -1, batch_cluster_embeddings.size(-1))

        batch_prototype_embedding = torch.sum(batch_mention_embeddings, dim=1) / batch_cluster_size

        positive_entity_embedding = self.entity_encoder(batch_positive_entity_input)
        negative_entity_embedding = self.entity_encoder(batch_hard_negative_inputs)

        positive_dist = pairwise_euclidean_dist(batch_prototype_embedding, positive_entity_embedding)
        negative_dist = pairwise_euclidean_dist(batch_prototype_embedding, negative_entity_embedding)

        dual_encoder_opt.zero_grad()
        dual_encoder_loss = self.prototype_triplet_loss(positive_dist, negative_dist, margin=1.15)
        self.manual_backward(dual_encoder_loss)
        dual_encoder_opt.step()

        cluster_opt.zero_grad()
        cluster_loss = self.mse_loss(self.mean_radius, self.std_radius, mean_radius, std_radius)
        self.manual_backward(cluster_loss)
        cluster_opt.step()

        self.log_dict(
            {'train_loss': dual_encoder_loss,
             'negative_dist': torch.mean(negative_dist),
             'positive_dist': torch.mean(positive_dist),
             'mean_radius': self.mean_radius,
             'std_radius': self.std_radius,
             'real_mean_radius': mean_radius,
             'real_std_radius': std_radius
             },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size
        )

    def validation_step(self, batch, batch_idx):

        batch_mention_input = batch['mention_input']
        batch_mention_position = batch['mention_position']
        batch_label = batch['label']
        batch_candidate_inputs = batch['candidate_inputs']
        tmp = batch['candidate_labels']

        batch_size = batch_mention_position.size(0)
        batch_candidate_labels = [[tmp[j][i] for j in range(len(tmp))] for i in range(len(tmp[0]))]

        mention_embeddings = self.mention_encoder(batch_mention_input, batch_mention_position)

        batch_candidate_inputs = {key: value.view(-1, value.size(-1)) for key, value in batch_candidate_inputs.items()}
        batch_candidate_embeddings = self.entity_encoder(batch_candidate_inputs)

        # reshape to original
        batch_candidate_embeddings = batch_candidate_embeddings.view(batch_size, -1,
                                                                     batch_candidate_embeddings.size(-1))

        # calculate distance
        dist = torch.cdist(mention_embeddings[:, None], batch_candidate_embeddings, p=2.0).squeeze(
            1)  # (batch_size, max_candidates)

        _, topi = dist.topk(1, largest=False, dim=-1)  # (batch_isze, 1)
        topi = topi.flatten().detach()
        preds = [batch_candidate_labels[i][topi[i]] for i in range(len(topi))]

        accuracy = self.linking_metric(preds, batch_label)

        self.log_dict({
            'val_accuracy': accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size
        )

    def prototype_triplet_loss(self, positive_scores, negative_scores, margin):
        return torch.mean(torch.nn.functional.relu(positive_scores - negative_scores + margin))

    def mse_loss(self, mean_radius, std_radius, ground_truth_mean_radius, ground_truth_std_radius):
        loss = 0.5 * torch.nn.functional.mse_loss(mean_radius,
                                                  ground_truth_mean_radius) + 0.5 * torch.nn.functional.mse_loss(
            std_radius, ground_truth_std_radius)
        return loss

    def configure_optimizers(self):
        dual_encoder_optimizer = torch.optim.Adam(
            params=list(self.entity_encoder.parameters()) + list(self.mention_encoder.parameters()), lr=1e-5)
        cluster_optimizer = torch.optim.Adam(params=[self.mean_radius, self.std_radius], lr=1e-2)
        return dual_encoder_optimizer, cluster_optimizer
