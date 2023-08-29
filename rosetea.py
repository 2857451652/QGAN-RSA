import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import models
from torch.nn.functional import log_softmax, pad
import math
import copy
from utils.error import selfDefinedError
from arcface_loss import ArcFaceLoss


class RoseTea(nn.Module):
    def __init__(self, embedding_encoder, QRP_graph_model,
                 trans_decoder, pos_time_enc, time_enc, poi_enc,
                 imagery_embeddings, leaves, tile_poi_tensor, tile_k, d_model, device):
        super(RoseTea, self).__init__()
        self.EE = embedding_encoder  # first process for embeddings
        self.QRP = QRP_graph_model  # the graph model for the sub graph
        # self.mixture_layer = mixture_layer
        self.trans_decoder = trans_decoder  # same model logic with the transformer encoder
        self.poi_predictor = copy.deepcopy(trans_decoder)
        self.pos_time_enc = pos_time_enc  # seq position and loc position encoder
        self.time_enc = time_enc
        self.poi_embed_layer = poi_enc  # poi embed layer
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)# cos similarity on different dimension
        self.arcface_loss_tile = ArcFaceLoss(margin=0.05)  # ArcFaceLoss with different margin
        self.arcface_loss_poi = ArcFaceLoss(margin=0.1)

        # data part
        self.imagery_embeddings = imagery_embeddings
        self.leaves_id = torch.tensor(leaves).to(device)  # the leaves of the quad tree in hetero graph
        # self.poi_in_tiles = poi_in_tiles  # a dict indicates pois are in which tile
        self.tile_poi_tensor = tile_poi_tensor
        self.tile_k = tile_k  # top 'k' tiles to pick
        self.device = device

        # tests
        self.EE_test = nn.Embedding(imagery_embeddings.shape[0], d_model).to(device)
        # self.TNE_test = self.EE_test(torch.tensor(range(imagery_embeddings.shape[0])).to(device))

    def pickLastTarget(self, traj_recovery, b_target_tile, b_target_poi):
        b_target_tile = self.recoverEmbeddings(traj_recovery, b_target_tile)
        b_target_tile = self.pickLastEmbedding(b_target_tile[0], traj_recovery).flatten()
        b_target_poi = self.recoverEmbeddings(traj_recovery, b_target_poi)
        b_target_poi = self.pickLastEmbedding(b_target_poi[0], traj_recovery).view(-1, 2)
        return b_target_tile, b_target_poi

    def forward(self, traj_recovery, b_leaves_seq, b_poi_traj, b_pos_seq, b_time_stamp,
                history_traj_recovery, b_history_traj,
                subgraph_recovery, b_subgraph_nodes, b_subgraph,
                b_target_tile, b_target_poi, timer=None):

        # with timer[0]:
        # Tree node embeddings
        TNE = self.EE(self.imagery_embeddings)  # compress the embeddings
        leaves_embedding = TNE.index_select(0, self.leaves_id)  # leaves embeddings
        # leaves_embedding = self.EE_test(self.leaves_id)  # here

        LE = leaves_embedding  # use acronym to avoid long code lines

        # with timer[1]:
        # subgraph GAT
        subgraph_tile_embedding, \
        subgraph_poi_embedding = self.QRP(b_subgraph,
                                          TNE.index_select(0, b_subgraph_nodes),
                                          # self.EE_test(b_subgraph_nodes),  # here
                                          self.poi_embed_layer(b_history_traj))  # subgraph embeddings
        # with timer[2]:
        STE, STE_mask = self.recoverEmbeddings(subgraph_recovery, subgraph_tile_embedding)
        STE_mask = STE_mask.unsqueeze(1)
        SPE, SPE_mask = self.recoverEmbeddings(history_traj_recovery, subgraph_poi_embedding)
        SPE_mask = SPE_mask.unsqueeze(1)

        # STAGE 1: top k tiles prediction
        # generate "tile traj" embeddings and "poi traj" embeddings
        leaves_seq_embedding = TNE.index_select(0, b_leaves_seq)
        # leaves_seq_embedding = self.EE_test(b_leaves_seq)  # here

        LSE, LSE_mask = self.recoverEmbeddings(traj_recovery, leaves_seq_embedding)
        poi_embedding = self.poi_embed_layer(b_poi_traj)
        PE, PE_mask = self.recoverEmbeddings(traj_recovery, poi_embedding)
        # with timer[3]:
        # add pos and time encoding to tile traj embedding
        TE = self.pos_time_enc(LSE, b_pos_seq, b_time_stamp)  # target embeddings

        PE = self.time_enc(PE, b_time_stamp)  # added time embeddings
        TE_mask = LSE_mask.unsqueeze(1) | self.subsequentMask(LSE_mask.shape[1])  # transformer mask for tile traj
        PE_mask = PE_mask.unsqueeze(1) | self.subsequentMask(PE_mask.shape[1])  # transformer mask for poi traj
        # SPE_mask = SPE_mask | self.subsequentMask(SPE_mask.shape[1])

        # go through tile transformer decoder and only pick the last embedding
        tile_output = self.trans_decoder(TE, TE_mask, STE, STE_mask)
        tile_output = self.pickUsefulEmbedding(tile_output, traj_recovery)

        tile_probability = self.cos_sim(LE, tile_output.unsqueeze(1))  # cos sim indicates the probability

        # with timer[4]:
        # STAGE 2: top k POI prediction
        candidate_pois, poi_recovery, target_poi_index, num = \
            self.pickCandidatePOI(tile_probability, b_target_tile, b_target_poi)
        # with timer[5]:
        candidate_poi_embedding = self.poi_embed_layer(candidate_pois)  # generate poi candidate embeddings
        CPE, CPE_mask = self.recoverEmbeddings(poi_recovery, candidate_poi_embedding)  # Candidates Poi Embeddings
        poi_output = self.trans_decoder(PE, PE_mask, SPE, SPE_mask)
        poi_output = self.pickUsefulEmbedding(poi_output, traj_recovery)

        poi_probability = self.cos_sim(CPE, poi_output.unsqueeze(1))
        poi_probability = poi_probability.masked_fill(CPE_mask, -1)

        # part for loss
        tile_loss = self.arcface_loss_tile(tile_output, LE, b_target_tile)
        poi_loss = self.arcface_loss_poi(poi_output, CPE, target_poi_index, CPE_mask)

        if tile_output.isnan().any() or tile_loss.isnan().any():  # check for unexpected NaN
            raise selfDefinedError("WTF! {}, {}, ".format(tile_loss, tile_output))

        return tile_probability, tile_loss, poi_probability, poi_loss, target_poi_index

    def pickCandidatePOI(self, tile_output, b_target_tile, b_target_poi):  # pick the pois id in predicted tiles
        # this is unreadable, I need to make it efficient
        prediction = torch.topk(tile_output, k=self.tile_k, dim=1)
        filtered_tiles = prediction.indices

        hitted = (filtered_tiles == b_target_tile.view(-1, 1)).sum(dim=1)
        last_col = filtered_tiles[:, -1] * hitted
        last_col = last_col + b_target_tile * (hitted==False)
        filtered_tiles[:, -1] = last_col
        # for i in range(filtered_tiles.shape[0]):
        #     if b_target_tile[i] not in filtered_tiles[i]:
        #         filtered_tiles[i][-1] = b_target_tile[i]

        batch_size = filtered_tiles.shape[0]
        filtered_tiles = filtered_tiles.flatten()
        candidate = self.tile_poi_tensor.index_select(0, filtered_tiles)
        candidate = candidate.view(batch_size, -1, 2)  # candidates in

        mask = (candidate != -1)
        candidate_pois = candidate[mask].view(-1, 2)
        poi_recovery = mask.all(dim=-1).sum(dim=1)

        target_mask = (candidate[:, :, 0] == b_target_poi[:, 0].view(-1, 1))  # where is target in the candidate sets
        position = torch.where(target_mask)[1]
        position_mask = torch.arange(candidate.shape[1]).to(self.device)[None, :] < position[:, None]
        target_poi_index = (mask.all(dim=-1) & position_mask).sum(dim=1)

        return candidate_pois, poi_recovery, target_poi_index, mask.sum()

    def recoverEmbeddings(self, recovery, embeddings):  # recover the embeddings to 2D version
        splitted = torch.split(embeddings, list(recovery))
        max_length = recovery.max().item()  # fetch the max length
        mask = torch.arange(max_length).to(self.device)[None, :] >= recovery[:, None]
        return pad_sequence(splitted, batch_first=True, padding_value=0).to(self.device), mask.to(self.device)

    def pickLastEmbedding(self, embeddings, lengths, offset=1):  # pick the last embedding of unmasked part
        if offset == 1:
            off_set = torch.ones(len(lengths))
        else:
            off_set = torch.zeros(len(lengths))
        off_set = embeddings.shape[1] * torch.tensor(range(len(lengths))) - off_set
        em_picker = off_set.to(self.device) + lengths
        embeddings = embeddings.view(embeddings.shape[0] * embeddings.shape[1], -1)
        output = embeddings.index_select(0, em_picker.int())
        return output

    def pickUsefulEmbedding(self, embeddings, lengths):  # pick the useful embedding of unmasked part
        head = (embeddings.shape[1] * torch.tensor(range(len(lengths)))).to(self.device)
        tail = head + lengths
        embeddings = embeddings.view(embeddings.shape[0] * embeddings.shape[1], -1)
        output = [embeddings[h:t, :] for h, t in zip(head, tail)]
        return torch.cat(output, 0)

    def subsequentMask(self, size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        return (subsequent_mask == 1).to(self.device)


def make_model(config, imagery_embeddings, tile_poi_tensor, leaves, poi_loc):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    # sub layer 1
    # embedding_encoder = models.EmbeddingEncoder(config.hyper_length, config.d_model, config.dropout)
    embedding_encoder = models.ImageEncoderSimple(config.hyper_length, config.d_model)
    # sub layer 2
    QRP_graph_model = models.subHRGAT(config.d_model, config.d_hidden, config.d_model,
                                      config.gat_layer_num, heads_num=config.multi_head)
    # sub layer 3
    attn = models.MultiHeadedAttention(config.multi_head, config.d_model)
    ff = models.PositionwiseFeedForward(config.d_model, config.d_hidden, config.dropout)
    trans_decoder = models.TransDecoder(
        models.TransDecoderLayer(config.d_model, c(attn), c(attn), c(ff), config.dropout),
        config.trans_dec_layer)
    # mixture_layer = models.TransMixture(
    #     models.MixtureLayer(config.d_model, c(attn), c(attn), c(ff), c(ff), config.dropout),
    #     config.trans_enc_layer)

    # sub layer 4
    pos_time_enc = models.PosTimeEnc(config.d_model, config.dropout, config.device)
    time_enc = models.TimeEnc(config.d_model, 1, config.device)

    # sub layer 5
    poi_enc = models.POIEmbeddingLayer(config.d_model, poi_loc, config.device)

    model = RoseTea(embedding_encoder, QRP_graph_model,
                    trans_decoder, pos_time_enc, time_enc, poi_enc,
                    imagery_embeddings, leaves, tile_poi_tensor, config.tile_k, config.d_model, config.device)

    model.to(config.device)
    print('Let\'s use the {}'.format(config.device))

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.xavier_uniform_(p.unsqueeze(0))

    return model
