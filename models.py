import math
import copy
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import layers
import torch.nn.functional as F


########################################################################
# GAT models
########################################################################

class subHRGAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, layers_num, heads_num=8):
        '''
        :param in_feats: size of input feature
        :param hid_feats: size of hidden layer feature
        :param out_feats: size of output feature
        :param layers_num: how many hidden layers
        :param heads_num: multi-head number (default 8)
        '''

        super(subHRGAT, self).__init__()
        self.input_layer = dglnn.HeteroGraphConv({
            "tree_branch": dglnn.GATv2Conv(in_feats, hid_feats//heads_num, heads_num),
            "road": dglnn.GATv2Conv(in_feats, hid_feats//heads_num, heads_num),
            "contains": dglnn.GATv2Conv(in_feats, hid_feats // heads_num, heads_num)
        }, aggregate='sum')  # first layer
        self.hidden_layers = [dglnn.HeteroGraphConv({
            "tree_branch": dglnn.GATv2Conv(hid_feats, hid_feats//heads_num, heads_num),
            "road": dglnn.GATv2Conv(hid_feats, hid_feats//heads_num, heads_num),
            "contains": dglnn.GATv2Conv(hid_feats, hid_feats//heads_num, heads_num)
        }, aggregate='mean') for _ in range(layers_num - 2)]  # hidden layer
        self.output_layer = dglnn.HeteroGraphConv({
            "tree_branch": dglnn.GATv2Conv(hid_feats, out_feats//heads_num, heads_num),
            "road": dglnn.GATv2Conv(hid_feats, out_feats//heads_num, heads_num),
            "contains": dglnn.GATv2Conv(hid_feats, out_feats//heads_num, heads_num)
        }, aggregate='sum')
        # register
        self.add_module('input_layer', self.input_layer)
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.add_module('output_layer', self.output_layer)

    def forward(self, graph, tile_inputs, poi_inputs):
        h = self.input_layer(graph, {'tile': tile_inputs, 'poi': poi_inputs})
        h['tile'] = h['tile'].flatten(1)
        h['poi'] = h['poi'].flatten(1)
        h = {k: F.relu(v) for k, v in h.items()}
        for layer in self.hidden_layers:
            h = layer(graph, h)
            h['tile'] = h['tile'].flatten(1)
            h['poi'] = h['poi'].flatten(1)
            h = {k: F.relu(v) for k, v in h.items()}
        outputs = self.output_layer(graph, h)
        outputs['tile'] = outputs['tile'].flatten(1)
        outputs['poi'] = outputs['poi'].flatten(1)
        outputs = {k: F.relu(v) for k, v in outputs.items()}
        return outputs['tile'], outputs['poi']


class EmbeddingEncoder(nn.Module):
    def __init__(self, feature_size, d_model, dropout):
        super(EmbeddingEncoder, self).__init__()
        self.transit = nn.ModuleList([nn.Sequential(nn.Linear(feature_size, d_model*8), nn.Dropout(dropout)),
                                      nn.Sequential(nn.Linear(d_model*8, d_model*4), nn.Dropout(dropout)),
                                      nn.Linear(d_model*4, d_model)])

    def forward(self, embedding):
        output = embedding
        for layer in self.transit:
            output = layer(output)
        return output


class ImageEncoderSimple(nn.Module):
    def __init__(self, feature_size, d_model):
        super(ImageEncoderSimple, self).__init__()
        self.conv_1 = nn.Conv2d(3, 8, 2, 2, 0)

        self.conv_2 = nn.Conv2d(8, 8, 3, 2, 1)
        self.relu_2 = nn.ReLU(inplace=True)

        self.conv_3 = nn.Conv2d(8, 1, 3, 1, 1)
        self.relu_3 = nn.ReLU(inplace=True)

        self.linear = nn.Linear(feature_size//16, d_model)
        self.feature_size = feature_size//16
        self.batch_size = 64

    def forward_func(self, input):
        output = self.conv_1(input)

        output = self.conv_2(output)
        output = self.relu_2(output)

        output = self.conv_3(output)
        output = self.relu_3(output)

        output = self.linear(output.view(-1, self.feature_size))
        return output

    def forward(self, input):
        output = []
        for i in range(0, input.shape[0], self.batch_size):
            output.append(self.forward_func(input[i:i+self.batch_size, :, :, :]))
        output = torch.cat(output, dim=0)
        return self.normalize(output)

    def normalize(self, input):
        mean = input.mean(dim=0).unsqueeze(0)
        std = input.std(dim=0).unsqueeze(0)
        return (input-mean)/std


########################################################################
# Transformer decoder models
########################################################################
class TransEncoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(TransEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, src, src_mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)


class TransDecoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(TransDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, tgt, tgt_mask, src, src_mask):
        for layer in self.layers:
            tgt = layer(tgt, tgt_mask, src, src_mask)
        return self.norm(tgt)


class TransMixture(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(TransMixture, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, left, left_mask, right, right_mask):
        for layer in self.layers:
            left, right = layer(left, left_mask, right, right_mask)
        return self.norm(left), self.norm(right)


class TransEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([copy.deepcopy(SublayerConnection(size, dropout)) for _ in range(2)])
        self.size = size

    def forward(self, src, src_mask):
        "Follow Figure 1 (left) for connections."
        src = self.sublayer[0](src, lambda src: self.self_attn(src, src, src, src_mask))
        return self.sublayer[1](src, self.feed_forward)


class TransDecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(TransDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([copy.deepcopy(SublayerConnection(size, dropout)) for _ in range(3)])

    def forward(self, tgt, tgt_mask, src, src_mask):
        "Follow Figure 1 (right) for connections."
        tgt = self.sublayer[0](tgt, lambda tgt: self.self_attn(tgt, tgt, tgt, tgt_mask))
        tgt = self.sublayer[1](tgt, lambda tgt: self.src_attn(tgt, src, src, src_mask))
        return self.sublayer[2](tgt, self.feed_forward)


class MixtureLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, left_attn, right_attn, left_forward, right_forward, dropout):
        super(MixtureLayer, self).__init__()
        self.size = size
        self.left_attn = left_attn
        self.right_attn = right_attn
        self.left_forward = left_forward
        self.right_forward = right_forward
        self.left_layer = nn.ModuleList([copy.deepcopy(SublayerConnection(size, dropout)) for _ in range(2)])
        self.right_layer = nn.ModuleList([copy.deepcopy(SublayerConnection(size, dropout)) for _ in range(2)])

    def forward(self, left, left_mask, right, right_mask):
        "Follow Figure 1 (right) for connections."
        left = self.left_layer[0](left, lambda left: self.left_attn(left, right, right, right_mask))
        left = self.left_layer[1](left, self.left_forward)
        right = self.left_layer[0](right, lambda right: self.right_attn(right, left, left, left_mask))
        right = self.left_layer[1](right, self.right_forward)
        return left, right


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PosTimeEnc(nn.Module):
    def __init__(self, d_model, dropout, device):
        # 48+1 because need to leave one for the useless kind -1
        super(PosTimeEnc, self).__init__()
        rate = 0.5
        self.pos_enc = PosEnc(d_model, rate, device)
        self.time_enc = TimeEnc(d_model, 1-rate, device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, seq, pos, time_stamp):
        seq = self.pos_enc(seq, pos)
        seq = self.time_enc(seq, time_stamp)
        return self.dropout(seq)


class PosEnc(nn.Module):
    def __init__(self, d_model, rate, device):
        # 48+1 because need to leave one for the useless kind -1
        super(PosEnc, self).__init__()
        self.d_model = d_model
        self.rate = rate
        self.device = device

    def PosEnc2D(self, pos):  # return a 2d encoding vector according to pos
        '''
        pos = [ [[x11,x12,...], [x21,x22,...], ...],
                [[y11,y12,...], [y21,y22,...], ...] ]  # (2,batch,length)
        PE(x,y,2i) = sin(x/10000^(4i/D))
        PE(x,y,2i+1) = cos(x/10000^(4i/D))
        PE(x,y,2j+D/2) = sin(y/10000^(4j/D))
        PE(x,y,2j+1+D/2) = cos(y/10000^(4j/D))
        '''
        x = pos[0].unsqueeze(2)  # (batch, length, 1)
        y = pos[1].unsqueeze(2)
        div_term = torch.exp(
            torch.arange(0, self.d_model / 2, 2) * torch.tensor(-(math.log(10000.0) / self.d_model))
        ).to(self.device)  # (d_model/4)
        pe_2d = torch.zeros(pos.shape[1], pos.shape[2], self.d_model)  # (batch, length, d_model)
        pe_2d[:, :, 0: self.d_model//2: 2] = torch.sin(x * div_term)
        pe_2d[:, :, 1: self.d_model//2: 2] = torch.cos(x * div_term)
        pe_2d[:, :, self.d_model//2: self.d_model: 2] = torch.sin(y * div_term)
        pe_2d[:, :, self.d_model//2+1: self.d_model: 2] = torch.cos(y * div_term)
        return pe_2d.to(self.device)

    def forward(self, seq, pos):
        return seq + self.PosEnc2D(pos).requires_grad_(False) * self.rate  # add image position encoding


class TimeEnc(nn.Module):
    def __init__(self, d_model, rate, device, max_len=48+1):
        # 48+1 because need to leave one for the useless kind -1
        super(TimeEnc, self).__init__()
        self.d_model = d_model
        self.rate = rate
        self.max_len = max_len
        self.time_embed = nn.Embedding(max_len, d_model)
        self.device = device

    def TimeEmbedEnc(self, time_stamp):
        flat_time_stamp = time_stamp.flatten()
        flat_time_stamp = flat_time_stamp.masked_fill(flat_time_stamp == -1, self.max_len - 1)  # trans paddings to max
        time_embedding = self.time_embed(flat_time_stamp)
        time_embedding = time_embedding.view(time_stamp.shape[0], time_stamp.shape[1], -1)
        return time_embedding

    def forward(self, seq, time_stamp):
        return seq + self.TimeEmbedEnc(time_stamp) * self.rate


class POIEmbeddingLayer(nn.Module):
    def __init__(self, hidden_dim, poi_loc, device):
        super(POIEmbeddingLayer, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim

        # data
        self.category_embeddings = nn.Embedding(400, self.hidden_dim)
        self.single_embeddings = nn.Embedding(len(poi_loc[0]), self.hidden_dim)

    def forward(self, poi_vec):
        poi_vec = poi_vec.t()
        poi_index = poi_vec[0]
        cate_index = poi_vec[1]
        # get embeddings
        cate = self.category_embeddings(cate_index)
        single = self.single_embeddings(poi_index)
        poi_embedding = cate + single

        return poi_embedding
