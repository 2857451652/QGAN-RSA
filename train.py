import gc
import math

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
from collections import OrderedDict
from utils.error import selfDefinedError
from utils.utility import tqdm_dummy
import rosetea
from test import autoencoder_training_test
from utils.data_pp import loadTreeAndGraph, loadPoiInTiles
from utils.config import ae_config

import datasets
import autoencoder
from utils.timer import TotalTime
import os
import random


def SetSeed(seed):
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        random.seed(seed)

########################################################################
# Autoencoder part trainings
########################################################################
def saveAutoencoderEmbedding(config):
    config.device = torch.device("cuda:{}".format(config.device) if torch.cuda.is_available() else "cpu")
    config.batch_size = 1
    config.shuffle = False
    # config.encoder_model_path = os.path.join(config.model_saving_path, "encoder.pth")  # change the path
    img_loader = datasets.SatImageDataloader(config)  # image data loader
    img_encoder_model = autoencoder.get_encoder_only(config)  # autoencoder model
    maxPool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
    avgPool_2 = nn.AvgPool2d(kernel_size=2, stride=2)

    pbar = tqdm(total=len(img_loader), ncols=150)
    all_embeddings = []
    for i, data in enumerate(img_loader):
        # get the content_image batch
        sat_image = data.get('image')
        sat_image = sat_image.to(config.device)
        with torch.no_grad():
            embeddings = img_encoder_model(sat_image)['r41']
            embeddings = maxPool_2(embeddings)
            embeddings = avgPool_2(embeddings)
            embeddings = embeddings[:, 0:embeddings.shape[1]:2, :, :] + \
                         embeddings[:, 1:embeddings.shape[1]+1:2, :, :]
        all_embeddings.append(embeddings.flatten().unsqueeze(0))
        pbar.update(1)
    pbar.close()
    all_embeddings = torch.cat(all_embeddings)
    torch.save(all_embeddings, config.embeddings_saving_path)
    print("saved all image embeddings to {}".format(config.embeddings_saving_path))


def saveImageAsSimpleEmbedding(config):
    config.device = torch.device("cuda:{}".format(config.device) if torch.cuda.is_available() else "cpu")
    config.batch_size = 1
    config.shuffle = False
    img_loader = datasets.SatImageDataloader(config)  # image data loader

    pbar = tqdm(total=len(img_loader), ncols=150)
    all_embeddings = []
    for i, data in enumerate(img_loader):
        sat_image = data.get('image')
        all_embeddings.append(sat_image)
        pbar.update(1)
    pbar.close()
    all_embeddings = torch.cat(all_embeddings)
    torch.save(all_embeddings, config.embeddings_saving_path)
    print("saved all image embeddings to {}".format(config.embeddings_saving_path))


def loadRemoteSensing(embeddings_saving_path):
    return torch.load(embeddings_saving_path, map_location='cpu')


def generateRandomEmbedding(num, size):
    return nn.Embedding(num, size)


########################################################################
# RoseTea part trainings
########################################################################
def runEpoch(rosetea_model, traj_loader, optimizer, config, action="train"):
    print("====={}=====".format(action))
    pbar = tqdm(total=len(traj_loader), ncols=150)
    # pbar = tqdm_dummy(total=len(traj_loader))
    total_tile_loss = 0
    total_poi_loss = 0
    total_tile_acc_1 = 0
    total_tile_acc_k = 0
    total_poi_acc_1 = 0
    total_poi_acc_k = 0
    total_poi_dcg_k = 0
    total_poi_mrr = 0
    iter_times = 0
    timer = TotalTime(7)
    for i, data in enumerate(traj_loader):
        # load data
        traj_recovery, b_leaves_seq, b_poi_traj, b_pos_seq, b_time_stamp, \
            history_traj_recovery, b_history_traj, \
            subgraph_recovery, b_subgraph_nodes, b_subgraph, \
            b_target_tile, b_poi_target = data

        # data to device
        traj_recovery = traj_recovery.to(config.device)
        b_leaves_seq = b_leaves_seq.to(config.device)
        b_poi_traj = b_poi_traj.to(config.device)
        b_pos_seq = b_pos_seq.to(config.device)
        b_time_stamp = b_time_stamp.to(config.device)

        history_traj_recovery = history_traj_recovery.to(config.device)
        b_history_traj = b_history_traj.to(config.device)

        subgraph_recovery = subgraph_recovery.to(config.device)
        b_subgraph_nodes = b_subgraph_nodes.to(config.device)
        b_subgraph = b_subgraph.to(config.device)

        b_target_tile = b_target_tile.to(config.device)
        b_poi_target = b_poi_target.to(config.device)

        if action == "train":
            # go through model
            rosetea_model.train(True)
            optimizer.zero_grad()

            tile_pos, tile_loss, poi_pos, poi_loss, \
                target_poi_index = rosetea_model(traj_recovery, b_leaves_seq, b_poi_traj, b_pos_seq, b_time_stamp,
                                                 history_traj_recovery, b_history_traj,
                                                 subgraph_recovery, b_subgraph_nodes, b_subgraph,
                                                 b_target_tile, b_poi_target, timer)

            tile_acc_1 = accuracy(tile_pos, b_target_tile)
            tile_acc_k = topk_acc(tile_pos, b_target_tile, config.tile_k)
            poi_acc_k = topk_acc(poi_pos, target_poi_index, config.poi_k)
            poi_acc_k_true = topk_poi_acc_true(tile_pos, b_target_tile, poi_pos, target_poi_index,
                                               config.tile_k, config.poi_k)
            loss = 4*tile_loss + poi_loss
            loss.backward()
            nn.utils.clip_grad_norm_(rosetea_model.parameters(), max_norm=4, norm_type=2)
            if something_is_going_wrong(rosetea_model):
                print("FUCK!! HELP!")
            optimizer.step()

            total_tile_loss += tile_loss.item()
            total_poi_loss += poi_loss.item()
            total_tile_acc_1 += tile_acc_1.item()
            total_tile_acc_k += tile_acc_k
            total_poi_acc_1 += poi_acc_k
            total_poi_acc_k += poi_acc_k_true

            iter_times += 1
            postfix = OrderedDict([
                ('tile: loss', total_tile_loss / iter_times),
                ('ttop', total_tile_acc_1 / iter_times),
                ('ttop{}'.format(config.tile_k), total_tile_acc_k / iter_times),
                ('poi: loss', total_poi_loss / iter_times),
                ('ptop{}'.format(config.poi_k), total_poi_acc_1 / iter_times),
                ('ptop{}true'.format(config.poi_k), total_poi_acc_k / iter_times),
            ])
            # if i % 50 == 0 and i != 0:
            #     print(timer)
        else:
            rosetea_model.train(False)
            # rosetea_model.eval()
            tile_probability, tile_loss, poi_probability, poi_loss, \
                target_poi_index = rosetea_model(traj_recovery, b_leaves_seq, b_poi_traj, b_pos_seq, b_time_stamp,
                                                 history_traj_recovery, b_history_traj,
                                                 subgraph_recovery, b_subgraph_nodes, b_subgraph,
                                                 b_target_tile, b_poi_target, timer)

            poi_acc_k = topk_poi_acc_true(tile_probability, b_target_tile, poi_probability, target_poi_index,
                                           config.tile_k, config.poi_k)
            poi_dcg_k = topk_DCG(tile_probability, b_target_tile, poi_probability, target_poi_index,
                                           config.tile_k, config.poi_k)
            poi_mrr = MRR_index(tile_probability, b_target_tile, poi_probability, target_poi_index,
                                 config.tile_k)

            total_tile_loss += tile_loss.item()
            total_poi_loss += poi_loss.item()
            total_poi_acc_k += poi_acc_k
            total_poi_dcg_k += poi_dcg_k
            total_poi_mrr += poi_mrr

            iter_times += 1
            postfix = OrderedDict([
                ('tile{}: loss'.format(config.tile_k), "{:.4f}".format(total_tile_loss / iter_times)),
                ('poi{}: loss'.format(config.poi_k), "{:.4f}".format(total_tile_loss / iter_times)),
                ('poi_ACC@{}'.format(config.poi_k), "{:.4f}".format(total_poi_acc_k / iter_times)),
                ('poi_DCG@{}'.format(config.poi_k), "{:.4f}".format(total_poi_dcg_k / iter_times)),
                ('poi_MRR'.format(config.poi_k), "{:.4f}".format(total_poi_mrr / iter_times)),
            ])

        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return total_poi_acc_k / iter_times


def something_is_going_wrong(model):
    for name, para in model.named_parameters():
        if para.isnan().any():
            return 1
        try:
            if para.grad.isnan().any():
                return 1
        except:
            pass
    return 0


def train_rosetea(config):
    # SetSeed(config.seed)
    sqt, _ = loadTreeAndGraph()
    config.device = torch.device("cuda:{}".format(config.device) if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, val_loader = datasets.TrajDataLoader(config)  # data loader
    embeddings = loadRemoteSensing(config.embeddings_saving_path)
    print("embeddings loaded successfully.")
    tile_poi_tensor, leaves, poi_loc, poi_num = loadPoiInTiles(sqt, config.poi_path, config.device)
    print("pois loaded successfully.")

    rosetea_model = rosetea.make_model(config, embeddings.to(config.device),
                                       tile_poi_tensor, leaves, poi_loc)  # model
    if config.load_pretrained:
        load_rosetea(config.model_loading_path, rosetea_model, config.device)

    try:
        optimizer = optim.Adam(rosetea_model.parameters(), lr=config.lr)
    except:
        optimizer = optim.Adam(rosetea_model.modules.parameters(), lr=config.lr)

    if config.lr_decay > 0:
        torch.optim.lr_scheduler.StepLR(optimizer, 1, config.lr_decay, last_epoch=-1)  # lr decay

    best_acc = 0
    overfitting_judge = 0
    for epoch in range(1, config.epochs + 1):
        print('epoch: {}'.format(epoch))
        runEpoch(rosetea_model, train_loader, optimizer, config, action="train")
        acc = runEpoch(rosetea_model, test_loader, optimizer, config, action="test")

        if config.save_model:
            if acc > best_acc:
                save_rosetea(config.model_saving_path, rosetea_model)
                best_acc = acc
                overfitting_judge = 0
            elif best_acc - acc >= 0.01:
                rosetea_model = load_rosetea(config.model_saving_path, rosetea_model, config.device)
                overfitting_judge += 2
            else:
                overfitting_judge += 1

        if overfitting_judge > 8:
            print("I guess an over-fitting problem is happening..Auto break.")
            break

    # validate
    print("========================COMPLETE========================")
    runEpoch(rosetea_model, val_loader, optimizer, config, action="val")
    return overfitting_judge > 8


def test_rosetea(config, model_path):
    config.batch_size = 1
    sqt, _ = loadTreeAndGraph()
    config.device = torch.device("cuda:{}".format(config.device) if torch.cuda.is_available() else "cpu")
    _, _, val_loader = datasets.TrajDataLoader(config)  # data loader
    embeddings = loadRemoteSensing(config.embeddings_saving_path)
    print("embeddings loaded successfully.")
    poi_in_tiles, leaves, poi_loc, poi_num = loadPoiInTiles(sqt, config.poi_path, config.device)
    print("pois loaded successfully.")

    rosetea_model = rosetea.make_model(config, embeddings.to(config.device),
                                       poi_in_tiles, leaves, poi_loc)  # model

    load_rosetea(model_path, rosetea_model, config.device)
    tile_poi_top_k = [(10, 5), (12, 10), (15, 20)]
    # (15, 5), (5, 1)
    for top_k in tile_poi_top_k:
        print("validating top {} tiles and top {} pois".format(top_k[0], top_k[1]))
        rosetea_model.tile_k = top_k[0]
        config.tile_k = top_k[0]
        config.poi_k = top_k[1]
        runEpoch(rosetea_model, val_loader, None, config, action="val")


# accuracy about tile selecting
def accuracy(output, b_target):
    prediction = torch.argmax(output, dim=1)
    result = (prediction == b_target)
    return result.sum()/len(result)


def topk_acc(output, b_target, k=5):
    prediction = torch.topk(output, k=k, dim=1)
    result = [(t in topk) for t, topk in zip(b_target, prediction.indices)]
    return sum(result)/len(result)


def topk_poi_acc_true(tile_output, b_tile_target, poi_output, b_poi_target, k1=5, k2=10):
    tile_pred = torch.topk(tile_output, k=k1, dim=1)
    tile_result = [(t in topk) for t, topk in zip(b_tile_target, tile_pred.indices)]  # the correct predicted tiles
    poi_pred = torch.topk(poi_output, k=k2, dim=1)
    poi_result = [(t in topk) for t, topk in zip(b_poi_target, poi_pred.indices)]  # the correct predicted poi
    # the true prediction should be tile and poi correctly predicted at the same time
    result = [(tile_r & poi_r) for tile_r, poi_r in zip(tile_result, poi_result)]
    return sum(result)/len(result)  # calculate the rate


def topk_DCG(tile_output, b_tile_target, poi_output, b_poi_target, k1=5, k2=10):
    tile_pred = torch.topk(tile_output, k=k1, dim=1)
    tile_result = [(t in topk) for t, topk in zip(b_tile_target, tile_pred.indices)]  # the correct predicted tiles
    poi_pred = torch.topk(poi_output, k=k2, dim=1)

    DCG_score = 0
    length = 0
    for t, topk, correct in zip(b_poi_target, poi_pred.indices, tile_result):
        # the true prediction should be tile and poi correctly predicted at the same time
        topk = list(topk)
        length += 1
        if correct and t in topk:
            DCG_score += 1.0 / math.log2(topk.index(t) + 2)

    return DCG_score/length  # calculate the average score


def MRR_index(tile_output, b_tile_target, poi_output, b_poi_target, k1=5):
    tile_pred = torch.topk(tile_output, k=k1, dim=1)
    tile_result = [(t in topk) for t, topk in zip(b_tile_target, tile_pred.indices)]  # the correct predicted tiles
    poi_sorted = torch.sort(poi_output, dim=1, descending=True)

    MRR_score = 0
    length = 0
    for t, p_sort, correct in zip(b_poi_target, poi_sorted.indices, tile_result):
        # the true prediction should be tile and poi correctly predicted at the same time
        p_sort = list(p_sort)
        length += 1
        if correct:
            MRR_score += 1.0 / (p_sort.index(t) + 1)

    return MRR_score/length  # calculate the average score


def save_rosetea(save_path, rosetea_model):
    rosetea_state_dict = rosetea_model.state_dict()
    torch.save(rosetea_state_dict, save_path)
    print("successfully saved model to {}.".format(save_path))


def load_rosetea(load_path, rosetea_model, device):
    checkpoint = torch.load(load_path, map_location='cpu')
    rosetea_model.load_state_dict(gen_model_dict(rosetea_model, checkpoint))
    print("successfully loaded model {}.".format(load_path))
    return rosetea_model.to(device)


def gen_model_dict(model, checkpoint, ):
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    return model_dict
