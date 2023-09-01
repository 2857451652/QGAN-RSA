import os

from utils.data_pp import genHeteroGraph, TrajDataPP, MakeDataset
from utils.quadtree import SatQuadTree
from utils.config import ae_config
import utils.config as conf
import train
import test
from utils.config import pp_config, traj_config
from utils.utility import LocToTile
from dgl import load_graphs


def quadtree_program():
    print("processing...")
    # sqt = SatQuadTree((116.0200, 39.6500, 116.7400, 40.2100))  # beijing
    sqt = SatQuadTree(pp_config.boundary, max_depth=pp_config.qtree["max_depth"],
                      max_items=pp_config.qtree["max_items"])  # chengdu
    sqt.loadPoiData(pp_config.poi_path, geo_convert=False)
    # sqt.loadTreeByCsv(r"./data/grid/grid_2023-01-06_15-55-50.csv")
    sqt.drawBoxs_BFS()
    sqt.exportGridCsv(r"./data/grid")
    # print(sqt.getLeavesByScope([116.354183, 39.872929, 116.371510, 39.889844]))


def foursquare_data_pp_program():
    # # utility to make traj dataset for New York
    sqt = SatQuadTree(pp_config.boundary, max_depth=pp_config.qtree["max_depth"],
                      max_items=pp_config.qtree["max_items"])
    sqt.loadTreeByCsv(pp_config.qtree_csv_path)
    [g], _ = load_graphs(pp_config.graph_path)
    train_path = os.path.join(pp_config.traj_pp_dir, "train.txt")
    val_path = os.path.join(pp_config.traj_pp_dir, "val.txt")
    train_history_path = os.path.join(pp_config.traj_pp_dir, "train_history.txt")
    val_history_path = os.path.join(pp_config.traj_pp_dir, "val_history.txt")
    # generate training data
    tdp_train = TrajDataPP(sqt, g, train_path, train_history_path, traj_config.batch_size, "foursquare")
    tdp_train.preprocessTraj(pp_config.traj_processed_train, pp_config.traj_subgraph_train)
    # generate val data
    tdp_val = TrajDataPP(sqt, g, val_path, val_history_path, traj_config.batch_size, "foursquare")
    tdp_val.preprocessTraj(pp_config.traj_processed_val, pp_config.traj_subgraph_val)


if __name__ == '__main__':
    # train.autoencoder_finetuning(ae_config)
    # test.autoencoder_test(ae_config)  # generate comparison image

    # # data_pp.saveTrajPP(pp_config)
    # data_pp.makeChengduTrajPoints(pp_config)
    # data_pp.makeChengduData(traj_config)

    # datasets_program()
    #
    ######### data pp #########
    # # step 1
    # MakeDataset()
    # # step 2
    # quadtree_program()
    # # step 3
    # train.saveImageAsSimpleEmbedding(ae_config)
    # # step 4
    # genHeteroGraph("tree")
    # # step 5
    # foursquare_data_pp_program()
    ######### train #########
    if traj_config.action == "train":
        ret = train.train_rosetea(traj_config)
        if ret:  # if early stopped, need to train a second stage
            conf.initSettings()
            traj_config.lr = traj_config.lr/2
            traj_config.epoch = 10
            traj_config.load_pretrained = True
            ret = train.train_rosetea(traj_config)
    elif traj_config.action == "test":
        train.test_rosetea(traj_config, traj_config.model_saving_path)
    # train.getEmbeddedTiles(traj_config, traj_config.model_saving_path)
