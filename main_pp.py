import os
import time

import dgl
from dgl import DGLHeteroGraph
from .quadtree import SatQuadTree, TreeDummy
from .osmcenter import OsmCenter
from .config import pp_config, traj_config
from utils.data_pp import genHeteroGraph, TrajDataPP, MakeDataset
from utils.quadtree import SatQuadTree
from utils.config import pp_config, traj_config
from dgl import load_graphs


def genHeteroGraph(load="poi"):
    sqt = SatQuadTree(pp_config.boundary, max_depth=pp_config.qtree["max_depth"],
                      max_items=pp_config.qtree["max_items"])
    if load == "poi":
        sqt.loadPoiData(pp_config.poi_path, geo_convert=False)
    if load == "tree":
        sqt.loadTreeByCsv(pp_config.qtree_csv_path)
    oc = OsmCenter(sqt)
    oc.loadOsm(pp_config.osm_path)
    print("generating quadtree...")
    graph_data = sqt.exportTreeInHeteroFormat()
    # print("generating osm graph...")
    data_append, weight = oc.exportOsmInHeteroFormat()
    graph_data.update(data_append)
    g: DGLHeteroGraph = dgl.heterograph(graph_data)
    g.edges["road"].data["weight"] = weight
    dgl.save_graphs(pp_config.graph_path, [g])
    print("generated whole graph, and saved at {}".format(pp_config.graph_path))


def quadtree_program():
    print("processing...")
    # sqt = SatQuadTree((116.0200, 39.6500, 116.7400, 40.2100))  # beijing
    sqt = SatQuadTree(pp_config.boundary, max_depth=pp_config.qtree["max_depth"],
                      max_items=pp_config.qtree["max_items"])  # chengdu
    sqt.loadPoiData(pp_config.poi_path, geo_convert=False)
    # sqt.loadTreeByCsv(r"./data/grid/grid_2023-01-06_15-55-50.csv")
    # sqt.drawBoxs_BFS()
    sqt.exportGridCsv(pp_config.qtree_csv_path)
    # print(sqt.getLeavesByScope([116.354183, 39.872929, 116.371510, 39.889844]))


def data_pp_program():
    # # utility to make traj dataset for New York
    dir = pp_config.traj_pp_dir
    sqt = SatQuadTree(pp_config.boundary, max_depth=pp_config.qtree["max_depth"],
                      max_items=pp_config.qtree["max_items"])
    sqt.loadTreeByCsv(pp_config.qtree_csv_path)
    [g], _ = load_graphs(pp_config.graph_path)
    train_path = os.path.join(dir, "train.txt")
    val_path = os.path.join(dir, "valid.txt")
    test_path = os.path.join(dir, "test.txt")
    train_history_path = os.path.join(dir, "train_history.txt")
    val_history_path = os.path.join(dir, "val_history.txt")
    test_history_path = os.path.join(dir, "test_history.txt")
    # generate training data r"./data/trajectory/PPed/NY_train_data2.pt"
    tdp_train = TrajDataPP(sqt, g, train_path, train_history_path, traj_config.batch_size)
    tdp_train.preprocessTraj(os.path.join(dir, "train.pt"), os.path.join(dir, "train_graph.pt"))
    # generate val data
    tdp_val = TrajDataPP(sqt, g, val_path, val_history_path, traj_config.batch_size)
    tdp_val.preprocessTraj(os.path.join(dir, "valid.pt"), os.path.join(dir, "valid_graph.pt"))
    # generate val data
    tdp_test = TrajDataPP(sqt, g, test_path, test_history_path, traj_config.batch_size)
    tdp_test.preprocessTraj(os.path.join(dir, "test.pt"), os.path.join(dir, "test_graph.pt"))


if __name__ == "__main__":
    ######### data pp #########
    # # step 1
    # MakeDataset()
    # # step 2
    # quadtree_program()
    # # step 3
    # train.saveImageAsSimpleEmbedding(traj_config)
    # # step 4
    # genHeteroGraph("tree")
    # # step 5
    data_pp_program()
