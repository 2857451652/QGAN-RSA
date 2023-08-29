import time

import dgl
import xlrd
from dgl import DGLHeteroGraph
from dgl import save_graphs, load_graphs
import torch
from torch.nn.utils.rnn import pad_sequence
from .quadtree import SatQuadTree
from .osmcenter import OsmCenter
from .config import pp_config, traj_config
from .utility import GeoConvert
from .trajtools import calSpeed, calDistance
import datetime
import os
from utils.quadtree import SatQuadTree
from utils.trajtools import calNeighborSpace
from utils.utility import selfDefinedError
import math
from tqdm import tqdm
from dgl.data.utils import save_graphs
import random
import linecache
from collections import OrderedDict


########################################################################
# Imagery data
########################################################################
# data preprocessing
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
    print("generating osm graph...")
    data_append, weight = oc.exportOsmInHeteroFormat()
    graph_data.update(data_append)
    g: DGLHeteroGraph = dgl.heterograph(graph_data)
    g.edges["road"].data["weight"] = weight
    dgl.save_graphs(pp_config.graph_path, [g])
    print("generated whole graph, and saved at {}".format(pp_config.graph_path))


def loadTreeAndGraph():
    sqt = SatQuadTree(pp_config.boundary)
    # sqt.loadPoiData(pp_config.poi_path)
    sqt.loadTreeByCsv(pp_config.qtree_csv_path)
    [graph], _ = dgl.load_graphs(pp_config.graph_path)
    return sqt, graph


########################################################################
# Trajectory data ( for New York and Tokyo )
########################################################################
# Tue Apr 03 18:00:09 +0000 2012
def time_projection(tm):
    tid = tm.tm_hour * 2
    tid += (tm.tm_min > 30)
    return tid


def getUserTraj(path):
    file = open(path, encoding="Latin-1")
    print("opened successfully")
    line = file.readline()
    count = 0
    user_list = []
    traj_list = []
    poi_dict = {}
    poi_cate = []
    # spilt user traj from the large file
    while line:
        splited_line = line.split('\t')
        user_id = splited_line[0]
        poi_id = splited_line[1]
        poi_c_id = splited_line[2]
        lon = float(splited_line[4])
        lat = float(splited_line[5])
        time_offset = int(splited_line[-2])
        tm = time.strptime(splited_line[-1], "%a %b %d %H:%M:%S +0000 %Y\n")
        tm = time.localtime(time.mktime(tm) + time_offset*60)  # cal an offset to the UTC time

        if poi_id not in poi_dict:
            if poi_c_id not in poi_cate:
                poi_cate_id = len(poi_cate)
                poi_cate.append(poi_c_id)
            else:
                poi_cate_id = poi_cate.index(poi_c_id)
            poi_dict[poi_id] = (lon, lat, poi_cate_id)
        else:
            lon, lat, poi_cate_id = poi_dict[poi_id]

        if user_id in user_list:
            traj_list[user_list.index(user_id)].append((poi_id, lon, lat, tm, poi_cate_id))
        else:
            user_list.append(user_id)
            traj_list.append([(poi_id, lon, lat, tm, poi_cate_id)])
        line = file.readline()
        count += 1
    file.close()
    return user_list, traj_list, poi_dict


def trajSplit(user_list, traj_list):
    splited_traj = {}
    traj_count = 0
    for user_id, traj in tqdm(zip(user_list, traj_list)):  # go through the whole traj
        last_tid = 0
        single_traj = []
        user_traj = []
        for record in traj:
            # judge the time gap
            tm = record[-2]
            tid = int(time.mktime(tm))  # generate time id (s)
            if last_tid == 0:
                last_tid = tid
                single_traj.append(record)
            elif (tid - last_tid) / 3600 > 48 or len(single_traj) > 20:  # hours gap
                if len(single_traj) > 5:
                    user_traj.append(single_traj)
                single_traj = [record]  # if gap is too long restart a new traj
                last_tid = tid
            elif (tid - last_tid) / 60 > 10:  # minute gap
                single_traj.append(record)
                last_tid = tid
            else:
                pass

        if len(user_traj) >= 6:  # record the user trajs
            splited_traj[user_id] = user_traj
            traj_count += len(user_traj)
    print("{} users, {} trajs".format(len(splited_traj.keys()), traj_count))
    return splited_traj


def splitDataset(splited_traj, split_rate=0.8):  # split the traj dict into training set and val sets
    train_dataset = []
    val_dataset = []
    train_history = []
    val_history = []
    for user_id in tqdm(splited_traj):
        user_trajs = splited_traj[user_id]
        history_traj = []
        h = []
        for traj in user_trajs[:-1]:
            h += traj
            if len(h) > 100:  # avoid history traj too long
                history_traj.append(h[-100:])
            else:
                history_traj.append(h)
        # len(history_traj) is shorter than len(user_trajs) by 1
        split_num = int((len(user_trajs)-1) * split_rate)
        train_dataset += user_trajs[1:split_num+1]  # get rid of the first traj, consider it as history
        val_dataset += user_trajs[split_num+1:]
        train_history += history_traj[:split_num]
        val_history += history_traj[split_num:]
    print("{} trajs in training dataset, "
          "{} trajs in val dataset".format(len(train_dataset), len(val_dataset)))
    return train_dataset, val_dataset, train_history, val_history


def saveDataset(dataset, poi_dict, save_path):
    poi_list = list(poi_dict.keys())
    target_file = open(save_path, "w")
    for traj in tqdm(dataset):
        line = []
        for record in traj:
            item = "{}|{}|{:.6f}|{:.6f}|{}".format(poi_list.index(record[0]), time_projection(record[3]),
                                        record[1], record[2], record[4])
            line.append(item)
            assert poi_dict[record[0]] == (record[1], record[2], record[4])
        line = ",".join(line) + "\n"
        target_file.write(line)
    target_file.close()
    print("saved dataset {}".format(save_path))
    return


def savePoiPoints(poi_dict, save_path):
    target_file = open(save_path, "w")
    target_file.write("id,longitude,latitude,poi_id,cate\n")
    for i, poi_id in tqdm(enumerate(poi_dict)):
        record = poi_dict[poi_id]
        target_file.write("{},{:.6f},{:.6f},{},{}\n".format(i, record[1], record[0], poi_id, record[2]))
    target_file.close()
    print("saved poi points to {}".format(save_path))
    return


def MakeDataset(path=pp_config.data_file_path,
                save_dir=pp_config.traj_pp_dir):
    user_list, traj_list, poi_dict = getUserTraj(path)
    splited_traj = trajSplit(user_list, traj_list)
    train_dataset, val_dataset, train_history, val_history = splitDataset(splited_traj, 0.8)
    saveDataset(train_dataset, poi_dict, os.path.join(save_dir, "train.txt"))
    saveDataset(val_dataset, poi_dict, os.path.join(save_dir, "val.txt"))
    saveDataset(train_history, poi_dict, os.path.join(save_dir, "train_history.txt"))
    saveDataset(val_history, poi_dict, os.path.join(save_dir, "val_history.txt"))
    savePoiPoints(poi_dict, os.path.join(save_dir, "poi.txt"))


########################################################################
# all item preparation
########################################################################
class TrajDataPP:
    def __init__(self, quad_tree: SatQuadTree, hetero_graph: DGLHeteroGraph,
                 traj_pp_path, history_path, batch=8, choice="foursquare"):
        self.quad_tree = quad_tree
        self.hetero_graph = hetero_graph
        self.choice = choice
        self.traj_pp_path = traj_pp_path
        self.history_path = history_path
        self.batch = batch
        return

    def getTraj(self, index):  # get traj from processed data
        line = linecache.getline(self.traj_pp_path, index+1)
        loc_traj, poi_traj, time_traj = self.line_to_traj(line)
        return loc_traj, poi_traj, time_traj

    def getHistoryTraj(self, index):  # get traj from history data
        line = linecache.getline(self.history_path, index+1)
        loc_traj, poi_traj, _ = self.line_to_traj(line)
        return loc_traj, poi_traj

    def line_to_traj(self, line):
        loc_traj = []
        poi_traj = []
        time_traj = []
        line_buf = line.split(',')
        for content in line_buf:
            content = content.split('|')
            loc_traj.append((float(content[3]), float(content[2])))
            poi_traj.append([int(content[0]), int(content[4])])
            time_traj.append(int(content[1]))
        return loc_traj, poi_traj, time_traj

    def get_graph_dict(self, g: DGLHeteroGraph):
        etypes = g.etypes
        graph_dict = {}
        for type in etypes:
            edge_name = g.to_canonical_etype(type)
            edge_data = g.edges(etype=type)
            graph_dict[edge_name] = edge_data
        return graph_dict

    def getitem(self, index):
        '''
        :param index: trajectory index
        :return:
        leaves_seq: traj corresponding leaves sequence
        rel_pos_seq: traj point relative position in its leaf

        '''
        traj, poi_traj, time_traj = self.getTraj(index)
        traj_leaves_seq = self.quad_tree.getLeavesByLocs(traj)
        leaves_id = [self.quad_tree.idProject(node_id=id) for id in self.quad_tree.retAllLeaves()]
        # item 1: traj corresponding leaves sequence
        graph_leaves_seq = torch.tensor([self.quad_tree.idProject(node_id=node) for node in traj_leaves_seq])
        # item 2: traj point relative position in its leaf
        # rel_pos_seq = torch.tensor(self.quad_tree.getRelPosInTiles(traj, traj_leaves_seq))
        rel_pos_seq = torch.tensor(self.quad_tree.getRelPos(traj, traj_leaves_seq))
        # item 3: traj poi id traj
        poi_traj = torch.tensor(poi_traj)
        # item 4: time seq
        time_stamp = torch.tensor(time_traj)
        # item 4: target
        tmp = [leaves_id.index(int(leaf)) for leaf in graph_leaves_seq[1:]]
        target = {"tile_id": torch.tensor(tmp),
                  "poi_id": poi_traj[1:]}
        # item 5: subgraph node id
        # item 6: subgraph
        history_traj, history_poi_traj = self.getHistoryTraj(index)
        history_leaves_seq = self.quad_tree.getLeavesByLocs(history_traj)
        tree_node_id = self.quad_tree.getSubTreeNodesByLeaves(history_leaves_seq)
        graph_node_id = [self.quad_tree.idProject(node_id=node) for node in tree_node_id]
        subgraph_nodes = torch.tensor(graph_node_id)  # subgraph nodes
        subgraph = dgl.node_subgraph(self.hetero_graph, graph_node_id)  # the subgraph structure

        # item 7: poi nodes put in subgraph
        subgraph_dict = self.get_graph_dict(subgraph)
        poi_id = torch.tensor(range(len(history_poi_traj)))
        node_id = torch.tensor([tree_node_id.index(leaf) for leaf in history_leaves_seq])
        subgraph_dict[("tile", "contains", "poi")] = (node_id, poi_id)
        subgraph_dict[("poi", "contains", "tile")] = (poi_id, node_id)
        subgraph = dgl.heterograph(subgraph_dict)

        history_poi_traj = torch.tensor(history_poi_traj)

        return (graph_leaves_seq[:-1], rel_pos_seq[:-1], poi_traj[:-1], time_stamp[:-1],
                history_poi_traj, target, subgraph_nodes), subgraph

    def preprocessTraj(self, data_saving_path, subgraph_saving_path):
        max_len = self.calLength()
        max_len -= max_len % self.batch
        whole_savings = []
        all_subgraph = []
        for i in tqdm(range(max_len)):
            result, subgraph = self.getitem(i)
            whole_savings.append(result)
            all_subgraph.append(subgraph)
        torch.save(whole_savings, data_saving_path, _use_new_zipfile_serialization=True)
        save_graphs(subgraph_saving_path, all_subgraph)

    def calLength(self):
        file = open(self.traj_pp_path, 'r')
        length = len(file.readlines())
        return length


def loadPoiInTiles(sqt: SatQuadTree, poi_path, device):
    workbook = xlrd.open_workbook(filename=poi_path)
    table = workbook.sheets()[0]
    rows = table.nrows
    field_names = table.row_values(rowx=0, start_colx=0, end_colx=None)  # get the first line of field names
    id_col = field_names.index('id')
    lon_col = field_names.index('longitude')
    lat_col = field_names.index('latitude')  # find those columns
    cate_col = field_names.index('cate')
    poi_in_tiles = {}

    leaves = [sqt.idProject(node_id=leaf) for leaf in sqt.retAllLeaves()]
    poi_loc = [[], []]
    for leaf in leaves:
        poi_in_tiles[leaf] = []
    for row in tqdm(range(1, rows)):  # ignore the first line
        id = int(table.cell_value(row, id_col))
        lon = table.cell_value(row, lon_col)
        lat = table.cell_value(row, lat_col)
        cate_id = int(table.cell_value(row, cate_col))
        leaf_id = sqt.findLeafByLoc(lon, lat)
        # node = sqt.findNodeWithNodeId(leaf_id)
        node = sqt.root_node
        position = node.calRatio(lon, lat)
        poi_loc[0].append(position[0])
        poi_loc[1].append(position[1])
        node_id = sqt.idProject(node_id=leaf_id)
        poi_in_tiles[node_id].append([id, cate_id])

    padding = -1
    poi_tile_helper = {}
    for tile_id in poi_in_tiles:

        if poi_in_tiles[tile_id] == []:
            poi_tile_helper[tile_id] = torch.tensor([[padding, padding]])
        else:
            poi_tile_helper[tile_id] = torch.tensor(poi_in_tiles[tile_id])
        # poi_in_tiles[tile_id] = torch.tensor(poi_in_tiles[tile_id]).to(device)
    tile_poi_list = [poi_tile_helper[key] for key in leaves]
    tile_poi_tensor = pad_sequence(tile_poi_list, batch_first=True, padding_value=padding).to(device)

    return tile_poi_tensor, leaves, poi_loc, rows-1
