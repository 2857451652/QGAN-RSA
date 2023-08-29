import os.path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import deque

import torch
import xlwt
import xlrd
from tqdm import tqdm
import time
import overpy
from .error import selfDefinedError
import math
from .trajtools import calDistance, calMeterToDegree
from .utility import GeoConvert
# reference!!!
# box seq: according to coordinate value increase
#   2 | 3
#  ———+———
#   0 | 1
# something to say about the naming method:
# when tree node has been used as a node, the func will name with 'node'
# while when tree node has been as a tile in map, the func will name with 'tile'


# self defined quad tree for satellite imagery
class SatQuadTree:
    def __init__(self, bbox, max_items=10, max_depth=8):
        self.bbox = {"xmin": bbox[0], "ymin": bbox[1],
                     "xmax": bbox[2], "ymax": bbox[3]}  # x means longitude, y means latitude
        self.max_items = max_items
        self.max_depth = max_depth
        self.root_node = BoxNode(bbox, max_items, max_depth, depth=0, node_id=0)
        self.error_count = 0
        self.__lock_up = False  # when locked up, tree structure cannot be changed anymore
        self.nodes_map = [0]  # map the node ids to a continuous domain
        self.GC = GeoConvert()
        # self.img = mpimg.imread('./data/imagery/newyork_base.png')

    def __addNewPoint(self, id, lon, lat):  # add new point into the tree
        if self.__lock_up:
            raise selfDefinedError("new point shouldn't be added when tree is locked up!")

        if self.bbox["xmin"] <= lon <= self.bbox["xmax"] and\
                self.bbox["ymin"] <= lat <= self.bbox["ymax"]:  # new point should be in the boundary
            self.root_node.addItem(id, lon, lat)
        else:
            self.error_count += 1

    ########################################################################
    # load tree
    ########################################################################
    def loadTreeByCsv(self, path):
        workbook = xlrd.open_workbook(filename=path)
        table = workbook.sheets()[0]
        rows = table.nrows
        field_names = table.row_values(rowx=0, start_colx=0, end_colx=None)  # get the first line of field names
        id_col = field_names.index('node_id')  # only need ID !
        for row in tqdm(range(1, rows)):  # ignore the first line
            node_id = int(table.cell_value(row, id_col))

            tree_pass = []  # the pass from the node to root
            recur_node = node_id
            # generate pass
            while recur_node > 0:
                recur_node -= 1
                tree_pass.append(recur_node % 4)  # which node to route
                recur_node //= 4  # exact divide 4
            # walk the pass to find the node
            node = self.root_node
            while len(tree_pass):
                route = tree_pass.pop()
                if not node.have_subboxs:
                    node.createSubBoxs()
                node = node.subboxs[route]
        self.generateMapping()
        return

    def loadPoiData(self, path, geo_convert=False):  # directory need to add a 'r', means raw string
        workbook = xlrd.open_workbook(filename=path)
        table = workbook.sheets()[0]
        rows = table.nrows
        field_names = table.row_values(rowx=0, start_colx=0, end_colx=None)  # get the first line of field names
        id_col = field_names.index('id')
        lon_col = field_names.index('longitude')
        lat_col = field_names.index('latitude')  # find those columns
        self.error_count = 0
        for row in tqdm(range(1, rows)):  # ignore the first line
            id = table.cell_value(row, id_col)
            lon = table.cell_value(row, lon_col)
            lat = table.cell_value(row, lat_col)
            if geo_convert:
                lon, lat = self.GC.wgs84_to_gcj02(lon, lat)  # convert coord
            self.__addNewPoint(id, lon, lat)
        self.generateMapping()
        print("successfully read POI data, with {} error point detected".format(self.error_count))
        return

    def loadPoiFromOverpass(self):  # get data directly from overpass api
        api = overpy.Overpass()
        # fetch all ways and nodes
        print("processing...")
        result = api.query("""
                [out:json][timeout:3600][bbox:{},{},{},{}];
                (node["amenity"]["name"];);
                // print results
                out body;
                """.format(self.bbox["ymin"], self.bbox["xmin"], self.bbox["ymax"], self.bbox["xmax"]))
        for node in result.nodes:
            self.__addNewPoint(node.id, node.lon, node.lat)
        self.generateMapping()
        print("successfully download POI data from overpass and added into the tree.")
        return

    ########################################################################
    # export tree
    ########################################################################
    def exportGridCsv(self, directory):  # export grid coordinate in csv format
        print("exporting grid...")
        book = xlwt.Workbook(encoding='utf-8')
        sheet = book.add_sheet('divided_grids', cell_overwrite_ok=True)
        # fill the field names
        sheet.write(0, 0, "wkt_geom")
        sheet.write(0, 1, "id")
        sheet.write(0, 2, "node_id")
        sheet.write(0, 3, "xmin")
        sheet.write(0, 4, "ymin")
        sheet.write(0, 5, "xmax")
        sheet.write(0, 6, "ymax")

        line = 1
        queue = deque([self.root_node])  # queue without length limit
        while queue:
            cur_node = queue.popleft()
            # ------visit------
            # fill in the sheet
            sheet.write(line, 0, self.__polygonField(cur_node.bbox))
            sheet.write(line, 1, line)  # id is same to line number
            sheet.write(line, 2, cur_node.id)  # node_id
            sheet.write(line, 3, cur_node.bbox["xmin"])
            sheet.write(line, 4, cur_node.bbox["ymin"])
            sheet.write(line, 5, cur_node.bbox["xmax"])
            sheet.write(line, 6, cur_node.bbox["ymax"])
            line += 1  # add line
            # -----------------
            if cur_node.have_subboxs:
                for sub_nodes in cur_node.subboxs:  # add subboxs into queue
                    queue.append(sub_nodes)
        time_now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        book.save(os.path.join(directory, 'grid_' + time_now + '.csv'))
        print("successfully saved data to " + directory + "/grid_" + time_now + ".csv ")
        return

    def __polygonField(self, bbox):  # the field qgis needed
        return "Polygon (({} {}, {} {}, {} {}, {} {}, {} {}))"\
            .format(bbox["xmin"], bbox["ymax"], bbox["xmax"], bbox["ymax"],
                    bbox["xmax"], bbox["ymin"], bbox["xmin"], bbox["ymin"],
                    bbox["xmin"], bbox["ymax"])

    def exportTreeInHeteroFormat(self):  # export tree structure to a format that can be added to DGL
        self.generateMapping()  # regenerate just incase
        self.lockTree()  # stop tree structure from changing so that the mapping array has fixed match
        src_nodes = []
        des_nodes = []
        # BFS
        queue = deque([self.root_node])
        while queue:
            cur_node = queue.popleft()
            # ------visit------
            if cur_node.subboxs:
                for sub_node in cur_node.subboxs:
                    # current node to sub node
                    src_nodes.append(self.idProject(node_id=cur_node.id))
                    des_nodes.append(self.idProject(node_id=sub_node.id))
                    # sub node to current node
                    src_nodes.append(self.idProject(node_id=sub_node.id))
                    des_nodes.append(self.idProject(node_id=cur_node.id))
            # -----------------
            if cur_node.have_subboxs:
                for sub_nodes in cur_node.subboxs:  # add subboxs into queue
                    queue.append(sub_nodes)
        return {("tile", "tree_branch", "tile"):
                (torch.tensor(src_nodes), torch.tensor(des_nodes))}

    ########################################################################
    # print tree
    ########################################################################
    def drawBoxs_BFS(self, save_image=None, value_vector=None):  # draw the boxs with BFS algorithm
        print("drawing...")
        # set canvas
        fig = plt.figure()
        fig.set_figheight(15)
        fig.set_figwidth(15)
        # draw points set first
        point_x = []
        point_y = []
        for item in self.root_node.items_list:
            point_x.append(item["lon"] - self.bbox["xmin"])
            point_y.append(item["lat"] - self.bbox["ymin"])
        plt.plot(point_x, point_y, '.')
        # draw the boxs
        queue = deque([self.root_node])
        while queue:
            cur_node = queue.popleft()
            if cur_node.have_subboxs:
                for sub_nodes in cur_node.subboxs:  # add subboxs into queue
                    queue.append(sub_nodes)
            else:  # only need to draw the boxs on the bottom layer
                if value_vector is not None:
                    threshold = 0.4
                    alpha = value_vector[self.idProject(node_id=cur_node.id)].item()
                    # alpha = (alpha-threshold)/(1-threshold)
                    self.__drawArea(cur_node, alpha*0.8)
                else:
                    self.__drawBox(cur_node)
        plt.show()
        print("complete!")
        if save_image is not None:
            plt.savefig(save_image, transparent=True)
        return plt

    def __drawBox(self, box_node):  # draw one box with matplotlib according to its coordinate
        # draw the relative position
        plt.plot([box_node.bbox["xmin"] - self.bbox["xmin"],
                  box_node.bbox["xmax"] - self.bbox["xmin"],
                  box_node.bbox["xmax"] - self.bbox["xmin"],
                  box_node.bbox["xmin"] - self.bbox["xmin"],
                  box_node.bbox["xmin"] - self.bbox["xmin"]],  # x
                 [box_node.bbox["ymin"] - self.bbox["ymin"],
                  box_node.bbox["ymin"] - self.bbox["ymin"],
                  box_node.bbox["ymax"] - self.bbox["ymin"],
                  box_node.bbox["ymax"] - self.bbox["ymin"],
                  box_node.bbox["ymin"] - self.bbox["ymin"]],  # y
                 marker='.', ms=0)
        return

    def __drawArea(self, box_node, value):  # draw one box with matplotlib according to its coordinate
        # draw the relative position
        x_min, y_min = self.root_node.calRatio(box_node.bbox["xmin"], box_node.bbox["ymin"])
        x_max, y_max = self.root_node.calRatio(box_node.bbox["xmax"], box_node.bbox["ymax"])
        plt.fill([x_min, x_max, x_max, x_min, x_min],  # x
                 [y_min, y_min, y_max, y_max, y_min],  # y
                 'r', alpha=value)
        return

    ########################################################################
    # useful functions
    ########################################################################
    def generateMapping(self):  # generate tree node id mapping based on BFS
        self.nodes_map = []  # reset
        # BFS
        queue = deque([self.root_node])
        while queue:
            cur_node = queue.popleft()
            # ------visit------
            self.nodes_map.append(cur_node.id)
            # -----------------
            if cur_node.have_subboxs:
                for sub_nodes in cur_node.subboxs:  # add subboxs into queue
                    queue.append(sub_nodes)
        print("{} tiles in total.".format(len(self.nodes_map)))
        return

    def retAllLeaves(self):
        leaves = []
        # BFS
        queue = deque([self.root_node])
        while queue:
            cur_node = queue.popleft()
            if cur_node.have_subboxs:
                for sub_nodes in cur_node.subboxs:  # add subboxs into queue
                    queue.append(sub_nodes)
            else:
                leaves.append(cur_node.id)
        return leaves

    def findNodeWithNodeId(self, node_id):  # find node with its node id (id contains its position)
        tree_pass = []  # the pass from the node to root
        recur_node = node_id
        # generate pass
        while recur_node > 0:
            recur_node -= 1
            tree_pass.append(recur_node % 4)  # which node to route
            recur_node //= 4  # exact divide 4
        # walk the pass to find the node
        node = self.root_node
        while len(tree_pass):
            route = tree_pass.pop()
            node = node.subboxs[route]
        return node

    def findTilesByLoc(self, lon, lat):  # find tiles from root node down to the leaf
        if not self.root_node.inBox(lon, lat):
            raise selfDefinedError("location out of range.")
        node = self.root_node
        tiles = [node.id]
        while node.subboxs:
            node = node.subboxs[node.findSubBoxByLoc(lon, lat)]  # find next sub node
            tiles.append(node.id)
        return tiles

    def findLeafByLoc(self, lon, lat):  # find leaf id according to a specific location
        tiles = self.findTilesByLoc(lon, lat)
        return tiles[-1]

    def getSubTreeNodesByLocs(self, points_list):  # return a subtree node list according to a list of points
        subtree_nodes = []
        tiles_list = []
        min_len = self.max_depth
        for point in points_list:
            new_tiles = self.findTilesByLoc(point[0], point[1])
            tiles_list.append(new_tiles)
            subtree_nodes += [tile for tile in new_tiles if tile not in subtree_nodes]
            if min_len > len(tiles_list):
                min_len = tiles_list
        # exclude duplicated root, save only the deepest root
        for i in range(1, min_len):  # should start from 1
            mark = True
            for tiles in tiles_list:
                mark &= (tiles[i] == tiles_list[0][i])
            if mark:  # if this layer of nodes are the same
                subtree_nodes.remove(tiles_list[0][i-1])
            else:
                break
        return subtree_nodes

    def getLeavesByLocs(self, points_list):  # get leaves id by location list
        return [self.findLeafByLoc(point[0], point[1]) for point in points_list]

    def getRelPosInTiles(self, points_list, node_id_list):  # get relative location of points in tiles
        if len(points_list) != len(node_id_list):
            raise selfDefinedError("input should be at the same length!")
        rel_pos = []
        for point, node_id in zip(points_list, node_id_list):
            node = self.findNodeWithNodeId(node_id)
            rel_pos.append(list(node.calRatio(point[0], point[1])))
        return rel_pos

    def getRelPos(self, points_list, node_id_list):  # get relative location of points in tiles
        if len(points_list) != len(node_id_list):
            raise selfDefinedError("input should be at the same length!")
        rel_pos = []
        for point, node_id in zip(points_list, node_id_list):
            rel_pos.append(list(self.root_node.calRatio(point[0], point[1])))
        return rel_pos

    def getSubTreeNodesByLeaves(self, leaves):  # return a subtree according to the leaves id
        leaves_route = []
        subtree_nodes = []
        min_len = self.max_depth
        for leaf in list(set(leaves)):
            tree_pass = []  # the pass from the node to root
            recur_node = leaf
            # generate pass
            while recur_node > 0:
                tree_pass.append(recur_node)  # which node to route
                recur_node -= 1
                recur_node //= 4  # exact divide 4
            tree_pass.append(0)
            tree_pass.sort()
            leaves_route.append(tree_pass)
            subtree_nodes += [tile for tile in tree_pass if tile not in subtree_nodes]
            if min_len > len(tree_pass):
                min_len = len(tree_pass)
        # merge tree
        # exclude duplicated root, save only the deepest root
        for i in range(1, min_len):  # should start from 1
            mark = (len(leaves_route) > 1)
            for tiles_index in range(1, len(leaves_route)):
                mark &= (leaves_route[tiles_index][i] == leaves_route[0][i])
            if mark:  # if this layer of nodes are the same
                subtree_nodes.remove(leaves_route[0][i - 1])
            else:
                break
        subtree_nodes.sort()
        return subtree_nodes

    def getLeavesByScope(self, point, distance):  # find node id by a round scope
        return self.root_node.findSubBoxByScope(point, distance)

    def idProject(self, export_id=None, node_id=None):  # convert export id with node id
        if export_id is not None and node_id is None:
            return self.nodes_map[export_id]
        elif node_id is not None and export_id is None:
            return self.nodes_map.index(node_id)
        elif export_id is not None and node_id is not None:
            raise selfDefinedError("you should input only one id, the func will return the other.")
        else:
            raise selfDefinedError("you should at least input one id.")

    def lockTree(self):  # lock up tree structure
        self.__lock_up = True

    def unlockTree(self):  # unlock tree structure
        self.__lock_up = False


# box node definition
class BoxNode:
    def __init__(self, bbox, max_items, max_depth, depth, node_id):  # must be a square
        self.bbox = {"xmin": bbox[0], "ymin": bbox[1],
                     "xmax": bbox[2], "ymax": bbox[3]}
        self.center = ((self.bbox["xmax"] + self.bbox["xmin"])/2,
                       (self.bbox["ymax"] + self.bbox["ymin"])/2)  # grid center
        self.scale = (self.bbox["xmax"] - self.bbox["xmin"],
                       self.bbox["ymax"] - self.bbox["ymin"])  # grid scale (how big is this grid)
        self.maxItems = max_items
        self.max_depth = max_depth
        self.depth = depth
        self.have_subboxs = False
        self.subboxs = None
        self.items_list = []
        self.id = node_id  # quaternary calculated id, it indicates the position of this node

    def calRatio(self, lon, lat):  # calculate the position ratio of the location in this tile
        if not self.inBox(lon, lat):
            raise selfDefinedError("location out of tile!!")
        ratio = ((lon - self.bbox["xmin"]) / self.scale[0],
                 (lat - self.bbox["ymin"]) / self.scale[1])  # the item's relative location in this grid
        return ratio

    def __generateItem(self, id, lon, lat):  # calculate ratio of nodes in this grid
        ratio = self.calRatio(lon, lat)
        return {"id": id, "lon": lon, "lat": lat, "rel_loc": ratio}  # the item object

    def __findSubBoxs(self, item):  # find which subbox this item belongs to
        # box seq: according to coordinate value increase
        #   2   3
        #   0   1
        loc = item["rel_loc"]
        return (loc[0] > 0.5) + (loc[1] > 0.5) * 2

    def findSubBoxByLoc(self, lon, lat):  # find which subbox this location belongs to
        if not self.inBox(lon, lat):
            raise selfDefinedError("loc({}, {}) is not in box({})".format(lon, lat, self.id))
        return (lon > self.center[0]) + (lat > self.center[1]) * 2

    def findSubBoxByScope(self, point, distance):  # find subboxs that this scope covers
        if not self.have_subboxs:  # if it's the leaf
            corners = [(self.bbox['xmin'], self.bbox['ymin']),
                       (self.bbox['xmin'], self.bbox['ymax']),
                       (self.bbox['xmax'], self.bbox['ymin']),
                       (self.bbox['xmax'], self.bbox['ymax'])]
            for corner in corners:
                dist = calDistance(corner, point)
                if dist <= distance:
                    return [self.id]

            offsets = calMeterToDegree(distance)
            main_points = [point,
                           (point[0]+offsets[0], point[1]),
                           (point[0]-offsets[0], point[1]),
                           (point[0], point[1]+offsets[1]),
                           (point[0], point[1]-offsets[1])]
            for point in main_points:
                if self.inBox(point[0], point[1]):
                    return [self.id]
            return []
        else:
            leaves = []
            for i in range(4):
                leaves += self.subboxs[i].findSubBoxByScope(point, distance)
            return leaves

    def inBox(self, lon, lat):
        return (self.bbox["xmin"] <= lon <= self.bbox["xmax"]) and \
               (self.bbox["ymin"] <= lat <= self.bbox["ymax"])

    def createSubBoxs(self):  # create four subbox for this box node
        # create sub boxs
        B0 = BoxNode((self.bbox["xmin"], self.bbox["ymin"], self.center[0], self.center[1]),
                     self.maxItems, self.max_depth, self.depth+1, self.id*4+1)
        B1 = BoxNode((self.center[0], self.bbox["ymin"], self.bbox["xmax"], self.center[1]),
                     self.maxItems, self.max_depth, self.depth+1, self.id*4+2)
        B2 = BoxNode((self.bbox["xmin"], self.center[1], self.center[0], self.bbox["ymax"]),
                     self.maxItems, self.max_depth, self.depth+1, self.id*4+3)
        B3 = BoxNode((self.center[0], self.center[1], self.bbox["xmax"], self.bbox["ymax"]),
                     self.maxItems, self.max_depth, self.depth+1, self.id*4+4)
        self.subboxs = (B0, B1, B2, B3)
        self.have_subboxs = True
        # allocate items of this box into sub boxs
        for item in self.items_list:
            box_id = self.__findSubBoxs(item)
            self.subboxs[box_id].addItem(item["id"], item["lon"], item["lat"])
        return

    def countItem(self):  # count how many item is in this box
        return len(self.items_list)

    def addItem(self, id, lon, lat):  # add new item into this box
        new_item = self.__generateItem(id, lon, lat)
        self.items_list.append(new_item)
        if self.have_subboxs:  # if this box have subboxs: add item into subboxs
            box_id = self.__findSubBoxs(new_item)
            self.subboxs[box_id].addItem(id, lon, lat)
        elif self.countItem() > self.maxItems and self.depth < self.max_depth:
            # else if item number is more than maxItems, create subboxs
            self.createSubBoxs()
        return


if __name__ == '__main__':
    # example()
    print("hi")
