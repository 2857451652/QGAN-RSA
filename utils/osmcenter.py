import osmnx as ox
from .quadtree import SatQuadTree
from .utility import GeoConvert
from .error import selfDefinedError
from shapely.geometry import Point, LineString, mapping
import numpy as np
import torch
import osmium
import xlwt


class OsmCenter:
    def __init__(self, qtree):
        super(OsmCenter, self).__init__()
        self.qtree: SatQuadTree = qtree
        self.osm_graph = None
        self.nodes = None
        self.edges = None
        self.geo_convert = GeoConvert()
        self.src_nodes = []
        self.des_nodes = []

    def way(self, w):
        print(w)
        return

    def loadOsm(self, osm_path):
        print("loading osm file, may take a moment...")
        try:
            self.osm_graph = ox.graph_from_xml(osm_path)
            print("successfully loaded osm file.")
        except:
            print("osm file not found, trying to generate this file...")
            print("please check if your vpn is working.")
            self.getBoxOsm(osm_path)
            print("successfully saved osm file to {}.".format(osm_path))
            self.osm_graph = ox.graph_from_xml(osm_path)
            print("successfully loaded osm file.")

    def downloadPlace(self, place):
        ox.config(log_console=True)
        G = ox.graph_from_place(place, network_type='drive')
        ox.save_graph_xml(G, "../data/trajectory/{}.osm".format(place))

    def getBoxOsm(self, save_path):
        # this code is efficient, if you cannot download osm file, please check your vpn,
        # remember to use the env on your computer!
        # see if you can enter https://www.openstreetmap.org/
        ox.settings.all_oneway = True
        ox.settings.log_console = True
        # G = ox.graph_from_bbox(40.550852, 40.988332, -74.274767, -73.683825, custom_filter='["highway"]')
        G = ox.graph_from_bbox(self.qtree.bbox["ymin"], self.qtree.bbox["ymax"],
                               self.qtree.bbox["xmin"], self.qtree.bbox["xmax"], custom_filter='["highway"]')
        # G = ox.graph_from_place("New York", network_type='drive')
        print("get graph")
        # 形成路网图
        edges = ox.graph_to_gdfs(G, nodes=False)
        print(edges["highway"])
        ox.plot_graph(G)
        ox.save_graph_xml(G, save_path)
        # G = ox.graph_from_address('350 5th Ave, New York, New York', network_type='drive')
        # ox.plot_graph(G)

    def Point_to_GCJ02(self, point):
        dict = mapping(point)
        lon, lat = self.geo_convert.wgs84_to_gcj02(dict['coordinates'][0], dict['coordinates'][1])
        return (lon, lat)

    def LineString_to_GCJ02(self, line_string):
        dict = mapping(line_string)
        converted_line = []
        for (lon, lat) in dict['coordinates']:
            # lon_, lat_ = self.geo_convert.wgs84_to_gcj02(lon, lat)
            lon_, lat_ = lon, lat  # do not convert
            converted_line.append((lon_, lat_))
        return converted_line

    def __linestringField(self, line_string):  # the field qgis needed
        str = "LineString ("
        for node in line_string:
            str += "{} {},".format(node[0], node[1])
        str = str[:-1] + ")"
        return str

    def getLeafTileId(self, tiles):
        return tiles[-1]

    def exportOsmInHeteroFormat(self, output_path=None):
        if self.osm_graph is None:
            raise selfDefinedError("you should load osm graph first.")
        if output_path:  # if user need to output
            book = xlwt.Workbook(encoding='utf-8')
            sheet = book.add_sheet('connect_road', cell_overwrite_ok=True)
            # fill the field names
            sheet.write(0, 0, "wkt_geom")
            sheet.write(0, 1, "id")
            sheet.write(0, 2, "connection")

        ox_edges = ox.graph_to_gdfs(self.osm_graph, nodes=False, edges=True)
        in_boundary = self.qtree.root_node.inBox
        connection = {}
        src_nodes = []
        des_nodes = []
        weight = []
        for edge in ox_edges["geometry"]:  # go through all edges linestring
            node_list = self.LineString_to_GCJ02(edge)  # convert edge's crs to GCJ02
            for node_a, node_b in zip(node_list[:-1], node_list[1:]):  # for every node pair
                if in_boundary(node_a[0], node_a[1]) and \
                   in_boundary(node_b[0], node_b[1]):  # all node should be in the boundary
                    tile_a = self.qtree.idProject(node_id=self.qtree.findLeafByLoc(node_a[0], node_a[1]))
                    tile_b = self.qtree.idProject(node_id=self.qtree.findLeafByLoc(node_b[0], node_b[1]))
                    # ---------------------------------------------
                    # a switching process to control the appearance of tile id and road points in fixed order
                    # this process is essential for later operation
                    if tile_a < tile_b:  # the order is according to the size of projected tile id
                        trans = node_a
                        node_a = node_b
                        node_b = trans
                        trans = tile_a
                        tile_a = tile_b
                        tile_b = trans
                    # ---------------------------------------------
                    if tile_a != tile_b:  # if this is a valid road
                        new_road = self.__linestringField([node_a, node_b])
                        if (tile_a, tile_b) not in connection:  # if this connnection is not stored
                            connection[(tile_a, tile_b)] = [new_road]
                        elif new_road not in connection[(tile_a, tile_b)]:  # if this road is not in this connection
                            connection[(tile_a, tile_b)].append(new_road)

        line = 1
        for edge in connection:  # go through connection
            src_nodes.append(edge[0])
            src_nodes.append(edge[1])
            des_nodes.append(edge[1])
            des_nodes.append(edge[0])  # append forward and backward of the edge connection
            weight.append(len(connection[edge]))
            weight.append(len(connection[edge]))  # append two edges weight

            if output_path:  # if user need to output
                for road in connection[edge]:
                    sheet.write(line, 0, road)
                    sheet.write(line, 1, line)  # id is same to line number
                    sheet.write(line, 2, "({}, {})".format(edge[0], edge[1]))
                    line += 1

        return {("tile", "road", "tile"):
                (torch.tensor(src_nodes), torch.tensor(des_nodes))}, torch.tensor(weight)


if __name__ == '__main__':
    sqt = SatQuadTree((116.0200, 39.6500, 116.7400, 40.2100))
    sqt.loadPoiData(r"../data/POI/transformed data/POI_data_2022-11-16_16-41-09.xls")
    osm_center = OsmCenter(sqt)
    # print("loading...")
    # osm_center.apply_file("../data/trajectory/Beijing.osm", locations=True)

    # osm_center.loadOsm("../data/trajectory/Beijing.osm")
    # edges = ox.graph_to_gdfs(osm_center.osm_graph, nodes=False, edges=True)
    # print(len(edges["geometry"]))
