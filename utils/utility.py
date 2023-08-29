import math
import torch.nn as nn
import torch
from tqdm import tqdm

from .error import selfDefinedError


class GeoConvert:
    # 将WGS84坐标系转换为GCJ02火星坐标系
    # 返回坐标字符串lon,lat
    def __init__(self):
        self.__pi = 3.1415926535897932384626
        self.__a = 6378245.0
        self.__ee = 0.00669342162296594323

    def wgs84_to_gcj02(self, lon, lat):
        if self.out_of_china(lon, lat):
            raise selfDefinedError("input location out of China!")
        dLat = self.__transform_lat(lon - 105.0, lat - 35.0)
        dLon = self.__transform_lon(lon - 105.0, lat - 35.0)
        radLat = lat / 180.0 * self.__pi
        magic = math.sin(radLat)
        magic = 1 - self.__ee * magic * magic
        sqrtMagic = math.sqrt(magic)
        dLat = (dLat * 180.0) / ((self.__a * (1 - self.__ee)) / (magic * sqrtMagic) * self.__pi)
        dLon = (dLon * 180.0) / (self.__a / sqrtMagic * math.cos(radLat) * self.__pi)
        mgLat = lat + dLat
        mgLon = lon + dLon
        return mgLon, mgLat

    def __transform_lat(self, x, y):
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * self.__pi) + 20.0 * math.sin(2.0 * x * self.__pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * self.__pi) + 40.0 * math.sin(y / 3.0 * self.__pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * self.__pi) + 320 * math.sin(y * self.__pi / 30.0)) * 2.0 / 3.0
        return ret

    def __transform_lon(self, x, y):
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * self.__pi) + 20.0 * math.sin(2.0 * x * self.__pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * self.__pi) + 40.0 * math.sin(x / 3.0 * self.__pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * self.__pi) + 300.0 * math.sin(x / 30.0 * self.__pi)) * 2.0 / 3.0
        return ret

    def out_of_china(self, lon, lat):
        if lon < 72.004 or lon > 137.8347:
            return True
        if lat < 0.8293 or lat > 55.8271:
            return True
        return False


class tqdm_dummy:
    def __init__(self, total):
        self.total = total
        self.count = 1
        self.postfix = None

    def set_postfix(self, postfix):
        self.postfix = postfix

    def update(self, renew):
        self.count += renew
        #if self.count % (self.total//20) == 0:
            #print("iter:{} score:{}".format(self.count, self.postfix))

    def close(self):
        print(self.postfix)
        return


class LocToTile:
    def __init__(self, bbox, scale, saving_path):
        self.bbox = {"xmin": bbox[0], "ymin": bbox[1],
                     "xmax": bbox[2], "ymax": bbox[3]}  # x means longitude, y means latitude
        self.scale = scale
        self.col_num = (self.bbox["xmax"] - self.bbox["xmin"]) // scale + 1
        self.row_num = (self.bbox["ymax"] - self.bbox["ymin"]) // scale + 1
        self.id_map = []
        self.id_saving_path = saving_path

    def generateIdMap(self, traj_file):  # generate id map by going through all the points
        ids = []
        f = open(traj_file)
        for line in tqdm(f.readlines(), desc='Reading file', unit='lines'):
            splited_line = line.split(',')
            lon = float(splited_line[3])
            lat = float(splited_line[4])
            if self.inBox(lon, lat):
                inner_id = self.findInnerTileId(lon, lat)
                if inner_id not in ids:
                    ids.append(inner_id)
            else:
                print("ha?")
        self.id_map = sorted(ids)
        print("total {} ids.\n"
              "max id:{} min id:{}".format(len(self.id_map), max(self.id_map), min(self.id_map)))

    def findInnerTileId(self, lon, lat):
        col = (lon - self.bbox["xmin"])//self.scale
        row = (lat - self.bbox["ymin"])//self.scale
        id = int(self.col_num * row + col)
        return id

    def inBox(self, lon, lat):
        return (self.bbox["xmin"] <= lon <= self.bbox["xmax"]) and \
               (self.bbox["ymin"] <= lat <= self.bbox["ymax"])

    def findInnerTileCenter(self, inner_id):
        row = inner_id // self.col_num
        col = inner_id % self.col_num
        lon = self.bbox["xmin"] + (col + 0.5) * self.scale
        lat = self.bbox["ymin"] + (row + 0.5) * self.scale
        if lon > self.bbox["xmax"]:
            lon = self.bbox["xmax"]
        if lat > self.bbox["ymax"]:
            lat = self.bbox["ymax"]
        return lon, lat

    def retTileIdWithLoc(self, lon, lat):
        inner_id = self.findInnerTileId(lon, lat)
        lon, lat = self.findInnerTileCenter(inner_id)
        return self.id_map.index(inner_id), (lon, lat)

    def saveIdMap(self):
        file = open(self.id_saving_path, 'w')
        ids = [str(i) for i in self.id_map]
        content = ','.join(ids)
        file.write(content)
        file.close()

    def loadIdMap(self):
        file = open(self.id_saving_path)
        content = file.readline()
        self.id_map = content.split(',')
        file.close()


if __name__ == "__main__":
    gc = GeoConvert()
    print(gc.wgs84_to_gcj02(116.4497585, 39.9311036))