import math
from .config import traj_config


def calDistance(point_1, point_2):
    (lon_1, lat_1) = point_1
    (lon_2, lat_2) = point_2
    loc_angle = (lat_1 + lat_2)/2
    lon_rate = math.cos(math.radians(loc_angle))
    x = abs(lon_1 - lon_2) * traj_config.degree_to_meter * lon_rate
    y = abs(lat_1 - lat_2) * traj_config.degree_to_meter
    distance = math.sqrt(x ** 2 + y ** 2)
    return distance


def calSpeed(point_1, point_2):
    distance = calDistance(point_1, point_2)
    return distance / traj_config.time_interval


def calMeterToDegree(meter):
    lon_rate = math.cos(math.radians(traj_config.loc_angle))
    lon = meter / (traj_config.degree_to_meter * lon_rate)
    lat = meter / traj_config.degree_to_meter
    return lon, lat


def calNeighborSpace(point):
    distance = traj_config.max_speed * traj_config.time_interval
    # this equation refers to https://zhidao.baidu.com/question/1549354221298208507.html
    lon_rate = math.cos(math.radians(traj_config.loc_angle))
    lon_gap = distance / (traj_config.degree_to_meter * lon_rate)
    lat_gap = distance / traj_config.degree_to_meter
    bbox = [0, 0, 0, 0]  # lon_min, lat_min, lon_max, lat_max
    bbox[0] = point[0] - lon_gap
    bbox[1] = point[1] - lat_gap
    bbox[2] = point[0] + lon_gap
    bbox[3] = point[1] + lat_gap
    return bbox
