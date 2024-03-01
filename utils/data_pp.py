import time

import os
from tqdm import tqdm
import random

########################################################################
# Trajectory data ( for New York and Tokyo )
########################################################################
def split(s, node):
    result = []
    in_quotes = False
    start = 0
    for i, c in enumerate(s):
        if c == '"':
            in_quotes = not in_quotes
        elif c == node and not in_quotes:
            result.append(s[start:i])
            start = i + 1
    result.append(s[start:])
    return result


def getUserTraj(path, dataset="foursquare"):
    if dataset == "foursquare":
        encoding = "Latin-1"
    elif dataset == "gowalla":
        encoding = "gbk"
    else:
        encoding = "utf-8"

    file = open(path, encoding=encoding)
    print("opened successfully")
    # line = file.readline()
    count = 0
    user_list = []
    traj_list = []
    poi_dict = {}
    poi_cate = []
    range_bbox = [-124.409591, 32.534156, -114.131211, 42.009518]  # California
    # range_bbox = [-87.634643, 24.396308, -79.974307, 31.000888]  # florida
    # spilt user traj from the large file
    for line in tqdm(file):
        if line[-1] == '\n':
            line = line[:-1]  # get rid of '\n'

        if dataset == "foursquare":
            splited_line = line.strip().split('\t')
            user_id = splited_line[0]
            poi_id = splited_line[1]
            poi_c_id = splited_line[2]
            lat = float(splited_line[4])
            lon = float(splited_line[5])
            time_offset = int(splited_line[-2])
            tm = time.strptime(splited_line[-1], "%a %b %d %H:%M:%S +0000 %Y")
            tm = time.localtime(time.mktime(tm) + time_offset*60)  # cal an offset to the UTC time
        elif dataset == "gowalla":
            splited_line = line.strip().split('\t')
            user_id, t, lat, lon, poi_id = splited_line
            lat = float(lat)
            lon = float(lon)
            tm = time.strptime(t, '%Y-%m-%dT%H:%M:%SZ')
            # tm = time.mktime(tm)
        elif dataset == "weeplace":
            if line == "userid,placeid,datetime,lat,lon,city,category":
                continue
            splited_line = split(line, ',')
            user_id, poi_id, t, lat, lon, _, poi_c_id = splited_line
            lat = float(lat)
            lon = float(lon)
            tm = time.strptime(t, '%Y-%m-%dT%H:%M:%S')
            # tm = time.mktime(tm)
            if not (range_bbox[0] <= lon <= range_bbox[2] and range_bbox[1] <= lat <= range_bbox[3]):
                continue

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
            traj_list[user_list.index(user_id)].append((poi_id, lon, lat, tm, poi_cate_id, line))
        else:
            user_list.append(user_id)
            traj_list.append([(poi_id, lon, lat, tm, poi_cate_id, line)])
        # line = file.readline()
        count += 1
        # if count >= 1000000:
        #     break
    print("read {} records.".format(count))
    print("poi_num: ", len(poi_dict), "cate_num: ", len(poi_cate))
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
            tm = record[3]
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

        if len(user_traj) >= 5:  # record the user trajs
            splited_traj[user_id] = user_traj
            traj_count += len(user_traj)
    print("{} users, {} trajs".format(len(splited_traj.keys()), traj_count))
    return splited_traj


def splitTVT_history(splited_traj, type=0):  # split the traj dict into training set and val sets
    if type == 0:  # 仅对train和val进行随机
        train_val_dataset = []
        test_dataset = []
        train_val_history = []
        test_history = []
        for user_id in tqdm(splited_traj):
            user_trajs = splited_traj[user_id][1:]
            split_num = int(len(user_trajs) * 0.9)
            train_val_dataset += user_trajs[:split_num]  # get rid of the first traj, consider it as history
            test_dataset += user_trajs[split_num:]
            # len(history_traj) is shorter than len(user_trajs) by 1
        for user_id in tqdm(splited_traj):
            history_traj = []
            history = []
            for traj in splited_traj[user_id][:-1]:
                history += traj
                if len(history) > 100:  # avoid history traj too long
                    history_traj.append(history[-100:])
                else:
                    history_traj.append(history)
            split_num = int((len(splited_traj[user_id])-1) * 0.9)
            train_val_history += history_traj[:split_num]  # get rid of the first traj, consider it as history
            test_history += history_traj[split_num:]
        random.seed(42)
        random.shuffle(train_val_dataset)
        random.seed(42)
        random.shuffle(train_val_history)
        train_index = int(len(train_val_dataset) * 0.85)
        train_dataset = train_val_dataset[:train_index]
        val_dataset = train_val_dataset[train_index:]
        train_history = train_val_history[:train_index]
        val_history = train_val_history[train_index:]
    else:  # 全随机
        dataset = []
        data_history = []
        for user_id in tqdm(splited_traj):
            user_trajs = splited_traj[user_id][1:]
            dataset += user_trajs
            history_traj = []
            history = []
            for traj in splited_traj[user_id][:-1]:
                history += traj
                if len(history) > 100:  # avoid history traj too long
                    history_traj.append(history[-100:])
                else:
                    history_traj.append(history)
            data_history += history_traj  # get rid of the first traj, consider it as history
        random.seed(42)
        random.shuffle(dataset)
        random.seed(42)
        random.shuffle(data_history)
        train_index = int(len(dataset) * 0.8)
        val_index = int(len(dataset) * 0.9)
        train_dataset = dataset[:train_index]
        val_dataset = dataset[train_index:val_index]
        test_dataset = dataset[val_index:]
        train_history = data_history[:train_index]
        val_history = data_history[train_index:val_index]
        test_history = data_history[val_index:]

    print("{} trajs, {} trajs, {} trajs in datasets.".format(
        len(train_dataset), len(val_dataset), len(test_dataset))
    )
    return train_dataset, val_dataset, test_dataset, \
           train_history, val_history, test_history,


def saveUnifiedDataset(dataset, save_path, encoding="gbk"):
    target_file = open(save_path, "w", encoding=encoding)
    for traj in tqdm(dataset):
        for record in traj:
            target_file.write(record[-1]+"\n")
    target_file.close()
    print("saved dataset {}".format(save_path))
    return


def time_projection(tm):
    tid = tm.tm_hour * 2
    tid += (tm.tm_min > 30)
    return tid


def saveMyDataset(dataset, poi_list, save_path):
    target_file = open(save_path, "w")
    for traj in tqdm(dataset):
        line = []
        for record in traj:
            item = "{}|{}|{:.6f}|{:.6f}|{}".format(poi_list.index(record[0]), time_projection(record[3]),
                                                   record[1], record[2], record[4])
            line.append(item)
        line = ",".join(line) + "\n"
        target_file.write(line)
    target_file.close()
    print("saved dataset {}".format(save_path))
    return


def savePoiPoints(poi_dict, poi_list, save_path):
    target_file = open(save_path, "w")
    target_file.write("id,longitude,latitude,poi_id,cate\n")
    for i, poi_id in tqdm(enumerate(poi_list)):
        record = poi_dict[poi_id]
        target_file.write("{},{:.6f},{:.6f},{},{}\n".format(i, record[0], record[1], poi_id, record[2]))
    # target_file.write("{},{:.6f},{:.6f},{},{}\n".format(len(poi_list), 0, 0, 0, 0))
    target_file.close()
    print("saved poi points to {}".format(save_path))
    return


def getPoiList(dataset, poi_dict, type=1):
    poi_list = []
    if type == 1:
        for traj in tqdm(dataset):
            for record in traj:
                if record[0] not in poi_list:
                    poi_list.append(record[0])
    else:
        for user in tqdm(dataset):
            for traj in dataset[user]:
                for record in traj:
                    if record[0] not in poi_list:
                        poi_list.append(record[0])
    ret_list = [i for i in poi_dict if i in poi_list]
    print("only {} pois".format(len(ret_list)))
    return ret_list


def ReEncode(dataset, poi_list):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            record = dataset[i][j]
            if record[0] not in poi_list:
                new_record = (len(poi_list), record[1], record[2], record[3], record[4])
            else:
                new_record = (poi_list.index(record[0]), time_projection(record[3]), record[1], record[2], record[4])
            dataset[i][j] = new_record
    return dataset


def MakeDataset(path, save_dir, dataset="foursquare"):
    user_list, traj_list, poi_dict = getUserTraj(path, dataset)
    splited_traj = trajSplit(user_list, traj_list)
    train_dataset, val_dataset, test_dataset, \
        train_history, val_history, test_history = splitTVT_history(splited_traj, 1)
    poi_list = getPoiList(splited_traj, poi_dict, 2)
    saveMyDataset(train_dataset, poi_list, os.path.join(save_dir, "train.txt"))
    saveMyDataset(val_dataset, poi_list, os.path.join(save_dir, "valid.txt"))
    saveMyDataset(test_dataset, poi_list, os.path.join(save_dir, "test.txt"))
    saveMyDataset(train_history, poi_list, os.path.join(save_dir, "train_history.txt"))
    saveMyDataset(val_history, poi_list, os.path.join(save_dir, "val_history.txt"))
    saveMyDataset(test_history, poi_list, os.path.join(save_dir, "test_history.txt"))
    savePoiPoints(poi_dict, poi_list, os.path.join(save_dir, "poi.txt"))


if __name__ == "__main__":
    path = [""][0]
    save_dir = [""][0]
    MakeDataset(path, save_dir, dataset="foursquare")

