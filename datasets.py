import os
import pickle

import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from utils.config import traj_config
from utils.quadtree import SatQuadTree
from utils.trajtools import calNeighborSpace
from utils.utility import selfDefinedError
import re
import dgl


########################################################################
# part 1: Satellite imagery dataset
########################################################################
def isimage(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def image_loader(image_path, transformation):
    """
    loads an image
    :param image_name: the path of the image
    :param transformation: the transformation done on the image
    :return: the image on the current device
    """
    image = Image.open(image_path).convert('RGB')
    image = transformation(image)
    return image


def SatImageDataloader(config):
    loaders = {
        # 'std': transforms.Compose(
        #     [transforms.Resize(config.imgsize),
        #      transforms.RandomResizedCrop(256),
        #      transforms.ToTensor(),
        #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        # 'no_norm': transforms.Compose(
        #     [transforms.Resize(config.imgsize),
        #      transforms.RandomResizedCrop(256),
        #      transforms.ToTensor()]),
        'no_trans': transforms.Compose(
            [transforms.ToTensor()])
    }
    sat_dataset = SatImageDataset(config.image_data_path, loaders[config.loader])
    dataloader = DataLoader(sat_dataset, batch_size=int(config.batch_size),
                            shuffle=config.shuffle)
    return dataloader


class SatImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, loader):
        self.img_dir = img_dir
        self.img_list = [x for x in os.listdir(img_dir) if isimage(x)]
        self.sort_img_list()  # sort the above list by image id
        self.loader = loader
        self.len = len(self.img_list)

    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir, self.img_list[index])
        image = image_loader(img_name, self.loader)
        sample = {'image': image, "id": self.img_list[index]}
        return sample

    def __len__(self):
        return self.len

    def sort_img_list(self):
        self.img_list.sort(key=lambda l: int(re.findall('\d+', l)[0]))
        return


class ContrastDataset:
    def __init__(self, path="./data/POI/contrast_nyc.pt", num=100):
        self.neg_ix = torch.load(path)
        self.num = num

    def random_sample(self):
        contrast = []
        for row in self.neg_ix:
            rand_i = torch.randint(len(row), (self.num,))
            selected = row[rand_i]
            contrast.append(selected)
        return torch.stack(contrast)


########################################################################
# part 2: Trajectory dataset
########################################################################
class TrajDataset(torch.utils.data.Dataset):
    def __init__(self, pp_data_path, pp_subgraph_path):
        self.pp_data = torch.load(pp_data_path)
        self.pp_subgraph_path = pp_subgraph_path
        return

    def __getitem__(self, item):
        [subgraph], _ = dgl.load_graphs(self.pp_subgraph_path, [item])
        return self.pp_data[item] + (subgraph,)

    def __len__(self):
        return len(self.pp_data)

    @staticmethod
    def collect_fn(batch):
        padding = -1
        b_leaves_seq, b_pos_seq, b_poi_traj, b_time_stamp, b_history_traj, b_target, \
            b_subgraph_nodes, b_subgraph = zip(*batch)

        traj_recovery = torch.tensor([len(nodes) for nodes in b_leaves_seq])  # used to recover leaves
        b_leaves_seq = torch.cat(b_leaves_seq, 0)  # group a batch
        b_poi_traj = torch.cat(b_poi_traj, 0)  # group a batch
        b_time_stamp = pad_sequence(b_time_stamp, batch_first=True, padding_value=padding)
        b_pos_seq = pad_sequence(b_pos_seq, batch_first=True, padding_value=padding)
        # transpose pos_seq from (batch, length, 2) to (2, batch, length)
        b_pos_seq = b_pos_seq.transpose(0, 1).transpose(0, 2)

        history_traj_recovery = torch.tensor([len(nodes) for nodes in b_history_traj])  # used to recover leaves
        b_history_traj = torch.cat(b_history_traj, 0)

        subgraph_recovery = torch.tensor([len(nodes) for nodes in b_subgraph_nodes])  # used to recover subgraph
        b_subgraph_nodes = torch.cat(b_subgraph_nodes, 0)  # group a batch
        b_subgraph = dgl.batch(list(b_subgraph))

        b_target_tile = []
        b_target_poi = []
        for target in b_target:
            b_target_tile.append(target["tile_id"])
            b_target_poi.append(target["poi_id"])
        b_target_tile = torch.cat(b_target_tile, 0)
        b_target_poi = torch.cat(b_target_poi, 0)

        return traj_recovery, b_leaves_seq, b_poi_traj, b_pos_seq, b_time_stamp, \
            history_traj_recovery, b_history_traj, \
            subgraph_recovery, b_subgraph_nodes, b_subgraph, \
            b_target_tile, b_target_poi

dataset_zip = None
def TrajDataLoader(config):
    global dataset_zip
    if dataset_zip is None:
        train_dataset = TrajDataset(os.path.join(config.traj_pp_dir, "train.pt"),
                                    os.path.join(config.traj_pp_dir, "train_graph.pt"))
        val_dataset = TrajDataset(os.path.join(config.traj_pp_dir, "valid.pt"),
                                  os.path.join(config.traj_pp_dir, "valid_graph.pt"))
        test_dataset = TrajDataset(os.path.join(config.traj_pp_dir, "test.pt"),
                                   os.path.join(config.traj_pp_dir, "test_graph.pt"))
        collect_fn = train_dataset.collect_fn
        dataset_zip = (train_dataset, val_dataset, test_dataset, collect_fn)
        print("created datasets.")
    else:
        train_dataset, val_dataset, test_dataset, collect_fn = dataset_zip
    train_loader = DataLoader(train_dataset, batch_size=int(config.batch_size),
                              shuffle=config.shuffle, collate_fn=collect_fn)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=config.shuffle, collate_fn=collect_fn)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=config.shuffle, collate_fn=collect_fn)
    print("created loaders.")
    return train_loader, val_loader, test_loader


def SaveDatasetZip(path):
    global dataset_zip
    if dataset_zip is not None:
        with open(path, 'wb') as file:
            pickle.dump(dataset_zip, file)
    print("saved!")


def LoadDatasetZip(path):
    global dataset_zip
    with open(path, 'rb') as file:
        dataset_zip = pickle.load(file)


def pad_2d_tensor(tensor_list, tgt_size, padding_value=0):
    if isinstance(tgt_size, int):
        mould = torch.ones(tgt_size, tgt_size)
    elif isinstance(tgt_size, tuple):
        mould = torch.ones(tgt_size)
    else:
        raise selfDefinedError("tgt_size should be int or tuple!")
    mould *= padding_value  # initialize to padding value
    ret = []
    for tensor in tensor_list:
        padded = mould
        padded[:tensor.shape[0], :tensor.shape[1]] = tensor
        ret.append(padded)
    torch.cat(ret)
    return ret
