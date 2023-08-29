import os
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
        'std': transforms.Compose(
            [transforms.Resize(config.imgsize),
             transforms.RandomResizedCrop(256),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        'no_norm': transforms.Compose(
            [transforms.Resize(config.imgsize),
             transforms.RandomResizedCrop(256),
             transforms.ToTensor()]),
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


########################################################################
# part 2: Trajectory dataset
########################################################################
class TaxiTrajDataset(torch.utils.data.Dataset):
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


def TrajDataLoader(config):
    train_dataset_main = TaxiTrajDataset(config.traj_processed_train, config.traj_subgraph_train)
    val_dataset = TaxiTrajDataset(config.traj_processed_val, config.traj_subgraph_val)
    len_train = int(config.train_data_rate * len(train_dataset_main))
    len_val = len(train_dataset_main) - len_train
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset_main, [len_train, len_val])
    train_loader = DataLoader(train_dataset, batch_size=int(config.batch_size),
                              shuffle=config.shuffle, collate_fn=train_dataset_main.collect_fn)
    test_loader = DataLoader(test_dataset, batch_size=int(config.batch_size),
                             shuffle=config.shuffle, collate_fn=train_dataset_main.collect_fn)
    val_loader = DataLoader(val_dataset, batch_size=int(config.batch_size),
                            shuffle=config.shuffle, collate_fn=train_dataset_main.collect_fn)
    return train_loader, test_loader, val_loader


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
