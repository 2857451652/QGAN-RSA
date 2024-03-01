import os

import utils.config as conf
import train
from utils.config import pp_config, traj_config


if __name__ == '__main__':
    ######### train #########
    # traj_config.action = "test"
    if traj_config.action == "train":
        ret = train.train_rosetea(traj_config)
        if ret:  # if early stopped, need to train a second stage
            traj_config = conf.initSettings()
            traj_config.lr = traj_config.lr/2
            traj_config.tile_k = 15
            traj_config.epoch = 10
            traj_config.load_pretrained = True
            ret = train.train_rosetea(traj_config)
        traj_config = conf.initSettings()
        train.test_rosetea(traj_config, traj_config.model_saving_path)
    elif traj_config.action == "test":
        # LoadDatasetZip(traj_config.dataset_zip_path)
        train.test_rosetea(traj_config, traj_config.model_saving_path)
    ######### experiment #########
    # train.getEmbeddedTiles(traj_config, traj_config.model_saving_path)
