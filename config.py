import argparse

parser = argparse.ArgumentParser(description='Description of your script')

# 添加命令行选项
parser.add_argument('-a', '--action', type=str, choices=['train', 'test'], help='Input which city', default='train')
parser.add_argument('-c', '--city', type=str, choices=['nyc', 'tky'], help='Input which city', default='nyc')
parser.add_argument('-d', '--device', type=int, help='which gpu device', default=5)
parser.add_argument('-e', '--experiment', type=int, help='which experiment number', default=-5)
parser.add_argument('-l', '--load', help='whether to load model.')


# 解析命令行参数
args = parser.parse_args()


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value


# configuration for data preparing
pp_config = None
ae_config = None
traj_config = None


def experimentSettings(exp_id=0):  # record to mem the running experiments' settings
    global traj_config
    return

# lon:40.550852-40.988332, lat:-74.274767--73.683825
def initNewYork():  # init chengdu config
    global pp_config, ae_config, traj_config
    pp_config = DotDict({
        "boundary":                 (-74.274767, 40.550852, -73.683825, 40.988332),
        "qtree": {
            "max_items": 50,
            "max_depth": 8,
        },
        # raw data
        "data_file_path":           r"",
        "poi_path":                 r"",
        "osm_path":                 r"",
        "image_data_path":          r"",  # path to the painter-by-numbers images
        "grid_dir":                 r"",
        # processed data
        "graph_path":               r"",
        "traj_pp_dir":              r"",
        "qtree_csv_path":           r"",
    })

    # configuration for whole traj model
    traj_config = DotDict({
        "traj_pp_dir":                  pp_config["traj_pp_dir"],
        "traj_processed_train":         pp_config["traj_processed_train"],
        "traj_subgraph_train":          pp_config["traj_subgraph_train"],
        "traj_processed_val":           pp_config["traj_processed_val"],
        "traj_subgraph_val":            pp_config["traj_subgraph_val"],
        "image_data_path":              pp_config["image_data_path"],
        "poi_path":                     pp_config["poi_path"],
        "embeddings_saving_path":       "",    # where image embeddings are saved
        "model_loading_path":           "./models/nyc.pth",
        "model_saving_path":            "./models/nyc.pth",

        "max_speed":                    1.7,                                    # km/min
        "degree_to_meter":              111.1,                                  # km/degree
        "loc_angle":                    30.5,                                   # angle degree (of beijing and equator)
        "time_interval":                0.5,                                    # min

        "d_model":                      512,
        "d_hidden":                     1024,
        "dropout":                      0.1,
        "multi_head":                   8,
        "gat_layer_num":                3,
        "trans_dec_layer":              7,
        "trans_enc_layer":              4,
        "hyper_length":                 256 * 16 * 16,
        "lr":                           0.00001,                                # learning rate
        "lr_decay":                     0,

        "tile_k":                       15,
        "poi_k":                        5,

        "epochs":                       40,                                     # number of epochs the model is trained
        "batch_size":                   8,                                      # size of a batch the training is done in
        "train_data_rate":              0.85,                                   # train data rate
        "shuffle":                      True,
        "load_pretrained":              False,
        "save_model":                   True,
        "device":                       args.device,
        "action":                       'train',
        "seed":                         42,
    })


def initTokyo():  # init chengdu config
    global pp_config, ae_config, traj_config
    pp_config = DotDict({
        # 35.510184690694565-35.867150422391695, lat:139.4708776473999-139.91259321570396
        "boundary":                 (139.470877, 35.510184, 139.912594, 35.867151),
        # (lon_min, lat_min, lon_max, lat_max)
        "qtree": {
            "max_items": 100,
            "max_depth": 8,
        },
        # raw data
        "data_file_path":           r"",
        "poi_path":                 r"",
        "osm_path":                 r"",
        "image_data_path":          r"",  # path to the painter-by-numbers images
        "grid_dir":                 r"",
        # processed data
        "graph_path":               r"",
        "traj_pp_dir":              r"",
        "qtree_csv_path":           r"",
    })

    # configuration for whole traj model
    traj_config = DotDict({
        "traj_pp_dir":                  pp_config["traj_pp_dir"],
        "traj_processed_train":         pp_config["traj_processed_train"],
        "traj_subgraph_train":          pp_config["traj_subgraph_train"],
        "traj_processed_val":           pp_config["traj_processed_val"],
        "traj_subgraph_val":            pp_config["traj_subgraph_val"],
        "image_data_path":              pp_config["image_data_path"],
        "poi_path":                     pp_config["poi_path"],
        "embeddings_saving_path":       "",    # where image embeddings are saved
        "model_loading_path":           "./models/tky.pth",
        "model_saving_path":            "./models/tky.pth",

        "max_speed":                    1.7,                                    # km/min
        "degree_to_meter":              111.1,                                  # km/degree
        "loc_angle":                    30.5,                                   # angle degree (of beijing and equator)
        "time_interval":                0.5,                                    # min

        "d_model":                      512,
        "d_hidden":                     1024,
        "dropout":                      0.1,
        "multi_head":                   8,
        "gat_layer_num":                3,
        "trans_dec_layer":              7,
        "trans_enc_layer":              4,
        "hyper_length":                 256 * 256,
        "lr":                           0.00002,                                # learning rate
        "lr_decay":                     0,

        "tile_k":                       15,
        "poi_k":                        5,

        "epochs":                       40,                                     # number of epochs the model is trained
        "batch_size":                   8,                                      # size of a batch the training is done in
        "train_data_rate":              0.85,                                   # train data rate
        "shuffle":                      True,
        "load_pretrained":              False,
        "save_model":                   True,
        "device":                       args.device,
        "action":                       'train',
        "seed":                         42,
    })


def initSettings():
    if args.city == "nyc":
        initNewYork()
    else:
        initTokyo()

    if args.experiment:
        experimentSettings(exp_id=args.experiment)
    traj_config.device = args.device
    traj_config.action = args.action
    traj_config.load_pretrained = args.load


initSettings()
