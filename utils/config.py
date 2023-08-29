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
    if exp_id == -1:
        print("time: 2023.7.9 15:06")
        print("server: 114.251.134.132")
        print("description: 复现实验nyc")
        traj_config.lr = 0.000001
        traj_config.lr_decay = 0.95
        traj_config.epochs = 40
        traj_config.load_pretrained = False
        traj_config.model_loading_path = "./models/rosetea_useless.pth"
        traj_config.save_model = True
        traj_config.model_saving_path = "./models/rosetea_useless.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5
    if exp_id == -2:
        print("time: 2023.7.9 15:06")
        print("server: 114.251.134.132")
        print("description: tky tile 10")
        traj_config.lr = 0.00001
        traj_config.lr_decay = 0.95
        traj_config.epochs = 40
        traj_config.load_pretrained = False
        traj_config.model_loading_path = "./models/rosetea_useless_2.pth"
        traj_config.save_model = True
        traj_config.model_saving_path = "./models/rosetea_useless_2.pth"
        traj_config.tile_k = 10
        traj_config.poi_k = 5
    if exp_id == -3:
        print("time: 2023.7.9 15:06")
        print("server: 114.251.134.132")
        print("description: 复现实验nyc")
        traj_config.lr = 0.0001
        traj_config.lr_decay = 0.95
        traj_config.epochs = 40
        traj_config.load_pretrained = False
        traj_config.model_loading_path = "./models/rosetea_useless_3.pth"
        traj_config.save_model = True
        traj_config.model_saving_path = "./models/rosetea_useless_3.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5
    if exp_id == -4:
        print("time: 2023.7.9 15:06")
        print("server: 114.251.134.132")
        print("description: tky tile 20")
        traj_config.lr = 0.00001
        traj_config.lr_decay = 0.95
        traj_config.epochs = 40
        traj_config.load_pretrained = False
        traj_config.model_loading_path = "./models/rosetea_useless_4.pth"
        traj_config.save_model = True
        traj_config.model_saving_path = "./models/rosetea_useless_4.pth"
        traj_config.tile_k = 20
        traj_config.poi_k = 5
    if exp_id == -5:
        print("time: 2023.7.9 15:06")
        print("server: 114.251.134.132")
        print("description: 复现实验nyc")
        traj_config.lr = 0.00001
        traj_config.lr_decay = 0.95
        traj_config.epochs = 40
        traj_config.load_pretrained = False
        traj_config.model_loading_path = "./models/rosetea_useless_5.pth"
        traj_config.save_model = True
        traj_config.model_saving_path = "./models/rosetea_useless_5.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5
    if exp_id == -6:
        print("time: 2023.7.9 15:06")
        print("server: 114.251.134.132")
        print("description: tky tile 25")
        traj_config.lr = 0.00001
        traj_config.lr_decay = 0.95
        traj_config.epochs = 40
        traj_config.load_pretrained = False
        traj_config.model_loading_path = "./models/rosetea_useless_6.pth"
        traj_config.save_model = True
        traj_config.model_saving_path = "./models/rosetea_useless_6.pth"
        traj_config.tile_k = 25
        traj_config.poi_k = 5

    if exp_id == 0:
        print("time: 2023.7.9 15:06")
        print("server: 10.109.246.210")
        print("description: 消融实验-无卫星遥感-纽约")
        traj_config.lr = 0.0001
        traj_config.lr_decay = 0.95
        traj_config.load_pretrained = False
        traj_config.save_model = True
        traj_config.model_saving_path = "./models/rosetea_no_remote.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5
    if exp_id == 1:
        print("time: 2023.7.9 15:06")
        print("server: 10.109.246.210")
        print("description: 消融实验-无卫星遥感-东京")
        traj_config.lr = 0.0001
        traj_config.lr_decay = 0.95
        traj_config.load_pretrained = False
        traj_config.save_model = True
        traj_config.model_saving_path = "./models/rosetea_no_remote_tokyo.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5
    if exp_id == 2:
        print("time: 2023.7.9 15:06")
        print("server: 10.109.246.210")
        print("description: 消融实验-无历史-纽约")
        traj_config.device = 7
        traj_config.lr = 0.0001
        traj_config.lr_decay = 0.95
        traj_config.load_pretrained = False
        traj_config.save_model = True
        traj_config.model_saving_path = "./models/rosetea_no_qrp_NY.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5
    if exp_id == 3:
        print("time: 2023.7.9 15:06")
        print("server: 10.109.246.210")
        print("description: 消融实验-无历史-东京")
        traj_config.lr = 0.0001
        traj_config.lr_decay = 0.95
        traj_config.load_pretrained = False
        traj_config.save_model = True
        traj_config.model_saving_path = "./models/rosetea_no_qrp_T.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5
    if exp_id == 4:
        print("time: 2023.7.9 15:06")
        print("server: 114.251.134.132")
        print("description: 消融实验-无时空-纽约")
        traj_config.lr = 0.0001
        traj_config.lr_decay = 0.95
        traj_config.load_pretrained = False
        traj_config.save_model = True
        traj_config.model_saving_path = "./models/rosetea_no_ts_ny.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5
    if exp_id == 5:
        print("time: 2023.7.9 15:06")
        print("server: 114.251.134.132")
        print("description: 消融实验-无时空-东京")
        traj_config.lr = 0.0001
        traj_config.lr_decay = 0.95
        traj_config.load_pretrained = False
        traj_config.save_model = True
        traj_config.model_saving_path = "./models/rosetea_no_ts_tky.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5
    if exp_id == 6:
        print("time: 2023.7.9 15:06")
        print("server: 114.251.134.132")
        print("description: 消融实验-无分类-纽约")
        traj_config.lr = 0.0001
        traj_config.lr_decay = 0.95
        traj_config.load_pretrained = False
        traj_config.save_model = True
        traj_config.model_saving_path = "./models/rosetea_no_cate_ny.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5
    if exp_id == 7:
        print("time: 2023.7.9 15:06")
        print("server: 114.251.134.132")
        print("description: 消融实验-无分类-东京")
        traj_config.lr = 0.0001
        traj_config.lr_decay = 0.95
        traj_config.load_pretrained = False
        traj_config.save_model = True
        traj_config.model_saving_path = "./models/rosetea_no_cate_tky.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5
    if exp_id == 8:
        print("time: 2023.7.9 15:06")
        print("server: 114.251.134.132")
        print("description: 换loss: softmax")
        traj_config.lr = 0.0001
        traj_config.lr_decay = 0.9
        traj_config.load_pretrained = False
        traj_config.save_model = True
        if pp_config.poi_path == "./data/POI/poi.xls":
            traj_config.model_saving_path = "./models/rosetea_cross_entro_nyc.pth"
        else:
            traj_config.model_saving_path = "./models/rosetea_cross_entro_tky.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5
    if exp_id == 9:
        print("time: 2023.7.9 15:06")
        print("server: 114.251.134.132")
        print("description: 换loss: only cross entro")
        traj_config.lr = 0.0001
        traj_config.lr_decay = 0.9
        traj_config.load_pretrained = False
        traj_config.save_model = True
        if pp_config.poi_path == "./data/POI/poi.xls":
            traj_config.model_saving_path = "./models/rosetea_no_softmax_nyc.pth"
        else:
            traj_config.model_saving_path = "./models/rosetea_no_softmax_tky.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5
    if exp_id == 10:
        print("time: 2023.7.9 15:06")
        print("server: 114.251.134.132")
        print("description: 调参-")
        traj_config.lr = 0.0001
        traj_config.lr_decay = 0.9
        traj_config.load_pretrained = False
        traj_config.save_model = True
        if pp_config.poi_path == "./data/POI/poi.xls":
            traj_config.model_saving_path = "./models/rosetea_no_softmax_nyc.pth"
        else:
            traj_config.model_saving_path = "./models/rosetea_no_softmax_tky.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5
    if exp_id == 11:
        print("time: 2023.7.9 15:06")
        print("server: 114.251.134.132")
        print("description: 特征维度调参128")
        traj_config.lr = 0.00001
        traj_config.lr_decay = 0.95
        traj_config.save_model = True
        traj_config.d_model = 128
        if pp_config.poi_path == "./data/POI/poi.xls":
            traj_config.model_loading_path = "./models/rosetea_test_nyc.pth"
            traj_config.model_saving_path = "./models/rosetea_test_nyc.pth"
        else:
            traj_config.model_loading_path = "./models/rosetea_test_tky.pth"
            traj_config.model_saving_path = "./models/rosetea_test_tky.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5
    if exp_id == 12:
        print("time: 2023.7.9 15:06")
        print("server: 114.251.134.132")
        print("description: 特征维度调参1024")
        traj_config.lr = 0.00002
        traj_config.lr_decay = 0.95
        traj_config.save_model = True
        traj_config.d_model = 1024
        if pp_config.poi_path == "./data/POI/poi.xls":
            traj_config.model_loading_path = "./models/rosetea_test_nyc_2.pth"
            traj_config.model_saving_path = "./models/rosetea_test_nyc_2.pth"
        else:
            traj_config.model_loading_path = "./models/rosetea_test_tky_2.pth"
            traj_config.model_saving_path = "./models/rosetea_test_tky_2.pth"
        traj_config.tile_k = 15
        traj_config.poi_k = 5


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
        "data_file_path":           r"./data/trajectory/dataset_tsmc2014/dataset_TSMC2014_NYC.txt",
        "poi_path":                 r"./data/POI/poi.xls",
        "osm_path":                 r"./data/trajectory/NewYork.osm",
        "image_data_path":          r"./data/imagery/ImageryNewYorkMini",  # path to the painter-by-numbers images
        "grid_dir":                 r"./data/grid",
        # processed data
        "graph_path":               r"./data/graph/NewYork_graph.bin",
        "traj_pp_dir":              r"./data/trajectory/traj_datasets",
        "traj_processed_train":     r"./data/trajectory/PPed/NY_train_data2.pt",
        "traj_subgraph_train":      r"./data/trajectory/PPed/NY_train_subgraph2.pt",
        "traj_processed_val":       r"./data/trajectory/PPed/NY_val_data2.pt",
        "traj_subgraph_val":        r"./data/trajectory/PPed/NY_val_subgraph2.pt",
        "qtree_csv_path":           r"./data/grid/grid_2023-03-23_22-05-57.xls",
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
        "embeddings_saving_path":       "./data/imagery/newyork_image_embeddings.pt",    # where image embeddings are saved
        "model_loading_path":           "./models/rosetea.pth",
        "model_saving_path":            "./models/rosetea3.pth",

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
        "save_model":                   False,
        "device":                       args.device,
        "action":                       'train',
        # "seed":                         2022110916,
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
        "data_file_path":           r"./data/trajectory/dataset_tsmc2014/dataset_TSMC2014_TKY.txt",
        "poi_path":                 r"./data/POI/tky_poi.xls",
        "osm_path":                 r"./data/trajectory/Tokyo.osm",
        "image_data_path":          r"./data/imagery/ImageryTokyo",  # path to the painter-by-numbers images
        "grid_dir":                 r"./data/grid",
        # processed data
        "graph_path":               r"./data/graph/Tokyo_graph.bin",
        "traj_pp_dir":              r"./data/trajectory/TKY_traj_datasets",
        "traj_processed_train":     r"./data/trajectory/TKY_PPed/TKY_train_data.pt",
        "traj_subgraph_train":      r"./data/trajectory/TKY_PPed/TKY_train_subgraph.pt",
        "traj_processed_val":       r"./data/trajectory/TKY_PPed/TKY_val_data.pt",
        "traj_subgraph_val":        r"./data/trajectory/TKY_PPed/TKY_val_subgraph.pt",
        # "qtree_csv_path":           r"./data/grid/grid_2023-04-19_11-10-25.xls",
        "qtree_csv_path":           r"./data/grid/grid_2023-05-10_16-49-44.xls",
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
        "embeddings_saving_path":       "./data/imagery/tokyo_embeddings.pt",    # where image embeddings are saved
        "model_loading_path":           "./models/rosetea.pth",
        "model_saving_path":            "./models/rosetea2.pth",

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
        "save_model":                   False,
        "device":                       args.device,
        "action":                       'train',
        "seed":                         2022110916,
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
