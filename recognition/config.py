import numpy as np
import os
from easydict import EasyDict as edict

config = edict()

config.bn_mom = 0.9
config.workspace = 256
config.emb_size = 512
config.ckpt_embedding = True
config.net_se = 0
config.net_act = 'prelu'
config.net_unit = 3
config.net_input = 1
config.net_blocks = [1, 4, 6, 2]
config.net_output = 'E'
config.net_multiplier = 1.0
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
config.ce_loss = True
config.fc7_lr_mult = 1.0
config.fc7_wd_mult = 1.0
config.fc7_no_bias = False
config.max_steps = 100000
config.data_rand_mirror = True
config.data_cutoff = False
config.data_color = 0
config.data_images_filter = 0
config.count_flops = True
config.memonger = False  # not work now


# network settings
network = edict()

network.r100 = edict()
network.r100.net_name = 'fresnet'
network.r100.num_layers = 100

network.r100fc = edict()
network.r100fc.net_name = 'fresnet'
network.r100fc.num_layers = 100
network.r100fc.net_output = 'FC'

network.r50 = edict()
network.r50.net_name = 'fresnet'
network.r50.num_layers = 50

network.r50fc = edict()
network.r50fc.net_name = 'fresnet'
network.r50fc.num_layers = 50
network.r50fc.net_output = 'FC'

network.r50v1 = edict()
network.r50v1.net_name = 'fresnet'
network.r50v1.num_layers = 50
network.r50v1.net_unit = 1

network.d169 = edict()
network.d169.net_name = 'fdensenet'
network.d169.num_layers = 169
network.d169.per_batch_size = 64
network.d169.densenet_dropout = 0.0

network.d201 = edict()
network.d201.net_name = 'fdensenet'
network.d201.num_layers = 201
network.d201.per_batch_size = 64
network.d201.densenet_dropout = 0.0

network.y1 = edict()
network.y1.net_name = 'fmobilefacenet'
network.y1.emb_size = 128
network.y1.net_output = 'GDC'

network.y2 = edict()
network.y2.net_name = 'fmobilefacenet'
network.y2.emb_size = 256
network.y2.net_output = 'GDC'
network.y2.net_blocks = [2, 8, 16, 4]

network.m1 = edict()
network.m1.net_name = 'fmobilenet'
network.m1.emb_size = 256
network.m1.net_output = 'GDC'
network.m1.net_multiplier = 1.0

network.m05 = edict()
network.m05.net_name = 'fmobilenet'
network.m05.emb_size = 256
network.m05.net_output = 'GDC'
network.m05.net_multiplier = 0.5

network.mnas = edict()
network.mnas.net_name = 'fmnasnet'
network.mnas.emb_size = 256
network.mnas.net_output = 'GDC'
network.mnas.net_multiplier = 1.0

network.mnas05 = edict()
network.mnas05.net_name = 'fmnasnet'
network.mnas05.emb_size = 256
network.mnas05.net_output = 'GDC'
network.mnas05.net_multiplier = 0.5

network.mnas025 = edict()
network.mnas025.net_name = 'fmnasnet'
network.mnas025.emb_size = 256
network.mnas025.net_output = 'GDC'
network.mnas025.net_multiplier = 0.25

# dataset settings
dataset = edict()

dataset.vggface2_age_balanced = edict()
dataset.vggface2_age_balanced.dataset = 'vggface2_age_balanced'
dataset.vggface2_age_balanced.dataset_path = '../../ext_vol2/training_datasets/vggface2/zipped/train_112x112/age_balanced/'
dataset.vggface2_age_balanced.num_classes = 4936
dataset.vggface2_age_balanced.image_shape = (112, 112, 3)
dataset.vggface2_age_balanced.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.vggface2_old = edict()
dataset.vggface2_old.dataset = 'vggface2_old'
dataset.vggface2_old.dataset_path = '../../ext_vol2/training_datasets/vggface2/zipped/train_112x112/old/'
dataset.vggface2_old.num_classes = 2009
dataset.vggface2_old.image_shape = (112, 112, 3)
dataset.vggface2_old.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.vggface2_middle = edict()
dataset.vggface2_middle.dataset = 'vggface2_middle'
dataset.vggface2_middle.dataset_path = '../../ext_vol2/training_datasets/vggface2/zipped/train_112x112/middle/'
dataset.vggface2_middle.num_classes = 2009
dataset.vggface2_middle.image_shape = (112, 112, 3)
dataset.vggface2_middle.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.vggface2_young = edict()
dataset.vggface2_young.dataset = 'vggface2_young'
dataset.vggface2_young.dataset_path = '../../ext_vol2/training_datasets/vggface2/zipped/train_112x112/young/'
dataset.vggface2_young.num_classes = 2009
dataset.vggface2_young.image_shape = (112, 112, 3)
dataset.vggface2_young.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.vggface2_females = edict()
dataset.vggface2_females.dataset = 'vggface2_females'
dataset.vggface2_females.dataset_path = '../../ext_vol2/training_datasets/vggface2/zipped/train_112x112/females_only/'
dataset.vggface2_females.num_classes = 3477
dataset.vggface2_females.image_shape = (112, 112, 3)
dataset.vggface2_females.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.vggface2_males = edict()
dataset.vggface2_males.dataset = 'vggface2_males'
dataset.vggface2_males.dataset_path = '../../ext_vol2/training_datasets/vggface2/zipped/train_112x112/males_only/'
dataset.vggface2_males.num_classes = 3477
dataset.vggface2_males.image_shape = (112, 112, 3)
dataset.vggface2_males.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.vggface2_mixed_half = edict()
dataset.vggface2_mixed_half.dataset = 'vggface2_mixed_half'
dataset.vggface2_mixed_half.dataset_path = '../../ext_vol2/training_datasets/vggface2/zipped/train_112x112/mixed_half/'
dataset.vggface2_mixed_half.num_classes = 3478
dataset.vggface2_mixed_half.image_shape = (112, 112, 3)
dataset.vggface2_mixed_half.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.vggface2_m25_f75 = edict()
dataset.vggface2_m25_f75.dataset = 'vggface2_m25_f75'
dataset.vggface2_m25_f75.dataset_path = '../../ext_vol2/training_datasets/vggface2/zipped/train_112x112/males_25_females_75/'
dataset.vggface2_m25_f75.num_classes = 3477
dataset.vggface2_m25_f75.image_shape = (112, 112, 3)
dataset.vggface2_m25_f75.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.vggface2_m75_f25 = edict()
dataset.vggface2_m75_f25.dataset = 'vggface2_m75_f25'
dataset.vggface2_m75_f25.dataset_path = '../../ext_vol2/training_datasets/vggface2/zipped/train_112x112/males_75_females_25/'
dataset.vggface2_m75_f25.num_classes = 3477
dataset.vggface2_m75_f25.image_shape = (112, 112, 3)
dataset.vggface2_m75_f25.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.vggface2_mixed_full = edict()
dataset.vggface2_mixed_full.dataset = 'vggface2_mixed_full'
dataset.vggface2_mixed_full.dataset_path = '../../ext_vol2/training_datasets/vggface2/zipped/train_112x112/mixed_full/'
dataset.vggface2_mixed_full.num_classes = 6954
dataset.vggface2_mixed_full.image_shape = (112, 112, 3)
dataset.vggface2_mixed_full.val_targets = ['lfw', 'cfp_fp', 'agedb_30_balanced']

dataset.vggface2 = edict()
dataset.vggface2.dataset = 'vggface2'
dataset.vggface2.dataset_path = '../../ext_vol2/training_datasets/vggface2/zipped/train_112x112/full_dataset/'
dataset.vggface2.num_classes = 8631
dataset.vggface2.image_shape = (112, 112, 3)
dataset.vggface2.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.ms1m_v2 = edict()
dataset.ms1m_v2.dataset = 'ms1m_v2'
dataset.ms1m_v2.dataset_path = '../../ext_vol2/training_datasets/ms1m_v2/full_dataset/'
dataset.ms1m_v2.num_classes = 85742
dataset.ms1m_v2.image_shape = (112, 112, 3)
dataset.ms1m_v2.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

dataset.retina = edict()
dataset.retina.dataset = 'retina'
dataset.retina.dataset_path = '../datasets/ms1m-retinaface-t1'
dataset.retina.num_classes = 93431
dataset.retina.image_shape = (112, 112, 3)
dataset.retina.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

loss = edict()
loss.softmax = edict()
loss.softmax.loss_name = 'softmax'

loss.nsoftmax = edict()
loss.nsoftmax.loss_name = 'margin_softmax'
loss.nsoftmax.loss_s = 64.0
loss.nsoftmax.loss_m1 = 1.0
loss.nsoftmax.loss_m2 = 0.0
loss.nsoftmax.loss_m3 = 0.0

loss.arcface = edict()
loss.arcface.loss_name = 'margin_softmax'
loss.arcface.loss_s = 64.0
loss.arcface.loss_m1 = 1.0
loss.arcface.loss_m2 = 0.5
loss.arcface.loss_m3 = 0.0

loss.cosface = edict()
loss.cosface.loss_name = 'margin_softmax'
loss.cosface.loss_s = 64.0
loss.cosface.loss_m1 = 1.0
loss.cosface.loss_m2 = 0.0
loss.cosface.loss_m3 = 0.35

loss.combined = edict()
loss.combined.loss_name = 'margin_softmax'
loss.combined.loss_s = 64.0
loss.combined.loss_m1 = 1.0
loss.combined.loss_m2 = 0.3
loss.combined.loss_m3 = 0.2

loss.sphereface = edict()
loss.sphereface.loss_name = 'margin_softmax'
loss.sphereface.loss_s = 64.0
loss.sphereface.loss_m1 = 1.5
loss.sphereface.loss_m2 = 0.0
loss.sphereface.loss_m3 = 0.0

loss.triplet = edict()
loss.triplet.loss_name = 'triplet'
loss.triplet.images_per_identity = 5
loss.triplet.triplet_alpha = 0.3
loss.triplet.triplet_bag_size = 7200
loss.triplet.triplet_max_ap = 0.0
loss.triplet.per_batch_size = 60
loss.triplet.lr = 0.05

loss.atriplet = edict()
loss.atriplet.loss_name = 'atriplet'
loss.atriplet.images_per_identity = 5
loss.atriplet.triplet_alpha = 0.35
loss.atriplet.triplet_bag_size = 7200
loss.atriplet.triplet_max_ap = 0.0
loss.atriplet.per_batch_size = 60
loss.atriplet.lr = 0.05

# default settings
default = edict()

# default network
default.network = 'r100'
default.pretrained = ''
default.pretrained_epoch = 0
# default dataset
default.dataset = 'vggface2'
default.loss = 'arcface'
default.frequent = 20
default.verbose = 2000
default.kvstore = 'device'

default.end_epoch = 10000
default.lr = 0.001
default.wd = 0.0005
default.mom = 0.9
default.per_batch_size = 64
default.ckpt = 3
default.lr_steps = '50000,70000'
default.models_root = '../../ext_vol2/training/mxnet_scratch/'


def generate_config(_network, _dataset, _loss):
  for k, v in loss[_loss].items():
    config[k] = v
    if k in default:
      default[k] = v
  for k, v in network[_network].items():
    config[k] = v
    if k in default:
      default[k] = v
  for k, v in dataset[_dataset].items():
    config[k] = v
    if k in default:
      default[k] = v
  config.loss = _loss
  config.network = _network
  config.dataset = _dataset
  config.num_workers = 1
  if 'DMLC_NUM_WORKER' in os.environ:
    config.num_workers = int(os.environ['DMLC_NUM_WORKER'])
