# Colab references
# - https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/getting-started.ipynb#scrollTo=yUB12htcqU9W
# - https://github.com/pytorch/xla/blob/master/contrib/colab/multi-core-alexnet-fashion-mnist.ipynb
# - https://github.com/pytorch/xla/issues/2587

# === Colab 1st cell ===
!pip uninstall -y torch || pip uninstall -y torch || pip uninstall -y torch || true
!pip uninstall -y torchvision || true
!pip uninstall -y torchtext || true
!pip uninstall -y torchaudio || true
# Have to use PyTorch 1.9 because of the following issues:
# - https://github.com/pytorch/xla/issues/3211
# - https://github.com/pytorch/xla/issues/3186
!pip install cloud-tpu-client==0.10 torch==1.9.0 torchvision==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl

# === Colab 2nd cell ===
import torch_xla.core.xla_model as xm
assert "xla:1" in str(xm.xla_device())

# === Colab 3rd cell ===
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

# === Colab 4th cell ===
!rm -rf ./pytorch-image-models || true
!git clone https://github.com/yf225/pytorch-image-models.git -b vit_dummy_data
!cd pytorch-image-models && git pull

import sys
if './pytorch-image-models' not in sys.path:
  sys.path.append('./pytorch-image-models')

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint,\
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
from timm.models.helpers import build_model_with_cfg
from timm.models.vision_transformer import VisionTransformer

# === Colab 5th cell ===

DEBUG = True
VERBOSE = False

num_attention_heads = 16
hidden_size = 1280
num_layers = 32

image_size = 224
patch_size = 16  # Size of the patches to be extract from the input images

num_classes = 1000
num_epochs = 10

if 'COLAB_TPU_ADDR' in os.environ:  # Colab, meaning debug mode
  DEBUG = True

if DEBUG:
  print("Overwriting hyperparams since we are in debug mode...")
  num_attention_heads = 1
  hidden_size = 128
  num_layers = 1
  bits = 16
  micro_batch_size = 1
else:
  parser = argparse.ArgumentParser()
  parser.add_argument("--bits", type=int)
  parser.add_argument("--micro_batch_size", type=int)
  args = parser.parse_args()
  bits = args.bits
  micro_batch_size = args.micro_batch_size

global_batch_size = micro_batch_size * 8

assert bits in [16, 32]
if bits == 16:
  default_dtype = torch.bfloat16
elif bits == 32:
  default_dtype = torch.float32

def xm_master_print_if_verbose(message):
  if VERBOSE:
    xm.master_print(message, flush=True)

class VitDummyDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_size, crop_size, num_classes):
    self.dataset_size = dataset_size
    self.crop_size = crop_size
    self.num_classes = num_classes

  def __len__(self):
    return self.dataset_size

  def __getitem__(self, index):
    return (torch.rand(3, self.crop_size, self.crop_size), torch.randint(self.num_classes, (1,)).item())


# NOTE: need this to be consistent with TF-TPU impl
class PatchEncoder(torch.nn.Module):
  def __init__(self, img_size, patch_size, in_chans, embed_dim):
    super().__init__()
    img_size = (img_size, img_size)
    patch_size = (patch_size, patch_size)
    self.patch_size = patch_size
    self.in_chans = in_chans
    self.flatten_dim = self.patch_size[0] * self.patch_size[1] * in_chans
    self.img_size = img_size
    self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
    self.num_patches = self.grid_size[0] * self.grid_size[1]
    self.projection = torch.nn.Linear(
        self.flatten_dim, embed_dim
    )
    self.position_embedding = torch.nn.Embedding(
        num_embeddings=self.num_patches, embedding_dim=embed_dim
    )

  def forward(self, input):
    rearranged_input = input.view(-1, self.grid_size[0] * self.grid_size[1], self.patch_size[0] * self.patch_size[1] * self.in_chans)
    # rearranged_input = einops.rearrange(
    #     input,
    #     "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
    #     p1=self.patch_size[0],
    #     p2=self.patch_size[1],
    # )
    positions = torch.arange(start=0, end=self.num_patches, step=1).to(input.device)
    encoded = self.projection(rearranged_input) + self.position_embedding(positions)
    return encoded

def train_vit():
  # create train dataset
  train_dataset = VitDummyDataset(global_batch_size * 10, image_size, num_classes)
  # train_loader = create_loader(
  #   train_dataset,
  #   input_size=(3, 224, 224),
  #   batch_size=micro_batch_size * 8,
  #   is_training=True,
  #   no_aug=True,
  #   use_prefetcher=False,
  # )
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=micro_batch_size * 8,
      num_workers=2)

  torch.manual_seed(42)

  model = build_model_with_cfg(
    VisionTransformer,
    "vit_huge_patch{}_{}".format(patch_size, image_size),
    pretrained=False,
    default_cfg={},
    representation_size=None,  # NOTE: matching vit_tf_tpu_v2.py impl
    **dict(
      patch_size=patch_size, embed_dim=hidden_size, depth=num_layers, num_heads=num_attention_heads, num_classes=num_classes,
      embed_layer=PatchEncoder,
    )
  )

  device = xm.xla_device()
  model = model.to(device)
  optim_cls = optim.Adam
  optimizer = optim_cls(
      model.parameters(),
      lr=0.001,
  )
  loss_fn = nn.CrossEntropyLoss()

  step_duration_list = []

  def train_loop_fn(loader, epoch):
    model.train()
    step_start_time = time.time()
    for step, (data, target) in enumerate(loader):
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      xm.optimizer_step(optimizer)
      step_duration = time.time() - step_start_time
      step_duration_list.append(step_duration)
      xm_master_print_if_verbose("Step {}, time taken: {}".format(step, step_duration))
      step_start_time = time.time()

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  for epoch in range(1, num_epochs + 1):
    xm_master_print_if_verbose('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    train_loop_fn(train_device_loader, epoch)

  xm.master_print("median step duration: {:.3f}".format(statistics.median(step_duration_list)))

# "Map function": acquires a corresponding Cloud TPU core, creates a tensor on it,
# and prints its core
def map_fn(index, flags):
  # Sets a common random seed - both for initialization and ensuring graph is the same
  torch.manual_seed(42)

  # Acquires the (unique) Cloud TPU core corresponding to this process's index
  device = xm.xla_device()
  print("Process", index ,"is using", xm.xla_real_devices([str(device)])[0])

  # # Barrier to prevent master from exiting before workers connect.
  # xm.rendezvous('init')

  torch.set_default_dtype(default_dtype)
  train_vit()

# Spawns eight of the map functions, one for each of the eight cores on
# the Cloud TPU
flags = {}
# Note: Colab only supports start_method='fork'
xmp.spawn(map_fn, args=(flags,), nprocs=8, start_method='fork')
