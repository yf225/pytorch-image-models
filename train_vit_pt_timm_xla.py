# For TPU VM
"""
export XRT_TPU_CONFIG="localservice;0;localhost:51011"

rm -rf ./pytorch-image-models || true
git clone https://github.com/yf225/pytorch-image-models.git -b vit_dummy_data
cd pytorch-image-models && git pull

# Cloud Shell session disconnects very frequently.
# This saves stdout and stderr to local file on VM, to persist the output through multiple Cloud Shell sessions.
python3 train_vit_pt_timm_xla.py --bits=16 --micro_batch_size=64 >> output.txt 2>&1

# References
- https://github.com/pytorch/xla/blob/master/contrib/colab/multi-core-alexnet-fashion-mnist.ipynb
- https://github.com/pytorch/xla/blob/master/test/test_train_mp_imagenet.py
- https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm
"""

"""On AWS GPU node
conda activate torch-1.10

cd /fsx/users/willfeng/repos
rm -rf ./pytorch-image-models || true
git clone https://github.com/yf225/pytorch-image-models.git -b vit_dummy_data
cd pytorch-image-models && git pull

export PYTHONPATH=/fsx/users/willfeng/repos/pytorch-image-models:${PYTHONPATH}

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_vit_pt_timm_xla.py --bits=16 --micro_batch_size=2
"""

# Colab references
# - https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/getting-started.ipynb#scrollTo=yUB12htcqU9W
# - https://github.com/pytorch/xla/blob/master/contrib/colab/multi-core-alexnet-fashion-mnist.ipynb
# - https://github.com/pytorch/xla/issues/2587

# !pip uninstall -y torch || pip uninstall -y torch || pip uninstall -y torch || true
# !pip uninstall -y torchvision || true
# !pip uninstall -y torchtext || true
# !pip uninstall -y torchaudio || true
# # Have to use PyTorch 1.9 because of the following issues:
# # - https://github.com/pytorch/xla/issues/3211
# # - https://github.com/pytorch/xla/issues/3186
# !pip install cloud-tpu-client==0.10 torch==1.9.0 torchvision==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl

import argparse
import os
import sys
import time
import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
from torch_xla.distributed import parallel_loader as pl
from torch_xla.distributed import xla_multiprocessing as xmp

# !rm -rf ./pytorch-image-models || true
# !git clone https://github.com/yf225/pytorch-image-models.git -b vit_dummy_data
# !cd pytorch-image-models && git pull

import sys
if './pytorch-image-models' not in sys.path:
  sys.path.append('./pytorch-image-models')

from timm.utils import *
from timm.loss import *
from custom_vit_model import create_vit_model

DEBUG = False
VERBOSE = False

num_attention_heads = 16
hidden_size = 1280
num_layers = 32

image_size = 224
patch_size = 16  # Size of the patches to be extract from the input images

num_classes = 1000
num_epochs = 3

if "CUDA_VISIBLE_DEVICES" in os.environ:
  num_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
else:
  num_devices = 8

# if 'COLAB_TPU_ADDR' in os.environ:  # Colab, meaning debug mode
#   DEBUG = True

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

global_batch_size = micro_batch_size * num_devices

assert bits in [16, 32]
if bits == 16:
  default_dtype = torch.bfloat16
elif bits == 32:
  default_dtype = torch.float32

def xm_master_print_if_verbose(message):
  if VERBOSE:
    torch_xla.core.xla_model.master_print(message, flush=True)

class VitDummyDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_size, num_classes):
    self.dataset_size = dataset_size
    self.num_classes = num_classes

  def __len__(self):
    return self.dataset_size

  def __getitem__(self, index):
    return (torch.zeros(image_size, image_size, 3), torch.zeros(1).item())


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
    assert rearranged_input.shape[0] == micro_batch_size
    # rearranged_input = einops.rearrange(
    #     input,
    #     "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
    #     p1=self.patch_size[0],
    #     p2=self.patch_size[1],
    # )
    positions = torch.arange(start=0, end=self.num_patches, step=1).to(input.device)
    encoded = self.projection(rearranged_input) + self.position_embedding(positions)
    return encoded

def create_dataloader(dataset):
  sampler = torch.utils.data.distributed.DistributedSampler(
    dataset,
    num_replicas=torch_xla.core.xla_model.xrt_world_size(),
    rank=torch_xla.core.xla_model.get_ordinal(),
  )
  return torch.utils.data.DataLoader(
    dataset,
    batch_size=args.micro_batch_size,  # NOTE: this should be batch size per TPU core, re. https://discuss.pytorch.org/t/72769/2
    sampler=sampler,
    num_workers=1,
  )

def train_vit():
  assert torch_xla.core.xla_model.xrt_world_size() == num_devices
  torch_xla.core.xla_model.master_print("Working on: bits: {}, global_batch_size: {}, micro_batch_size per core: {}".format(bits, global_batch_size, micro_batch_size))
  # create train dataset
  train_dataset = VitDummyDataset(micro_batch_size * torch_xla.core.xla_model.xrt_world_size() * 10, num_classes)
  train_loader = create_dataloader(train_dataset)
  debug_train_loader = create_dataloader(train_dataset)
  sample_batch = next(iter(debug_train_loader))
  print("sample_batch[0].shape: ", sample_batch[0].shape)
  assert list(sample_batch[0].shape) == [args.micro_batch_size, image_size, image_size, 3]

  torch.manual_seed(42)

  model = create_vit_model(
    image_size=image_size,
    patch_size=patch_size,
    num_classes=num_classes,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_attention_heads=num_attention_heads,
  )

  device = torch_xla.core.xla_model.xla_device()
  model = model.to(device).train()
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
      # Note: optimizer_step uses the implicit Cloud TPU context to
      #  coordinate and synchronize gradient updates across processes.
      #  This means that each process's network has the same weights after
      #  this is called.
      # Warning: this coordination requires the actions performed in each
      #  process are the same. In more technical terms, the graph that
      #  PyTorch/XLA generates must be the same across processes.
      torch_xla.core.xla_model.optimizer_step(optimizer)  # Note: barrier=True not needed when using ParallelLoader
      step_duration = time.time() - step_start_time
      step_duration_list.append(step_duration)
      xm_master_print_if_verbose("Step {}, time taken: {}".format(step, step_duration))
      step_start_time = time.time()

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  for epoch in range(1, num_epochs + 1):
    xm_master_print_if_verbose('Epoch {} train begin'.format(epoch))
    train_loop_fn(train_device_loader, epoch)

  torch_xla.core.xla_model.master_print("bits: {}, global_batch_size: {}, micro_batch_size per core: {}, median step duration: {:.3f}".format(bits, global_batch_size, micro_batch_size, statistics.median(step_duration_list)))

# "Map function": acquires a corresponding Cloud TPU core, creates a tensor on it,
# and prints its core
def map_fn(index, flags):
  torch.manual_seed(42)

  # Acquires the (unique) Cloud TPU core corresponding to this process's index
  device = torch_xla.core.xla_model.xla_device()
  if VERBOSE:
    print("Process", index ,"is using", torch_xla.core.xla_model.xla_real_devices([str(device)])[0])

  # # Barrier to prevent master from exiting before workers connect.
  # torch_xla.core.xla_model.rendezvous('init')

  torch.set_default_dtype(default_dtype)
  train_vit()

# Spawns eight of the map functions, one for each of the eight cores on
# the Cloud TPU
flags = {}

if 'COLAB_TPU_ADDR' in os.environ:
  # Note: Colab only supports start_method='fork'
  xmp.spawn(map_fn, args=(flags,), nprocs=num_devices, start_method='fork')

if __name__ == "__main__":
  xmp.spawn(map_fn, args=(flags,), nprocs=num_devices, start_method='fork')
