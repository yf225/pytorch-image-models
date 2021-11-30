#!/usr/bin/env python3

"""On AWS GPU node
conda activate torch-1.10

cd /fsx/users/willfeng/repos
rm -rf ./pytorch-image-models || true
git clone https://github.com/yf225/pytorch-image-models.git -b vit_dummy_data
cd pytorch-image-models && git pull

export PYTHONPATH=/fsx/users/willfeng/repos/pytorch-image-models:${PYTHONPATH}

python -m torch.distributed.launch --nproc_per_node=4 \
train_vit_pt_timm_gpu.py --mode=graph --micro_batch_size=2

python -m torch.distributed.launch --nproc_per_node=4 \
train_vit_pt_timm_gpu.py --mode=eager --micro_batch_size=4
"""
import argparse
import time
import os
import logging
import statistics
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import einops

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint,\
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.models.helpers import build_model_with_cfg
from timm.models.vision_transformer import VisionTransformer


torch.backends.cudnn.benchmark = True


# Hyperparams

VERBOSE = False
num_attention_heads = 16
hidden_size = 1280
num_layers = 32

image_size = 224
patch_size = 16  # Size of the patches to be extract from the input images

num_classes = 1000
num_epochs = 10


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Batch size
parser.add_argument("--micro_batch_size", default=32, type=int)

# Misc
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--mode', type=str,
                    help='"eager" or "graph"')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

def print_if_verbose(msg):
    if VERBOSE:
        print(msg, flush=True)

class VitDummyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_size, crop_size, num_classes):
        self.dataset_size = dataset_size
        self.crop_size = crop_size
        self.num_classes = num_classes

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return (torch.rand(3, self.crop_size, self.crop_size).to(torch.half), torch.randint(self.num_classes, (1,)).to(torch.long))


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
        ret = self.projection(rearranged_input)
        ret = ret + self.position_embedding(positions)
        return ret

step_duration_list = []

def main():
    args = parser.parse_args()

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print_if_verbose('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        print_if_verbose('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    random_seed(42, args.rank)

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

    if args.local_rank == 0:
        print_if_verbose(
            f'Model created, param count:{sum([m.numel() for m in model.parameters()])}')

    # move model to GPU, enable channels last layout if set
    model = model.to(torch.half)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    assert args.mode in ["eager", "graph"]
    if args.mode == "graph":
        model = torch.jit.script(torch.fx.symbolic_trace(model))

    model = model.cuda()

    optimizer = create_optimizer_v2(model, 'adam', lr=1e-6)

    # setup distributed training
    if args.distributed:
        model = NativeDDP(model, device_ids=[args.local_rank])

    start_epoch = 0

    if args.local_rank == 0:
        print_if_verbose('Scheduled epochs: {}'.format(num_epochs))

    # create train dataset
    dataset_train = VitDummyDataset(args.micro_batch_size * torch.distributed.get_world_size() * 10, image_size, num_classes)

    loader_train = create_loader(
        dataset_train,
        input_size=(3, 224, 224),
        batch_size=args.micro_batch_size * torch.distributed.get_world_size(),
        is_training=True,
        no_aug=True,
        fp16=True,
    )

    # setup loss function
    train_loss_fn = nn.CrossEntropyLoss().to(torch.half)
    train_loss_fn = train_loss_fn.cuda()

    try:
        for epoch in range(start_epoch, num_epochs):
            if should_profile:
                prof = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                )
                prof.__enter__()

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args)

            if should_profile:
                prof.__exit__(None, None, None)
                trace_dir_path = "train_vit_pt_timm_gpu_trace"
                if not os.path.isdir(trace_dir_path):
                    os.mkdir(trace_dir_path)
                prof.export_chrome_trace(os.path.join(trace_dir_path, "trace_{}_{}.json".format(str(int(time.time())), str(torch.distributed.get_rank()))))
            should_profile = False  # NOTE: only profile one epoch

        if args.local_rank == 0:
            print("micro_batch_size: {}, median step duration: {:.3f}".format(args.micro_batch_size, statistics.median(step_duration_list)))
    except KeyboardInterrupt:
        pass


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args):

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        output = model(input)
        loss = loss_fn(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward(create_graph=second_order)
        optimizer.step()

        torch.cuda.synchronize()
        num_updates += 1
        batch_time = time.time() - end
        batch_time_m.update(batch_time)
        step_duration_list.append(batch_time)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                print_if_verbose(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

        end = time.time()
        # end for

    return OrderedDict([('loss', losses_m.avg)])


if __name__ == '__main__':
    main()
