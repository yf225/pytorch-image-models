#!/usr/bin/env python3

"""On AWS GPU node
conda activate torch-1.10

cd /fsx/users/willfeng/repos
rm -rf ./pytorch-image-models || true
git clone https://github.com/yf225/pytorch-image-models.git -b vit_dummy_data
cd pytorch-image-models && git pull

export PYTHONPATH=/fsx/users/willfeng/repos/pytorch-image-models:${PYTHONPATH}

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
train_vit_pt_timm_gpu.py --mode=eager --micro_batch_size=8

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
train_vit_pt_timm_gpu.py --mode=graph --micro_batch_size=2

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
train_vit_pt_timm_gpu.py --mode=eager --micro_batch_size=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
train_vit_pt_timm_gpu.py --mode=eager --micro_batch_size=8

rsync -avr ab101835-ddb5-466f-9d25-55b1d5a16351:/fsx/users/willfeng/repos/pytorch-image-models/train_vit_pt_timm_gpu_trace/* ~/train_vit_pt_timm_gpu_trace/

manifold put ./trace_1638305255_0.json gpu_traces_manual/tree/AWS_V100_traces/trace_1638305255_0.json
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

should_profile = False
VERBOSE = False
num_attention_heads = 16
hidden_size = 1280
num_layers = 32

image_size = 224
patch_size = 16  # Size of the patches to be extract from the input images

num_classes = 1000
num_epochs = 3


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Batch size
parser.add_argument("--micro_batch_size", default=32, type=int)

# Misc
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--mode', type=str,
                    help='"eager" or "graph"')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

def print_if_verbose(msg):
    if VERBOSE:
        print(msg, flush=True)

class VitDummyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_size, num_classes):
        self.dataset_size = dataset_size
        self.num_classes = num_classes

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return (torch.rand(3, image_size, image_size).to(torch.half), torch.randint(self.num_classes, (1,)).to(torch.long))


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

def create_dataloader(dataset):
  loader_train = create_loader(
    dataset,
    input_size=(3, 224, 224),
    batch_size=args.micro_batch_size,  # NOTE: this should be batch size per GPU, re. https://discuss.pytorch.org/t/72769/2
    is_training=True,
    no_aug=True,
    fp16=True,
    distributed=args.distributed,
  )

def main():
    global should_profile

    args = parser.parse_args()

    args.distributed = False
    args.num_devices = 1
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
        args.num_devices = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print_if_verbose('Training in distributed mode with multiple processes, 1 GPU per process. global rank: {}, local rank: {}, total {}'.format(
            args.rank, args.local_rank, args.world_size))
    else:
        print_if_verbose('Training with a single process on 1 GPUs.')
        torch.cuda.set_device(args.local_rank)
    assert args.rank >= 0

    random_seed(42, args.rank)

    model = VisionTransformer(
        img_size=image_size, patch_size=patch_size, in_chans=3, num_classes=num_classes, embed_dim=hidden_size, depth=num_layers,
        num_heads=num_attention_heads, mlp_ratio=4, qkv_bias=True, representation_size=None, distilled=False,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEncoder, norm_layer=nn.LayerNorm,
        act_layer=nn.GELU, weight_init=''
    )

    # model = build_model_with_cfg(
    #     VisionTransformer,
    #     "vit_huge_patch{}_{}".format(patch_size, image_size),
    #     pretrained=False,
    #     default_cfg={},
    #     representation_size=None,  # NOTE: matching vit_tf_tpu_v2.py impl
    #     **dict(
    #         img_size=image_size, patch_size=patch_size, embed_dim=hidden_size, depth=num_layers, num_heads=num_attention_heads, num_classes=num_classes,
    #         embed_layer=PatchEncoder,
    #     )
    # )

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
        model = NativeDDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    start_epoch = 0

    if args.local_rank == 0:
        print_if_verbose('Scheduled epochs: {}'.format(num_epochs))

    # create train dataset
    dataset_train = VitDummyDataset(args.micro_batch_size * args.num_devices * 10, num_classes)
    loader_train = create_dataloader(dataset_train)
    sample_batch = next(iter(create_dataloader(dataset_train)))
    print("sample_batch[0].shape: ", sample_batch[0].shape)
    assert list(sample_batch[0].shape) == [args.micro_batch_size, 3, image_size, image_size]

    # setup loss function
    train_loss_fn = nn.CrossEntropyLoss().to(torch.half)
    train_loss_fn = train_loss_fn.cuda()

    try:
        from fvcore.nn import FlopCountAnalysis
        from fvcore.nn import flop_count_table
        flops = FlopCountAnalysis(model, sample_batch[0])
        if args.local_rank == 0:
            print(flop_count_table(flops))

        for epoch in range(start_epoch, num_epochs):
            if should_profile and args.local_rank == 0:
                def recorder_enter_hook(module, input):
                    module._torch_profiler_recorder = torch.autograd.profiler.record_function(str(module.__class__))
                    module._torch_profiler_recorder.__enter__()

                def recorder_exit_hook(module, input, output):
                    module._torch_profiler_recorder.__exit__(None, None, None)

                torch.nn.modules.module.register_module_forward_pre_hook(recorder_enter_hook)
                torch.nn.modules.module.register_module_forward_hook(recorder_exit_hook)

                prof = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                )
                prof.__enter__()

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args)

            if should_profile and args.local_rank == 0:
                prof.__exit__(None, None, None)
                trace_dir_path = "train_vit_pt_timm_gpu_trace"
                if not os.path.isdir(trace_dir_path):
                    os.mkdir(trace_dir_path)
                prof.export_chrome_trace(os.path.join(trace_dir_path, "trace_{}_{}_{}.json".format(str(int(time.time())), args.num_devices, args.local_rank)))
                should_profile = False  # NOTE: only profile one epoch

        if args.local_rank == 0:
            print("micro_batch_size: {}, median step duration: {:.3f}".format(args.micro_batch_size, statistics.median(step_duration_list)))
    except KeyboardInterrupt:
        pass


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args):

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with torch.autograd.profiler.record_function("### forward ###"):
            output = model(input)
            loss = loss_fn(output, target)

        with torch.autograd.profiler.record_function("### zero_grad ###"):
            optimizer.zero_grad()

        with torch.autograd.profiler.record_function("### backward ###"):
            loss.backward(create_graph=second_order)

        with torch.autograd.profiler.record_function("### optimizer step ###"):
            optimizer.step()

        torch.cuda.synchronize()
        num_updates += 1
        batch_time = time.time() - end
        batch_time_m.update(batch_time)
        step_duration_list.append(batch_time)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.local_rank == 0:
                print_if_verbose(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

        end = time.time()
        # end for

    return OrderedDict([('loss', -1)])


if __name__ == '__main__':
    main()
