# export PYTHONPATH=/mnt/home/willfeng/repos/pytorch-image-models:${PYTHONPATH}

from timm.models.helpers import build_model_with_cfg
from timm.models.vision_transformer import VisionTransformer

num_attention_heads = 16
hidden_size = 1280
num_layers = 32

image_size = 224
patch_size = 16  # Size of the patches to be extract from the input images

model = build_model_with_cfg(
    VisionTransformer,
    "vit_huge_patch{}_{}".format(patch_size, image_size),
    pretrained=False,
    default_cfg={},
    representation_size=None,
    **dict(
        patch_size=patch_size, embed_dim=hidden_size, depth=num_layers, num_heads=num_attention_heads
    )
)

print(model)
