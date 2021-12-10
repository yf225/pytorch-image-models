import torch

from timm.models.helpers import build_model_with_cfg
from timm.models.vision_transformer import VisionTransformer

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


def create_vit_model(
    image_size,
    patch_size,
    num_classes,
    hidden_size,
    num_layers,
    num_attention_heads,
):
    model = VisionTransformer(
        img_size=image_size, patch_size=patch_size, in_chans=3, num_classes=num_classes, embed_dim=hidden_size, depth=num_layers,
        num_heads=num_attention_heads, mlp_ratio=4, qkv_bias=True, representation_size=None, distilled=False,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEncoder, norm_layer=torch.nn.LayerNorm,
        act_layer=torch.nn.GELU, weight_init=''
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

    return model
