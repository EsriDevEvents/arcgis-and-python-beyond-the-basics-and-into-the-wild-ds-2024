import torch, os
import torch.hub
from mmcv.utils import load_url
from collections import OrderedDict

chk = load_url(
    "https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/resolve/main/Prithvi_100M.pt"
)
hub_dir = torch.hub.get_dir()
modchkp = OrderedDict(
    (i, j) for i, j in chk.items() if not (i.startswith("decoder") or i == "mask_token")
)
torch.save(modchkp, os.path.join(hub_dir, "checkpoints", "Prithvi_100M_Encoder.pth"))

# model settings
custom_imports = dict(imports=["arcgis.learn.models._prithvi_archs"])

bands = [0, 1, 2, 3, 4, 5]
nframes = 3

norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="TemporalEncoderDecoder",
    frozen_backbone=True,
    backbone=dict(
        type="TemporalViTEncoder",
        pretrained=os.path.join(hub_dir, "checkpoints", "Prithvi_100M_Encoder.pth"),
        img_size=224,
        patch_size=16,
        num_frames=nframes,
        tubelet_size=1,
        in_chans=len(bands),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_pix_loss=False,
    ),
    neck=dict(
        type="ConvTransformerTokensToEmbeddingNeck",
        embed_dim=768 * nframes,
        output_embed_dim=768 * nframes,
        drop_cls_token=True,
        Hp=14,
        Wp=14,
    ),
    decode_head=dict(
        in_channels=768 * nframes,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
    ),
    auxiliary_head=dict(
        in_channels=768 * nframes,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
