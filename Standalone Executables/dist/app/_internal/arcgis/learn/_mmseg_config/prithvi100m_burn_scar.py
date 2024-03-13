# model settings
custom_imports = dict(imports=["arcgis.learn.models._prithvi_archs"])

# required bands Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2
bands = [0, 1, 2, 3, 4, 5]

CLASSES = ("Unburnt land", "Burn scar")

img_norm_burn_model = dict(
    means=[
        0.033349706741586264,
        0.05701185520536176,
        0.05889748132001316,
        0.2323245113436119,
        0.1972854853760658,
        0.11944914225186566,
    ],
    stds=[
        0.02269135568823774,
        0.026807560223070237,
        0.04004109844362779,
        0.07791732423672691,
        0.08708738838140137,
        0.07241979477437814,
    ],
)

norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="TemporalEncoderDecoder",
    pretrained=None,
    frozen_backbone=True,
    backbone=dict(
        type="TemporalViTEncoder",
        pretrained=None,
        img_size=224,
        patch_size=16,
        num_frames=1,
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
        embed_dim=768 * 1,
        output_embed_dim=768 * 1,
        drop_cls_token=True,
        Hp=14,
        Wp=14,
    ),
    decode_head=dict(
        in_channels=768 * 1,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type="DiceLoss", use_sigmoid=False, loss_weight=1, ignore_index=-1
        ),
    ),
    auxiliary_head=dict(
        in_channels=768 * 1,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type="DiceLoss", use_sigmoid=False, loss_weight=1, ignore_index=-1
        ),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

checkpoint = "https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-burn-scar/resolve/main/burn_scars_Prithvi_100M.pth"
