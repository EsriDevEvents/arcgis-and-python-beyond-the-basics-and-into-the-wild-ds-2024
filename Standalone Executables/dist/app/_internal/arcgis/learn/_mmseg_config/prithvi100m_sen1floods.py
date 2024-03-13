# model settings
custom_imports = dict(imports=["arcgis.learn.models._prithvi_archs"])

bands = [1, 2, 3, 8, 11, 12]
# required bands Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2

CLASSES = ("flooded", "non-flooded")

img_norm_flood_model = dict(
    means=[0.14245495, 0.13921481, 0.12434631, 0.31420089, 0.20743526, 0.12046503],
    stds=[0.04036231, 0.04186983, 0.05267646, 0.0822221, 0.06834774, 0.05294205],
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
        embed_dim=768,
        output_embed_dim=768,
        drop_cls_token=True,
        Hp=14,
        Wp=14,
    ),
    decode_head=dict(
        num_classes=2,
        in_channels=768,
        ignore_index=2,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1,
            class_weight=[0.3, 0.7, 0],
        ),
    ),
    auxiliary_head=dict(
        num_classes=2,
        in_channels=768 * 1,
        ignore_index=2,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1,
            class_weight=[0.3, 0.7, 0],
        ),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

checkpoint = "https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-sen1floods11/resolve/main/sen1floods11_Prithvi_100M.pth"
