voxel_size = [0.05, 0.05, 0.1]

model = dict(
    type="VoxelNet",
    voxel_layer=dict(
        max_num_points=5,
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        voxel_size=voxel_size,
        max_voxels=(16000, 40000),
    ),
    voxel_encoder=dict(type="HardSimpleVFE"),
    middle_encoder=dict(
        type="SparseEncoder",
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=("conv", "norm", "act"),
    ),
    backbone=dict(
        type="SECOND",
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256],
    ),
    neck=dict(
        type="SECONDFPN",
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256],
    ),
    bbox_head=dict(
        type="Anchor3DHead",
        num_classes=1,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type="Anchor3DRangeGenerator",
            ranges=[[0, -40.0, -1.78, 70.4, 40.0, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True,
        ),
        diff_rad_by_sin=True,
        bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder"),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2),
    ),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            iou_calculator=dict(type="BboxOverlapsNearest3D"),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1,
        ),
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50,
    ),
)

checkpoint = "https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_6x8_80e_kitti-3d-car/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth"
