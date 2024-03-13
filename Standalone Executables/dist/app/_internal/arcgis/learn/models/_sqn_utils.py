import torch
import torch.nn as nn
from ._rand_lanet_utils import Conv1d, Dilated_res_block, batch_preprocess_dict


class SQNRandLANet(nn.Module):
    def __init__(self, config, in_chanls):
        super().__init__()
        self.config = config

        self.fc0 = Conv1d(in_chanls, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config["num_layers"]):
            d_out = self.config["out_channels"][i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = 2 * sum(self.config["out_channels"])

        self.fc1 = Conv1d(d_out, 256, kernel_size=1, bn=True)
        self.fc2 = Conv1d(256, 128, kernel_size=1, bn=True)
        self.fc3 = Conv1d(128, 64, kernel_size=1, bn=True)
        self.dropout = nn.Dropout(0.5)
        self.logits = Conv1d(
            64,
            self.config["num_classes"],
            kernel_size=1,
            bn=False,
            activation=None,
        )

    def forward(self, end_points):
        # transform input for infrencing
        if not isinstance(end_points, dict):
            device = end_points.device
            end_points = batch_preprocess_dict(
                end_points.cpu(), self.config, is_sqn=True
            )
            for key in end_points:
                if type(end_points[key]) is list:
                    for i in range(len(end_points[key])):
                        end_points[key][i] = end_points[key][i].to(device)
                else:
                    end_points[key] = end_points[key].to(device)

        features = end_points["features"]  # Batch*channel*npoints
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################

        f_encoder_list = []
        for i in range(self.config["num_layers"]):
            f_encoder_i = self.dilated_res_blocks[i](
                features, end_points["xyz"][i], end_points["neigh_idx"][i]
            )

            f_sampled_i = self.random_sample(f_encoder_i, end_points["sub_idx"][i])
            features = f_sampled_i
            f_encoder_list.append(f_sampled_i)

        # ###########################Query Network######################

        query_feature_list = []
        xyz_query = end_points["xyz"][0]  # original points
        for i in range(self.config["num_layers"]):
            xyz_current = end_points["xyz"][i + 1]
            features_current = f_encoder_list[i]
            f_query_feature_i = self.trilinear_interpolation(
                xyz_query, xyz_current, features_current
            )
            query_feature_list.append(f_query_feature_i)

        features_combined = torch.cat(
            query_feature_list, dim=1
        )  # (Batch , 928, npoints)

        features = self.fc1(features_combined)
        features = self.fc2(features)
        features = self.fc3(features)
        features = self.dropout(features)
        f_out = self.logits(features)

        return f_out.permute(0, 2, 1)

    @staticmethod
    def trilinear_interpolation(xyz_query, xyz_support, features_support):
        # Method based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
        """
        xyz_query : Tensor
            (B, N, 3) tensor of the xyz positions of the unknown points
        xyz_support : Tensor
            (B, M, 3) tensor of the xyz positions of the known points (i.e. B PC examples, each is mx3 shape)
        features_support : Tensor
            (B, C, M, 1) tensor of features to be propagated
        Returns
        new_features : torch.Tensor
            (B, C, N) upsampled tensor
        """
        features_support = features_support.squeeze(dim=3).permute(
            0, 2, 1
        )  # batch*npoints*channel(B, M, C)

        B, N, C = xyz_query.shape
        _, M, _ = xyz_support.shape

        dists = square_distance(xyz_query, xyz_support)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_features = torch.sum(
            index_points(features_support, idx) * weight.view(B, N, 3, 1), dim=2
        )

        interpolated_features = interpolated_features.permute(
            0, 2, 1
        )  # batch*channel*npoints
        return interpolated_features

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(
            feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        )
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[
            0
        ]  # batch*channel*npoints*1
        return pool_features


def square_distance(src, dst):
    # Method based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    # Method based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points
