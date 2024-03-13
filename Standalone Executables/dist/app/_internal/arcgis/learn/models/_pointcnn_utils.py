import random
import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F
from fastai.basic_train import LearnerCallback

# For AverageMetric callback.
from fastai.callback import Callback
from fastai.core import first_el, is_listy
from fastai.torch_core import add_metrics, num_distrib
import torch.distributed as dist
import sklearn.metrics as metrics


def farthest_point_sample(pts, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    from torch_geometric.nn import fps

    device = pts.device
    B, N, C = pts.shape

    batch = (torch.arange(0, B * N) // N).to(pts.device).contiguous()
    pts = pts.view(-1, C).float().contiguous()
    indices = fps(pts, batch, ratio=(npoint / N))
    if (indices.shape[0] / B) > npoint:
        each_block_point = int(indices.shape[0] / B)
        pts = pts[indices].view(B, each_block_point, C)
        drop_point_num = each_block_point - npoint
        pts = pts[:, :-drop_point_num, :]
    else:
        pts = pts[indices].view(B, npoint, C)
    return pts.contiguous()


def find_k_neighbor(rep_pts, pts, K, D):
    """
    Find K nearest neighbor points
    :param pts: represent points(B, P, C)
    :param rep_pts: original points (B, N, C)
    :param K:
    :param D: Dilation rate
    :return group_pts: K neighbor points(B, P, K, C)
    """
    from torch_geometric.nn import knn

    device = pts.device
    B, N, C = pts.shape
    _, N_rep, _ = rep_pts.shape
    batch_pts = (torch.arange(0, B * N) // N).to(pts.device).contiguous()
    batch_rep_pts = (torch.arange(0, B * N_rep) // N_rep).to(pts.device).contiguous()
    pts = pts.view(-1, C).contiguous()
    rep_pts = rep_pts.view(-1, C).contiguous()
    knn_indices = knn(pts, rep_pts, K * D, batch_pts, batch_rep_pts)
    knn_indices = knn_indices[
        1
    ]  ## grab the indices at 0th index it is [0,0,0,0,0,1,1,1,1,1,2,2,2,2,...]
    group_pts = pts[knn_indices].view(B, N_rep, K * D, C).contiguous()
    rand_col = torch.randint(K * D, (K,))
    group_pts = group_pts[:, :, rand_col, :]

    return group_pts.contiguous(), knn_indices.contiguous(), rand_col.contiguous()


class DepthwiseConv2D(nn.Module):
    def __init__(
        self,
        in_channels,
        depth_multiplier,
        kernel_size,
        final_reshape=False,
        point_wise=None,
    ):
        """
        in_channels: input channels of point cloud,
        depth_multiplier: in nn.Conv2d out_channels is set to depth_multiplier * in_channels and groups=in_channels
        final_reshape: returns after reshaping and permuting the tensor
        point_wise: accepts a integer which is the out channel of the XConv
        """
        super(DepthwiseConv2D, self).__init__()
        self.depth_multiplier = depth_multiplier
        self.final_reshape = final_reshape
        self.point_wise = bool(point_wise)
        if point_wise:
            self.module = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=depth_multiplier * in_channels,
                    kernel_size=kernel_size,
                    groups=in_channels,
                    bias=False,
                ),
                nn.Conv2d(
                    in_channels=depth_multiplier * in_channels,
                    out_channels=point_wise,
                    kernel_size=1,
                    bias=False,
                ),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(point_wise),
            )
        else:
            self.module = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=depth_multiplier * in_channels,
                    kernel_size=kernel_size,
                    groups=in_channels,
                    bias=False,
                ),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(depth_multiplier * in_channels),
            )

    def forward(self, inp):
        if self.point_wise:
            out = self.module(inp.contiguous())
            if self.final_reshape:
                return out.squeeze(3).permute(0, 2, 1).contiguous()
            else:
                return out
        else:
            B, C, P, K = inp.shape
            C = int(C**0.5)
            inp = inp.view(B, P, C, C).permute(0, 3, 1, 2).contiguous()
            out = self.module(inp.contiguous())
            return (
                out.view(B, P, self.depth_multiplier, C)
                .permute(0, 3, 1, 2)
                .contiguous()
                if self.final_reshape
                else out
            )


class XConvDepthwise(nn.Module):
    def __init__(
        self,
        in_channel,
        lift_channel,
        out_channel,
        P,
        K,
        D=1,
        sampling="fps",
        with_global=False,
    ):
        """
        :param in_channel: Input channel of the points' features
        :param lift_channel: Lifted channel C_delta
        :param out_channel:
        :param P: P represent points
        :param K: K neighbors to operate
        :param D: Dilation rate
        """
        super(XConvDepthwise, self).__init__()
        self.P = P
        self.K = K
        self.D = D
        self.sampling = sampling
        self.with_global = with_global

        # Input should be (B, 3, P, K)
        self.MLP_delta = nn.Sequential(
            nn.Conv2d(3, lift_channel, kernel_size=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(lift_channel),
            nn.Conv2d(lift_channel, lift_channel, kernel_size=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(lift_channel),
        )

        # Input should be (B, 3, P, K)  --> (B, K, P, K)
        self.MLP_X = nn.Sequential(
            nn.Conv2d(3, K * K, kernel_size=(1, K)),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(K * K),
            DepthwiseConv2D(in_channels=K, depth_multiplier=K, kernel_size=(1, K)),
            DepthwiseConv2D(
                in_channels=K,
                depth_multiplier=K,
                kernel_size=(1, K),
                final_reshape=True,
            ),
        )

        depth_multiplier = math.ceil(out_channel / (in_channel + lift_channel))
        self.seperable_conv = DepthwiseConv2D(
            in_channel + lift_channel,
            depth_multiplier=depth_multiplier,
            kernel_size=(1, K),
            point_wise=out_channel,
            final_reshape=True,
        )

        if with_global:
            lift_channel = out_channel // 4
            self.MLP_g = nn.Sequential(
                nn.Conv2d(3, lift_channel, kernel_size=1),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(lift_channel),
                nn.Conv2d(lift_channel, lift_channel, kernel_size=1),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(lift_channel),
            )

    def forward(self, pts, fts, represent_pts=None):
        """
        :param x: (rep_pt, pts, fts) where
          - pts: Regional point cloud (B, N, 3)
          - fts: Regional features (B, N, C)
        :return: Features aggregated into point rep_pt.
        """
        B, N, _ = pts.size()

        if represent_pts is None:
            if (self.P == -1) or (self.P == N):
                self.P = N
                represent_pts = pts
            else:
                represent_pts = farthest_point_sample(pts, self.P)  # (B, P, 3)

        group_pts, k_ind, rand_col = find_k_neighbor(
            represent_pts, pts, self.K, self.D
        )  # (B, P, K, 3), (B, P, K)
        center_pts = torch.unsqueeze(represent_pts, dim=2)  # (B, P, 1, 3)
        group_pts = group_pts - center_pts  # (B, P, K, 3)

        # fts_lifted
        group_pts = group_pts.permute(0, 3, 1, 2).contiguous()
        fts_lifted = self.MLP_delta(group_pts.contiguous())  # (B, C_delta, P, K)

        if fts is not None:
            _, _, nf = fts.shape
            group_fts = fts.contiguous().view(-1, nf)
            group_fts = group_fts[k_ind].view(B, self.P, self.K * self.D, nf)
            group_fts = group_fts[:, :, rand_col, :]
            group_fts = group_fts.permute(0, 3, 1, 2).contiguous()
            feat = torch.cat(
                (fts_lifted, group_fts), 1
            ).contiguous()  # (B, C_delta + C_in, P, K)
        else:
            feat = fts_lifted.contiguous()

        # XConv operation
        X = self.MLP_X(group_pts).permute(0, 2, 3, 1)  # (B, P, K, K)
        X = X.contiguous().view(B * self.P, self.K, self.K)

        feat = feat.permute(0, 2, 3, 1).contiguous().view(B * self.P, self.K, -1)
        feat = (
            torch.bmm(X, feat).view(B, self.P, self.K, -1).permute(0, 3, 1, 2)
        )  # (B, C_delta + C_in, P, K)

        feat = self.seperable_conv(feat.contiguous())  # (B, self.P, C_out)

        if self.with_global:
            feat_global = (
                self.MLP_g(
                    represent_pts.unsqueeze(dim=2).permute(0, 3, 1, 2).contiguous()
                )
                .squeeze(3)
                .permute(0, 2, 1)
                .contiguous()
            )
            return (
                represent_pts.contiguous(),
                torch.cat([feat, feat_global], dim=-1).contiguous(),
            )  # (B, P, 3), (B, P, C_out + C_out / 4)
        else:
            return (
                represent_pts.contiguous(),
                feat.contiguous(),
            )  # (B, P, 3), (B, P, C_out)


def get_indices(batch_size, sample_num, point_num, pool_setting=None):
    if not isinstance(point_num, np.ndarray):
        point_nums = np.full((batch_size), point_num.cpu())
    else:
        point_nums = point_num

    indices = []
    for i in range(batch_size):
        pt_num = point_nums[i]
        if pool_setting is None:
            pool_size = pt_num
        else:
            if isinstance(pool_setting, int):
                pool_size = min(pool_setting, pt_num)
            elif isinstance(pool_setting, tuple):
                pool_size = min(
                    random.randrange(pool_setting[0], pool_setting[1] + 1), pt_num
                )
        if pool_size > sample_num:
            choices = np.random.choice(pool_size, sample_num, replace=False)
        else:
            choices = np.concatenate(
                (
                    np.random.choice(pool_size, pool_size, replace=False),
                    np.random.choice(pool_size, sample_num - pool_size, replace=True),
                )
            )
        if pool_size < pt_num:
            choices_pool = np.random.choice(pt_num, pool_size, replace=False)
            choices = choices_pool[choices]
        choices = np.expand_dims(choices, axis=1)
        choices_2d = np.concatenate((np.full_like(choices, i), choices), axis=1)
        indices.append(choices_2d)
    return np.stack(indices)


class SamplePointsCallback(LearnerCallback):
    def __init__(self, learn, sample_point_num):
        super().__init__(learn)
        self.sample_point_num = sample_point_num

    def on_batch_begin(self, last_input, last_target, **kwargs):
        """
        Random Sample points from input and output.
        """
        (
            last_input,
            point_nums,
        ) = last_input  ## (batch, total_points, num_features), (batch,)
        batch, _, num_features = last_input.shape

        ## get indices to sample (tf implementation function)
        indices = torch.tensor(
            get_indices(batch, self.sample_point_num, point_nums.long())
        ).to(last_input.device)

        ## Get indices in the correct shape to be used for indexing
        indices = indices.view(-1, 2).long()

        ##  sample points from all the input and output points
        last_input = (
            last_input[indices[:, 0], indices[:, 1]]
            .view(batch, self.sample_point_num, num_features)
            .contiguous()
        )  ## batch, self.sample_point_num, num_features
        last_target = (
            last_target[indices[:, 0], indices[:, 1]]
            .view(batch, self.sample_point_num)
            .contiguous()
        )  ## batch, self.sample_point_num

        del indices

        if self.learn.data.transform_fn is not None and self.learn.model.training:
            if random.random() > 0.5:
                if self.learn.data.pc_type == "PointCloud_TF":
                    last_input[:, :, :3] = self.learn.data.transform_fn(last_input)
                if self.learn.data.pc_type == "PointCloud":
                    last_input[:, :, :3] = self.learn.data.transform_fn._transform_tool(
                        last_input
                    )

        return {
            "last_input": last_input.contiguous(),
            "last_target": last_target.contiguous(),
        }


class PointCNNSeg(nn.Module):
    def __init__(
        self,
        sampled_num_points,
        num_classes,
        num_extra_features,
        encoder_params=None,
        dropout=None,
    ):
        """
        sampled_num_points: The number that goes into the model.
        num_classes: number of classes
        num_extra_features: number of features other than XYZ
        encoder_params: Length of out_channels, P, K, D should be same. The length denotes the number of layers in encoder.
        {
                'out_channels': Number of channels in each layer multiplied by m,
                 'P': Number of points in each layer,
                 'K': Number of K-nearest neighbour in each layer,
                 'D': Dilation in each layer,
                 'm': Multiplier which is multiplied by each out_channel.
        }
        """
        super().__init__()

        if encoder_params is None:
            encoder_params = {
                "out_channels": [16, 32, 64, 96],
                "P": [-1, 768, 384, 128],
                "K": [12, 16, 16, 16],
                "D": [1, 1, 2, 2],
                "m": 8,
            }

        if dropout is None:
            dropout = 0.5

        self.num_extra_features = num_extra_features
        self.num_classes = num_classes
        self.encoder_params = encoder_params

        ## m is the multiplier value as on case of config file of yangli/pointcnn
        ## This value is multiplied by the out_channels value
        out_channels, P, K, D, m = list(encoder_params.values())

        ############## Encoder ################
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(P)):
            self.encoder_layers.append(
                ## in_channels is equal to num_extra_features for the first layer.
                XConvDepthwise(
                    in_channel=num_extra_features
                    if i == 0
                    else out_channels[i - 1] * m,
                    lift_channel=out_channels[i] * m // 4,
                    out_channel=out_channels[i] * m,
                    P=P[i],
                    K=K[i],
                    D=D[i],
                    with_global=True if i == (len(P) - 1) else False,
                )
            )

        ############## Decoder ################
        self.decoder_layers = nn.ModuleList([])
        self.densecat_layers = nn.ModuleList([])
        for j in reversed(range(len(P) - 1)):
            if P[j] == -1:
                P[j] = sampled_num_points

            self.decoder_layers.append(  ## append decoder layers
                ## Since in the
                XConvDepthwise(
                    in_channel=out_channels[j + 1] * m + (out_channels[j + 1] * m) // 4
                    if (j + 1) == (len(P) - 1)
                    else out_channels[j + 1] * m,
                    lift_channel=out_channels[j] * m // 4,
                    out_channel=out_channels[j] * m,
                    P=P[j],
                    K=K[j],
                    D=D[j],
                )
            )

            self.densecat_layers.append(  ## append concat layers
                nn.Sequential(
                    nn.Conv1d(out_channels[j] * m * 2, out_channels[j] * m, 1),
                    nn.ELU(inplace=True),
                    nn.BatchNorm1d(out_channels[j] * m),
                )
            )

        self.fc_layers = nn.Sequential(
            nn.Conv1d(out_channels[j] * m, 256, 1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Conv1d(256, 256, 1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Conv1d(256, num_classes, 1),
        )

    def forward(self, pts):
        """
        pts: points tensor of the following shape(Batch, sampled_num_points, 3 + num_extra_features)
        """

        if pts.shape[2] > 3:
            pts, fts = (
                pts[:, :, : -self.num_extra_features],
                pts[:, :, -self.num_extra_features :],
            )
        else:
            fts = None

        rep_pts_fts = []

        for i, module in enumerate(self.encoder_layers):
            rep_pts_fts.append(module(*((pts, fts) if i == 0 else rep_pts_fts[-1])))

        for i, module in enumerate(self.decoder_layers):
            if i == 0:
                # If i==0, we do not perform densecat operation.
                _, fts_final = module(
                    *rep_pts_fts[-(i + 1)], represent_pts=rep_pts_fts[-(i + 2)][0]
                )
            else:
                dc_module = self.densecat_layers[i - 1]
                ## Here we are performing the densecat operation with the earlier features of the pointcnn model.
                fts_final = (
                    dc_module(
                        torch.cat((fts_final, rep_pts_fts[-(i + 1)][1]), dim=-1)
                        .permute(0, 2, 1)
                        .contiguous()
                    )
                    .permute(0, 2, 1)
                    .contiguous()
                )
                ## The concatenated features from are then passed throught the XConv module.
                _, fts_final = module(
                    rep_pts_fts[-(i + 1)][0],
                    fts_final,
                    represent_pts=rep_pts_fts[-(i + 2)][0],
                )

        ## The final concatenation happens here.
        fts_final = self.densecat_layers[-1](
            torch.cat((fts_final, rep_pts_fts[0][1]), dim=-1).permute(0, 2, 1)
        )

        ## Model is then passed through the fully connected layers to get the logits.
        return self.fc_layers(fts_final).permute(0, 2, 1)


class CrossEntropyPC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, inp, target):
        inp = inp.contiguous()
        target = target.contiguous()
        inp = inp.view(-1, self.num_classes).contiguous()
        target = target.view(-1).contiguous()
        return F.cross_entropy(inp, target).contiguous()


def accuracy(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()
    pred = pred.argmax(dim=-1)
    return (pred == target).float().mean()


def accuracy_non_zero(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()
    pred = pred.argmax(dim=-1)
    mask = target != 0
    accuracy_value = (pred[mask] == target[mask]).float().mean()
    return accuracy_value


## Redefines AverageMetric Callback so that it handle nan values and does not compute average on those.
class AverageMetric(Callback):
    "Wrap a `func` in a callback for metrics computation."

    def __init__(self, func):
        # If func has a __name__ use this one else it should be a partial
        name = func.__name__ if hasattr(func, "__name__") else func.func.__name__
        self.func, self.name = func, name
        self.world = num_distrib()

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = 0.0, 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        if not is_listy(last_target):
            last_target = [last_target]
        val = self.func(last_output, *last_target)
        ## If nan do not increase counter and return
        if torch.isnan(val).tolist():
            return
        self.count += first_el(last_target).size(0)
        if self.world:
            val = val.clone()
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            val /= self.world
        self.val += first_el(last_target).size(0) * val.detach().cpu()

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        if self.count == 0:
            return add_metrics(last_metrics, [None])
        return add_metrics(last_metrics, self.val / self.count)


## Iteration Stop Callback, i.e stops epoch after certain number of iterations.
class IterationStop(LearnerCallback):
    def __init__(self, learn, stop_iteration):
        super().__init__(learn)
        self.stop_iteration = stop_iteration

    def on_batch_end(self, **kwargs):
        if (kwargs["iteration"] + 1) % self.stop_iteration == 0:
            return {"stop_epoch": True}


def precision(y_pred, y_true):
    y_true = y_true.cpu().numpy().reshape(-1)
    y_pred = y_pred.argmax(dim=-1).cpu().numpy().reshape(-1)
    return torch.tensor(
        metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
    )


def recall(y_pred, y_true):
    y_true = y_true.cpu().numpy().reshape(-1)
    y_pred = y_pred.argmax(dim=-1).cpu().numpy().reshape(-1)
    return torch.tensor(
        metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)
    )


def f1(y_pred, y_true):
    y_true = y_true.cpu().numpy().reshape(-1)
    y_pred = y_pred.argmax(dim=-1).cpu().numpy().reshape(-1)
    return torch.tensor(
        metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    )


def balanced_accuracy(y_pred, y_true):
    y_true = y_true.cpu().numpy().reshape(-1)
    y_pred = y_pred.argmax(dim=-1).cpu().numpy().reshape(-1)
    return torch.tensor(metrics.balanced_accuracy_score(y_true, y_pred))


class CalculateClassificationReport(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)
        self.learn._epoch_metrics = []

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = {}, 0

    def add_all(self, new_val, size):
        for k, v in new_val.items():
            self.val.setdefault(k, {})
            if k not in ["accuracy", "macro avg", "weighted avg"]:
                self.val[k].setdefault("precision", 0.0)
                self.val[k]["precision"] += v["precision"]
                self.val[k].setdefault("recall", 0.0)
                self.val[k]["recall"] += v["recall"]
                self.val[k].setdefault("f1-score", 0.0)
                self.val[k]["f1-score"] += v["f1-score"]
                self.val[k].setdefault("count", 0)
                self.val[k]["count"] += 1
                # self.val[k]['support'] += v['support']

    def average_all(self):
        final_val = {}
        for k, v in self.val.items():
            if k not in ["accuracy", "macro avg", "weighted avg"]:
                final_val[k] = {}
                final_val[k]["precision"] = v["precision"] / v["count"]
                final_val[k]["recall"] = v["recall"] / v["count"]
                final_val[k]["f1-score"] = v["f1-score"] / v["count"]
                # final_val[k]['support'] = v['support'] / self.count
        return final_val

    def rename_metrics(self, current_epoch_metrics):
        idx2class = self.learn.data.idx2class
        class_mapping = self.learn.data.class_mapping
        renamed = {}
        for k, v in current_epoch_metrics.items():
            cname = class_mapping.get(idx2class.get(int(k), int(k)), str(k))
            renamed[cname] = {}
            renamed[cname]["precision"] = v["precision"]
            renamed[cname]["recall"] = v["recall"]
            renamed[cname]["f1-score"] = v["f1-score"]
            # renamed[cname]['support'] = v['support']
        return renamed

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        if not kwargs.get("train"):
            if is_listy(last_target):
                last_target = last_target[0]
            size = last_target.size(0)
            self.count += size
            last_output = last_output.argmax(-1).cpu().numpy().reshape(-1)
            last_target = last_target.cpu().numpy().reshape(-1)
            val = metrics.classification_report(
                last_output, last_target, zero_division=0, output_dict=True
            )
            self.add_all(val, size)

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        # Lr find case
        if self.val is {}:
            return
        current_epoch_metrics = self.average_all()
        current_epoch_metrics = self.rename_metrics(current_epoch_metrics)
        self.learn._epoch_metrics.append(current_epoch_metrics)
