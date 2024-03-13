import PIL

try:
    import os
    import torch
    import cv2
    from osgeo import gdal_array
    import numpy as np
    from torchvision.models import vgg16_bn
    import torch.nn.functional as F
    import torch.nn as nn
    from fastai.vision.learner import create_body
    from fastai.vision.models.unet import DynamicUnet
    from fastai.callbacks import hook_outputs
    from fastai.torch_core import requires_grad, children, apply_init
    from fastprogress.fastprogress import progress_bar
    from .._utils.superres import psnr, ssim
    from .._utils.common import ArcGISMSImage

    HAS_FASTAI = True
except Exception as e:
    # import_exception = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))

    HAS_FASTAI = False


def resize_to_new(img, targ_sz: int, use_min: bool = False):
    "Size to resize to, to hit `targ_sz` at same aspect ratio, in PIL coords (i.e w*h)"
    w, h, _ = img.shape
    min_sz = (min if use_min else max)(w, h)
    ratio = targ_sz / min_sz
    return int(w * ratio), int(h * ratio)


def resize_one(fn, i, path_lr, size, path_hr, img_size):
    dest = path_lr / fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = ArcGISMSImage.read_image(fn)
    targ_sz = resize_to_new(img, size, use_min=True)
    img = cv2.resize(img, targ_sz, interpolation=cv2.INTER_LINEAR)
    dest = str(dest.with_suffix(".tif"))
    if os.path.isfile(dest):
        os.remove(dest)
    ds = gdal_array.SaveArray(
        np.transpose(img, (2, 0, 1)), os.path.join(os.path.split(dest)[0], "test.tif")
    )
    ds = None
    os.rename(os.path.join(os.path.split(dest)[0], "test.tif"), dest)


def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2)) / (c * h * w)


def UNetSR(data, arch, norm_type):
    from ._arcgis_model import _change_tail

    body = create_body(arch, pretrained=True)
    new_body = _change_tail(body, data)
    model = DynamicUnet(
        new_body,
        n_classes=data._n_channel,
        img_size=data.train_ds[0][0].size,
        blur=True,
        self_attention=True,
        norm_type=norm_type,
    )
    apply_init(model[2], nn.init.kaiming_normal_)
    return model


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts, base_loss=F.l1_loss):
        super().__init__()
        self.m_feat = m_feat
        self.base_loss = base_loss
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = [
            "pixel_loss",
        ]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [self.base_loss(input, target)]
        self.feat_losses += [
            self.base_loss(f_in, f_out) * w
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]
        self.feat_losses += [
            self.base_loss(gram_matrix(f_in), gram_matrix(f_out)) * w**2 * 5e3
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self):
        self.hooks.remove()


def create_loss(c, device_type="cuda"):
    if device_type == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    vggmodel = vgg16_bn(True)
    vggmodel.features[0] = nn.Conv2d(c, 64, 3, 1, 1)
    vgg_m = vggmodel.features.to(device).eval()
    requires_grad(vgg_m, False)
    blocks = [
        i - 1 for i, o in enumerate(children(vgg_m)) if isinstance(o, nn.MaxPool2d)
    ]
    blocks, [vgg_m[i] for i in blocks]
    feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5, 15, 2])
    return feat_loss


def compute_metrics(model, dl, show_progress, **kwargs):
    sampling = kwargs.get("sampling_type", "ddim")
    ntimestep = kwargs.get("n_timestep")

    avg_psnr = 0
    avg_ssim = 0
    model.learn.model.eval()
    with torch.no_grad():
        for input, target in progress_bar(dl, display=False):
            if model.model_type == "UNet":
                prediction = model.learn.model(input)
            else:
                device = next(model.learn.model.parameters()).device.type

                if sampling == "ddim":
                    n_timestep = ntimestep if ntimestep else 200
                    nstp = {"n_timestep": model.kwargs.get("n_timestep", 1000)}
                else:
                    n_timestep = (
                        ntimestep if ntimestep else model.kwargs.get("n_timestep", 1000)
                    )
                    nstp = {"n_timestep": n_timestep}
                combkwargs = {**kwargs, **model.kwargs, **nstp}
                model.learn.model.set_new_noise_schedule(device, **combkwargs)

                from tqdm import tqdm

                preds = []
                for k in tqdm(
                    range(input[0].shape[0]), desc="sampling loop time step per image"
                ):
                    if sampling == "ddim":
                        preds.append(
                            model.learn.model.super_resolution(
                                input[0][k, None],
                                continous=False,
                                sampling_timesteps=n_timestep,
                                ddim_sampling_eta=1,
                                sampling="ddim",
                            )
                        )
                    else:
                        preds.append(
                            model.learn.model.super_resolution(
                                input[0][k, None], continous=False
                            )
                        )
                prediction = torch.cat(preds)

                avg_psnr += psnr(prediction, target)
                avg_ssim += ssim(prediction, target)
                break
            avg_psnr += psnr(prediction, target)
            avg_ssim += ssim(prediction, target)
    if model.model_type == "UNet":
        return avg_psnr / len(dl), avg_ssim.item() / len(dl)
    else:
        return avg_psnr, avg_ssim


def get_resize(y, z, max_size, f):
    if y * f <= max_size and z * f <= max_size:
        y_new = y * f
        z_new = z * f
    else:
        if y > z:
            y_new = max_size
            z_new = int(round_up_to_even(z * max_size / y))
        else:
            z_new = max_size
            y_new = int(round_up_to_even(y * max_size / z))
    return (y_new, z_new)
