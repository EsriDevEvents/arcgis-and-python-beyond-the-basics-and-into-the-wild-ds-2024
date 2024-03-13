import numpy as np
from math import exp, log10, ceil
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from fastai.vision import ImageImageList, Tuple, subplots, plt, random
from .common import ArcGISMSImage, ArcGISImageListRGB, ArcGISImageList
from .._utils.common import get_nbatches, get_top_padding


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def psnr(img1, img2):
    mse = F.l1_loss(img1, img2)
    psnr = 10 * log10(1 / mse)
    return psnr


class ArcGISImageListSR_MS(ArcGISImageList):
    _div = None


def standardize(x, means, stds):
    return (x - np.array(means)[:, None, None]) / np.array(stds)[:, None, None]


class ImageImageListSR(ImageImageList):
    label_cls = ArcGISImageListSR_MS

    @classmethod
    def from_folders(cls, path, folder, image_stats, is_multispec=False, **kwargs):
        global _image_stats, is_ms
        res = super().from_folder(folder, **kwargs)
        res.path = path
        _image_stats, is_ms = image_stats, is_multispec
        return res

    def open(self, fn):
        if is_ms:
            return ArcGISMSImage(
                standardize(
                    ArcGISMSImage.open(fn, div=None).data,
                    _image_stats[0],
                    _image_stats[1],
                ).type(torch.float32)
            )
        else:
            return ArcGISMSImage.open(fn, div=255)

    def label_from_func(self, func, label_cls=None, **kwargs):
        if is_ms:
            label_cls = ArcGISImageListSR_MS
        else:
            label_cls = ArcGISImageListRGB
        return super().label_from_func(func, label_cls=label_cls, **kwargs)


def show_batch(self, rows=4, **kwargs):
    """
    This function randomly picks a few training chips and visualizes them.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    rows                    Optional int. Number of rows of results
                            to be displayed.
    =====================   ===========================================
    """
    xs, ys = [], []
    for n, imgs in enumerate(self.train_ds):
        if n != rows:
            xs.append(imgs[0])
            ys.append(imgs[1])
        else:
            break

    axs = subplots(len(xs), 2, figsize=(20, rows * 5))
    for i, (x, y) in enumerate(zip(xs, ys)):
        x, y = ArcGISMSImage(x.data), ArcGISMSImage(y.data)
        x.show(ax=axs[i, 0], **kwargs)
        y.show(ax=axs[i, 1], **kwargs)
    axs[0, 0].title.set_text("Low Resolution")
    axs[0, 1].title.set_text("High Resolution")


def show_results(self, rows, **kwargs):
    from .._data_utils.pix2pix_data import display_row, denormalize

    sampling = kwargs.get("sampling_type", "ddim")
    ntimestep = kwargs.get("n_timestep")
    device = next(self.learn.model.parameters()).device.type

    self.learn.model.eval()
    activ = []
    top = get_top_padding(title_font_size=16, nrows=rows, imsize=5)
    denormfunc = lambda img, max, min: (img + 1) * (max - min) / 2 + min

    if self.model_type == "UNet":
        x_batch, y_batch = get_nbatches(
            self._data.valid_dl, ceil(rows / self._data.batch_size)
        )
        if isinstance(x_batch[0], list):
            x_batch = x_batch[0]

        x_A, x_B = torch.cat(x_batch), torch.cat(y_batch)

        for i in range(0, x_A.shape[0], self._data.batch_size):
            preds = self.learn.model(x_A[i : i + self._data.batch_size].detach())
            activ.append(preds)
        activations = torch.cat(activ)

        x_A = denormalize(x_A.cpu(), *self._data._image_stats)
        x_B = denormalize(x_B.cpu(), *self._data._image_stats)
        activations = denormalize(activations.cpu(), *self._data._image_stats)
        rows = min(rows, x_A.shape[0])
    else:
        x_A = torch.cat([i[0] for i, _ in self.learn.data.valid_dl])[:rows]
        x_B = torch.cat([j for _, j in self.learn.data.valid_dl])[:rows]

        x_A_batch = x_A.detach()

        if sampling == "ddim":
            n_timestep = ntimestep if ntimestep else 200
            nstp = {"n_timestep": self.kwargs.get("n_timestep", 1000)}
        else:
            n_timestep = ntimestep if ntimestep else self.kwargs.get("n_timestep", 1000)
            nstp = {"n_timestep": n_timestep}
        combkwargs = {**kwargs, **self.kwargs, **nstp}

        self.learn.model.set_new_noise_schedule(device, **combkwargs)

        preds = []
        for k in tqdm(
            range(x_A_batch.shape[0]), desc="sampling loop time step per image"
        ):
            if sampling == "ddim":
                preds.append(
                    self.learn.model.super_resolution(
                        x_A_batch[k, None],
                        continous=False,
                        sampling_timesteps=n_timestep,
                        ddim_sampling_eta=1,
                        sampling="ddim",
                    )
                )
            else:
                preds.append(
                    self.learn.model.super_resolution(
                        x_A_batch[k, None], continous=False
                    )
                )
        activations = torch.cat(preds)

        maxvals_a = self._data.batch_stats_a["band_max_values"][..., None, None]
        minvals_a = self._data.batch_stats_a["band_min_values"][..., None, None]
        maxvals_b = self._data.batch_stats_b["band_max_values"][..., None, None]
        minvals_b = self._data.batch_stats_b["band_min_values"][..., None, None]
        x_A = denormfunc(x_A.cpu(), maxvals_a, minvals_a)
        x_B = denormfunc(x_B.cpu(), maxvals_b, minvals_b)
        activations = denormfunc(activations.cpu(), maxvals_b, minvals_b)

    fig, axs = plt.subplots(
        nrows=rows, ncols=3, figsize=(4 * 5, rows * 5), squeeze=False
    )
    plt.subplots_adjust(top=top)
    axs[0, 0].title.set_text("Input")
    axs[0, 1].title.set_text("Target")
    axs[0, 2].title.set_text("Prediction")
    for r in range(rows):
        display_row(
            axs[r],
            (
                ArcGISMSImage(x_A[r].cpu()),
                ArcGISMSImage(x_B[r].cpu()),
                ArcGISMSImage(activations[r].detach().cpu()),
            ),
            kwargs.get("rgb_bands", None),
        )
