# Code from https://github.com/fastai/course-v3/blob/master/nbs/dl2/cyclegan_ws.ipynb, https://github.com/eriklindernoren/PyTorch-GAN & https://github.com/haofengac/MonoDepth-FPN-PyTorch

import torch
import random
import numpy as np
from fastai.vision import (
    nn,
    Callable,
    List,
    LearnerCallback,
    optim,
    ifnone,
    F,
    flatten_model,
    requires_grad,
    SmoothenValue,
    add_metrics,
)
from .._utils.cyclegan import (
    calculate_activation_statistics,
    calculate_frechet_distance,
)
from fastprogress.fastprogress import progress_bar
from .._utils.superres import psnr, ssim
from skimage.feature import match_template

import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#          Hybrid W-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, kernel=4, stride=2, padding=1):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, kernel, stride, padding, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, *skip_input):
        x = self.model(x)
        if skip_input:
            x = torch.cat((x, skip_input[0], skip_input[1]), 1)
            return x
        else:
            return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.enc_down1 = UNetDown(in_channels, 64, normalize=False)
        self.enc_down2 = UNetDown(64, 128)
        self.enc_down3 = UNetDown(128, 256)
        self.enc_down4 = UNetDown(256, 512, dropout=0.5)
        self.enc_down5 = UNetDown(512, 512, dropout=0.5)
        self.enc_down6 = UNetDown(512, 512, dropout=0.5)
        self.enc_down7 = UNetDown(512, 512, dropout=0.5)
        self.enc_down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(1024, 1024, dropout=0.5, kernel=1, stride=1, padding=0)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1536, 512, dropout=0.5)
        self.up4 = UNetUp(1536, 512, dropout=0.5)
        self.up5 = UNetUp(1536, 512, dropout=0.5)
        self.up6 = UNetUp(1536, 256)
        self.up7 = UNetUp(768, 128)
        self.up8 = UNetUp(384, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(192, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x1, x2):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.enc_down1(x1)
        d2 = self.enc_down2(d1)
        d3 = self.enc_down3(d2)
        d4 = self.enc_down4(d3)
        d5 = self.enc_down5(d4)
        d6 = self.enc_down6(d5)
        d7 = self.enc_down7(d6)
        d8 = self.enc_down8(d7)

        e1 = self.enc_down1(x2)
        e2 = self.enc_down2(e1)
        e3 = self.enc_down3(e2)
        e4 = self.enc_down4(e3)
        e5 = self.enc_down5(e4)
        e6 = self.enc_down6(e5)
        e7 = self.enc_down7(e6)
        e8 = self.enc_down8(e7)

        conc = torch.cat((d8, e8), 1)

        u1 = self.up1(conc)
        u2 = self.up2(u1, d7, e7)
        u3 = self.up3(u2, d6, e6)
        u4 = self.up4(u3, d5, e5)
        u5 = self.up5(u4, d4, e4)
        u6 = self.up6(u5, d3, e3)
        u7 = self.up7(u6, d2, e2)
        u8 = self.up8(u7, d1, e1)

        return self.final(u8)


##############################
#        Discriminator
##############################


class discriminator_block(nn.Module):
    def __init__(self, in_filters, out_filters, normalization=True):
        super(discriminator_block, self).__init__()
        """Returns downsampling layers of each discriminator block"""
        layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
        if normalization:
            layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.disc_blk_1 = discriminator_block(in_channels * 2, 64, normalization=False)
        self.disc_blk_2 = discriminator_block(64, 128)
        self.disc_blk_3 = discriminator_block(128, 256)
        self.disc_blk_4 = discriminator_block(256, 512)
        self.final = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        disc_1 = self.disc_blk_1(img_input)
        disc_2 = self.disc_blk_2(disc_1)
        disc_3 = self.disc_blk_3(disc_2)
        disc_4 = self.disc_blk_4(disc_3)

        return self.final(disc_4)


class WNet_cGAN(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        n_features: int = 64,
        disc_layers: int = 3,
        gen_blocks: int = 8,
        lsgan: bool = False,
        drop: float = 0.0,
        norm_layer: nn.Module = None,
    ):
        super().__init__()

        self.D = Discriminator(ch_in)
        self.G = GeneratorUNet(ch_in, ch_out)
        self.arcgis_results = False

    def forward(self, real_A, real_B, real_C):
        if real_A.shape[0] == 1 or real_B.shape[0] == 1 or real_C.shape[0] == 1:
            real_A, real_B, real_C = (
                torch.tensor([real_A.cpu().numpy()[0, :, :, :]] * 2).cuda(),
                torch.tensor([real_B.cpu().numpy()[0, :, :, :]] * 2).cuda(),
                torch.tensor([real_C.cpu().numpy()[0, :, :, :]] * 2).cuda(),
            )
        fake_C = self.G(real_A, real_B)
        if real_A.shape[0] == 1 or real_B.shape[0] == 1:
            fake_C = fake_C[None, 0, :, :, :]
        if self.training:
            self.arcgis_results = False
        if self.arcgis_results:
            return torch.cat([fake_C[:, None], fake_C[:, None]], 1)
        return [fake_C]


class AdaptiveLoss(nn.Module):
    def __init__(self, crit):
        super().__init__()
        self.crit = crit

    def forward(self, output, target: bool, **kwargs):
        targ = (
            output.new_ones(*output.size())
            if target
            else output.new_zeros(*output.size())
        )
        return self.crit(output, targ, **kwargs)


class Adaptivel2Loss(nn.Module):
    def __init__(self, crit1):
        super().__init__()
        self.crit1 = crit1

    def forward(self, output, target: bool, **kwargs):
        targ = (
            output.new_ones(*output.size())
            if target
            else output.new_zeros(*output.size())
        )
        return self.crit1(output, targ, **kwargs)


class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()

    def forward(self, fake, real):
        eps = 1e-7
        real = F.relu(real)
        fake = F.relu(fake)
        loss = torch.sqrt(
            torch.mean(torch.abs(torch.log(real + eps) - torch.log(fake + eps)) ** 2)
        )
        return loss


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def imgrad(self, img):
        img = torch.mean(img, 1, True)
        fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv1.weight = nn.Parameter(weight)
        grad_x = conv1(img)

        fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv2.weight = nn.Parameter(weight)
        grad_y = conv2(img)
        return grad_y, grad_x

    def imgrad_yx(self, img):
        N, C, _, _ = img.size()
        grad_y, grad_x = self.imgrad(img)
        return torch.cat((grad_y.view(N, C, -1), grad_x.view(N, C, -1)), dim=1)

    def forward(self, grad_fake, grad_real):
        c, d = (
            self.imgrad_yx((grad_fake[:, 0, :, :])[:, None, :, :]),
            self.imgrad_yx((grad_real[:, 0, :, :])[:, None, :, :]),
        )

        e = (c[:, :, None, :] @ d[:, :, :, None]).squeeze(-1).squeeze(-1)

        fake_norm = torch.sqrt(torch.sum(c**2, dim=-1))
        real_norm = torch.sqrt(torch.sum(d**2, dim=-1))

        return 1 - torch.mean(e / (fake_norm * real_norm))


class get_activation_from_layers:
    def __init__(self, name):
        self.name = name

    def get_activation(self):
        activation = {}

        def hook(model, input, output):
            activation[self.name] = output.detach()

        return hook, activation


class WNetcGANLoss(nn.Module):
    def __init__(
        self,
        cgan: nn.Module,
        lambda_A: float = 100.0,
        lambda_B: float = 10,
        lambda_idt: float = 0.5,
        gamma_g: float = 10.0,
        lsgan: bool = False,
    ):
        super().__init__()
        self.cgan, self.l_A, self.l_B, self.l_idt, self.gam_g = (
            cgan,
            lambda_A,
            lambda_B,
            lambda_idt,
            gamma_g,
        )
        self.crit = AdaptiveLoss(
            F.mse_loss if lsgan else F.binary_cross_entropy_with_logits
        )
        self.crit1 = Adaptivel2Loss(F.mse_loss)
        self.norm_loss = NormalLoss()
        self.rmse_log = RMSE_log()

    def set_input(self, input):
        self.real_A, self.real_B, self.real_C = input
        if (
            self.real_A.shape[0] == 1
            or self.real_B.shape[0] == 1
            or self.real_C.shape[0] == 1
        ):
            self.real_A, self.real_B, self.real_C = (
                torch.tensor([self.real_A.cpu().numpy()[0, :, :, :]] * 2).cuda(),
                torch.tensor([self.real_B.cpu().numpy()[0, :, :, :]] * 2).cuda(),
                torch.tensor([self.real_C.cpu().numpy()[0, :, :, :]] * 2).cuda(),
            )

    def forward(self, output, target):
        # fake_C = (output)[:,0,:,:,:]
        fake_C = (output)[0]
        self.gen_loss = self.crit(self.cgan.D(self.real_A, fake_C), True)
        self.l1_loss = F.l1_loss(self.real_C, fake_C)
        self.rmse_log_loss = self.rmse_log(self.real_C, fake_C)
        self.norm_vec_loss = self.norm_loss(fake_C, self.real_C)
        return (
            self.gen_loss
            + (self.l_A * self.l1_loss)
            + (self.l_B * self.norm_vec_loss)
            + self.rmse_log_loss
        )


class WNetcGANTrainer(LearnerCallback):
    _order = -20  # Need to run before the Recorder

    def _set_trainable(self, D=False):
        gen = not D
        # requires_grad(self.learn.model.G, gen)
        # requires_grad(self.learn.model.D, D)
        if not gen:
            self.opt_D.lr, self.opt_D.mom = self.learn.opt.lr, self.learn.opt.mom
            self.opt_D.wd, self.opt_D.beta = self.learn.opt.wd, self.learn.opt.beta

    def on_train_begin(self, **kwargs):
        self.G = self.learn.model.G
        self.D = self.learn.model.D
        self.crit = self.learn.loss_func.crit
        self.crit1 = self.learn.loss_func.crit1

        if not getattr(self, "opt_G", None):
            self.opt_G = self.learn.opt.new([nn.Sequential(*flatten_model(self.G))])
        else:
            self.opt_G.lr, self.opt_G.wd = self.opt.lr, self.opt.wd
            self.opt_G.mom, self.opt_G.beta = self.opt.mom, self.opt.beta

        if not getattr(self, "opt_D", None):
            self.opt_D = self.learn.opt.new([nn.Sequential(*flatten_model(self.D))])

        self.learn.opt.opt = self.opt_G.opt
        self._set_trainable()
        self.gen_smter, self.l1_smter, self.norm_vec_smter = (
            SmoothenValue(0.98),
            SmoothenValue(0.98),
            SmoothenValue(0.98),
        )
        self.d_smter = SmoothenValue(0.98)
        self.recorder.add_metric_names(
            ["gen_loss", "l1_loss", "norm_vec_loss", "D_loss"]
        )

    def on_batch_begin(self, last_input, **kwargs):
        self.learn.loss_func.set_input(last_input)

    def on_backward_begin(self, **kwargs):
        self.l1_smter.add_value(self.loss_func.l1_loss.detach().cpu())
        self.gen_smter.add_value(self.loss_func.gen_loss.detach().cpu())
        self.norm_vec_smter.add_value(self.loss_func.norm_vec_loss)

    def on_batch_end(self, last_input, last_output, **kwargs):
        self.G.zero_grad()
        fake_C = last_output[0].detach()
        real_A, real_B, real_C = last_input
        if real_A.shape[0] == 1 or real_B.shape[0] == 1 or real_C.shape[0] == 1:
            real_A, real_B, real_C = (
                torch.tensor([real_A.cpu().numpy()[0, :, :, :]] * 2).cuda(),
                torch.tensor([real_B.cpu().numpy()[0, :, :, :]] * 2).cuda(),
                torch.tensor([real_C.cpu().numpy()[0, :, :, :]] * 2).cuda(),
            )
        self._set_trainable(D=True)

        self.D.zero_grad()

        if random.choice([0, 1]) < 0.5:
            loss_D = self.crit(self.D(real_A, real_C), True)
        else:
            loss_D = self.crit(self.D(real_A, fake_C), False)

        self.d_smter.add_value(loss_D.detach().cpu())
        if self.learn.model.training == True:
            loss_D.backward()

        self.opt_D.step()

        self._set_trainable()

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(
            last_metrics,
            [
                s.smooth
                for s in [
                    self.gen_smter,
                    self.l1_smter,
                    self.norm_vec_smter,
                    self.d_smter,
                ]
            ],
        )


def compute_metrics(model, dl, show_progress):
    avg_psnr = 0
    avg_ssim = 0
    avg_ncc = 0
    model.learn.model.eval()
    with torch.no_grad():
        for input, target in progress_bar(dl, display=False):
            if (
                input[0].shape[0] != 1
                or input[1].shape[0] != 1
                or input[2].shape[0] != 1
            ):
                prediction = model.learn.model(input[0], input[1], input[2])
                if isinstance(prediction, list):
                    prediction = prediction[0]
                else:
                    prediction = prediction[:, 0, :, :, :]
                avg_ncc += match_template(
                    prediction.cpu().numpy()[:, :, 0], input[2].cpu().numpy()[:, :, 0]
                )
                avg_psnr += psnr(prediction, input[2])
                avg_ssim += ssim(prediction, input[2])
    return avg_psnr / len(dl), avg_ssim.item() / len(dl), float(avg_ncc / len(dl))
