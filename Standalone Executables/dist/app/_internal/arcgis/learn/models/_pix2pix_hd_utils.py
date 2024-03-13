# Copyright (C) 2019 NVIDIA Corporation. Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu.
# BSD License. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
# DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
# OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


# --------------------------- LICENSE FOR pytorch-CycleGAN-and-pix2pix ----------------
# Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Based on https://github.com/NVIDIA/pix2pixHD

from torch.autograd import Variable
from fastai.vision import (
    LearnerCallback,
    functools,
    np,
    flatten_model,
    requires_grad,
    SmoothenValue,
    add_metrics,
    F,
)
import torch
from torch import nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def define_gen(
    input_nc,
    output_nc,
    ngf,
    netG,
    n_downsample_global=3,
    n_blocks_global=9,
    n_local_enhancers=1,
    n_blocks_local=3,
    norm="instance",
    gpu_ids=[],
):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == "global":
        netG = GlobalGenerator(
            input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer
        )
    elif netG == "local":
        netG = LocalEnhancer(
            input_nc,
            output_nc,
            ngf,
            n_downsample_global,
            n_blocks_global,
            n_local_enhancers,
            n_blocks_local,
            norm_layer,
        )
    elif netG == "encoder":
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise ("generator not implemented!")
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_dscr(
    input_nc,
    ndf,
    n_layers_D,
    norm="instance",
    use_sigmoid=False,
    num_D=1,
    getIntermFeat=False,
    gpu_ids=[],
):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(
        input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat
    )
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


class GANLoss(nn.Module):
    def __init__(
        self,
        use_lsgan=True,
        target_real_label=1.0,
        target_fake_label=0.0,
        tensor=torch.FloatTensor,
    ):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = (self.real_label_var is None) or (
                self.real_label_var.numel() != input.numel()
            )
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = (self.fake_label_var is None) or (
                self.fake_label_var.numel() != input.numel()
            )
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


##############################################################################
# Generator
##############################################################################


class LocalEnhancer(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=32,
        n_downsample_global=3,
        n_blocks_global=9,
        n_local_enhancers=1,
        n_blocks_local=3,
        norm_layer=nn.BatchNorm2d,
        padding_type="reflect",
    ):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(
            input_nc,
            output_nc,
            ngf_global,
            n_downsample_global,
            n_blocks_global,
            norm_layer,
        ).model
        model_global = [
            model_global[i] for i in range(len(model_global) - 3)
        ]  # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                norm_layer(ngf_global),
                nn.ReLU(True),
                nn.Conv2d(
                    ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1
                ),
                norm_layer(ngf_global * 2),
                nn.ReLU(True),
            ]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [
                    ResnetBlock(
                        ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer
                    )
                ]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(
                    ngf_global * 2,
                    ngf_global,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                norm_layer(ngf_global),
                nn.ReLU(True),
            ]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                    nn.Tanh(),
                ]

            setattr(self, "model" + str(n) + "_1", nn.Sequential(*model_downsample))
            setattr(self, "model" + str(n) + "_2", nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False
        )

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, "model" + str(n_local_enhancers) + "_1")
            model_upsample = getattr(self, "model" + str(n_local_enhancers) + "_2")
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev


class GlobalGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        n_downsampling=3,
        n_blocks=9,
        norm_layer=nn.BatchNorm2d,
        padding_type="reflect",
    ):
        assert n_blocks >= 0
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            activation,
        ]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1
                ),
                norm_layer(ngf * mult * 2),
                activation,
            ]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                )
            ]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                norm_layer(int(ngf * mult / 2)),
                activation,
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(
        self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False
    ):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, activation, use_dropout
        )

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim),
            activation,
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Encoder(nn.Module):
    def __init__(
        self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d
    ):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b : b + 1] == int(i)).nonzero()  # n x 4
                for j in range(self.output_nc):
                    output_ins = outputs[
                        indices[:, 0] + b,
                        indices[:, 1] + j,
                        indices[:, 2],
                        indices[:, 3],
                    ]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[
                        indices[:, 0] + b,
                        indices[:, 1] + j,
                        indices[:, 2],
                        indices[:, 3],
                    ] = mean_feat
        return outputs_mean


class MultiscaleDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid=False,
        num_D=3,
        getIntermFeat=False,
    ):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat
            )
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(
                        self,
                        "scale" + str(i) + "_layer" + str(j),
                        getattr(netD, "model" + str(j)),
                    )
            else:
                setattr(self, "layer" + str(i), netD.model)

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False
        )

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [
                    getattr(self, "scale" + str(num_D - 1 - i) + "_layer" + str(j))
                    for j in range(self.n_layers + 2)
                ]
            else:
                model = getattr(self, "layer" + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid=False,
        getIntermFeat=False,
    ):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                [
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(nf),
                    nn.LeakyReLU(0.2, True),
                ]
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, "model" + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, "model" + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


def encode_input(
    label_map,
    label_nc,
    data_type=32,
    inst_map=None,
    real_image=None,
    feat_map=None,
    infer=False,
):
    if label_nc == 0:
        input_label = label_map.data.cuda()
    else:
        # create one-hot vector for label map
        size = label_map.size()
        oneHot_size = (size[0], label_nc, size[2], size[3])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
        if data_type == 16:
            input_label = input_label.half()

    #             get edges from instance map
    #             if not opt.no_instance:
    #                 inst_map = inst_map.data.cuda()
    #                 edge_map = self.get_edges(inst_map)
    #                 input_label = torch.cat((input_label, edge_map), dim=1)
    input_label = Variable(input_label, volatile=infer)

    # real images for training
    if real_image is not None:
        real_image = Variable(real_image.data.cuda())

    # instance map for feature encoding
    #             if self.use_features:
    #                 # get precomputed feature maps
    #                 if self.opt.load_features:
    #                     feat_map = Variable(feat_map.data.cuda())
    #                 if self.opt.label_feat:
    #                     inst_map = label_map.cuda()

    return input_label, inst_map, real_image, feat_map


class Pix2PixHDModel(nn.Module):
    def __init__(self, label_nc, input_nc, output_nc, gpu_ids, **kwargs):
        super().__init__()
        self.gpu_ids = gpu_ids
        ngf = kwargs.get("n_gen_filters", 64)
        n_downsample_global = kwargs.get("n_downsample_global", 4)
        n_blocks_global = kwargs.get("n_blocks_global", 9)
        n_local_enhancers = kwargs.get("n_local_enhancers", 1)
        n_blocks_local = kwargs.get("n_blocks_local", 3)
        self.lsgan = kwargs.get("lsgan", True)

        ndf = kwargs.get("n_dscr_filters", 64)
        self.n_layers_D = kwargs.get("n_layers_dscr", 3)
        self.num_D = kwargs.get("n_dscr", 2)
        self.feat_loss = kwargs.get("feat_loss", True)
        netG = kwargs.get("gen_network", "local")
        norm = kwargs.get("norm", "instance")
        self.label_nc = label_nc

        self.data_type = 32  # hardcoded
        if self.label_nc:
            input_nc = label_nc

        netG_input_nc = input_nc
        self.G = define_gen(
            netG_input_nc,
            output_nc,
            ngf,
            netG,
            n_downsample_global,
            n_blocks_global,
            n_local_enhancers,
            n_blocks_local,
            norm,
            gpu_ids=self.gpu_ids,
        )

        use_sigmoid = not self.lsgan
        netD_input_nc = input_nc + output_nc
        self.D = define_dscr(
            netD_input_nc,
            ndf,
            self.n_layers_D,
            norm,
            use_sigmoid,
            self.num_D,
            self.feat_loss,
            gpu_ids=self.gpu_ids,
        )
        self.arcgis_results = False

    def set_input(self, input):
        self.input_label, self.real_image = input

    def forward(self, label, image):
        if self.label_nc:
            if self.training:
                label = self.input_label
            else:
                label, _, _, _ = encode_input(label_map=label, label_nc=self.label_nc)

        fake_image = self.G(label)

        return [fake_image]


class Pix2PixHDLoss(nn.Module):
    def __init__(
        self,
        p2p_model: nn.Module,
        vgg_loss=True,
        lambda_feat=10.0,
        l1_loss=True,
        lambda_l1=100.0,
        gpu_ids=[],
    ):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.p2p_model = p2p_model
        self.criterionGAN = GANLoss(use_lsgan=self.p2p_model.lsgan, tensor=self.Tensor)
        self.criterionFeat = torch.nn.L1Loss()
        self.lambda_feat = lambda_feat
        self.vgg_loss = vgg_loss
        if self.vgg_loss:
            self.criterionVGG = VGGLoss(self.gpu_ids)
        self.l1_loss = l1_loss
        self.lambda_l1 = lambda_l1

    def set_input(self, input):
        self.input_label, self.real_image = input

    def discriminate(self, input_label, test_image):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        return self.p2p_model.D.forward(input_concat)

    def forward(self, output, target):
        fake_image = output[0]

        # Real Detection and Loss

        pred_real = self.discriminate(self.input_label, self.real_image)  #####

        # GAN loss (Fake Passability Loss)
        pred_fake = self.p2p_model.D.forward(
            torch.cat((self.input_label, fake_image), dim=1)
        )

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        self.loss_G_GAN_Feat = 0
        if self.p2p_model.feat_loss:
            feat_weights = 4.0 / (self.p2p_model.n_layers_D + 1)
            D_weights = 1.0 / self.p2p_model.num_D
            for i in range(self.p2p_model.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    self.loss_G_GAN_Feat += (
                        D_weights
                        * feat_weights
                        * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                        * self.lambda_feat
                    )
        # VGG feature matching loss
        self.loss_G_VGG = 0
        if self.vgg_loss:
            self.loss_G_VGG = (
                self.criterionVGG(fake_image, self.real_image) * self.lambda_feat
            )

        # l1 loss for MS data
        self.loss_G_l1 = 0
        if self.l1_loss:
            self.loss_G_l1 = (
                torch.mean(F.l1_loss(fake_image, self.real_image))
            ) * self.lambda_l1

        return self.loss_G_GAN + self.loss_G_GAN_Feat + self.loss_G_VGG + self.loss_G_l1


class Pix2PixHDTrainer(LearnerCallback):
    _order = -20

    def _set_trainable(self, D=False):
        gen = not D
        # requires_grad(self.learn.model.G, gen)
        # requires_grad(self.learn.model.D, D)
        if not gen:
            self.opt_D.lr, self.opt_D.mom = self.learn.opt.lr, self.learn.opt.mom
            self.opt_D.wd, self.opt_D.beta = self.learn.opt.wd, self.learn.opt.beta

    def discriminate(self, input_label, test_image):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        return self.D.forward(input_concat)

    def on_train_begin(self, **kwargs):
        self.G = self.learn.model.G
        self.D = self.learn.model.D
        self.criterionGAN = self.learn.loss_func.criterionGAN

        if not getattr(self, "opt_G", None):
            self.opt_G = self.learn.opt.new([nn.Sequential(*flatten_model(self.G))])
        else:
            self.opt_G.lr, self.opt_G.wd = self.opt.lr, self.opt.wd
            self.opt_G.mom, self.opt_G.beta = self.opt.mom, self.opt.beta

        if not getattr(self, "opt_D", None):
            self.opt_D = self.learn.opt.new([nn.Sequential(*flatten_model(self.D))])

        self.learn.opt.opt = self.opt_G.opt

        self._set_trainable()

        self.G_GAN_smter, self.G_GAN_Feat_smter, self.G_VGG_smter, self.G_l1_smter = (
            SmoothenValue(0.98),
            SmoothenValue(0.98),
            SmoothenValue(0.98),
            SmoothenValue(0.98),
        )
        self.D_fake_smter, self.D_real_smter = SmoothenValue(0.98), SmoothenValue(0.98)

        self.data_type = 32
        self.recorder.add_metric_names(["gen_loss", "disc_fake_loss", "disc_real_loss"])

        if self.model.feat_loss:
            self.recorder.add_metric_names(["feat_loss"])

        if self.loss_func.vgg_loss:
            self.recorder.add_metric_names(["VGG_loss"])

        if self.loss_func.l1_loss:
            self.recorder.add_metric_names(["l1_loss"])

    def on_batch_begin(self, last_input, **kwargs):
        last_input[0], _, last_input[1], _ = encode_input(
            last_input[0], label_nc=self.model.label_nc, real_image=last_input[1]
        )
        self.learn.model.set_input(last_input)
        self.learn.loss_func.set_input(last_input)

    def on_backward_begin(self, **kwargs):
        self.G_GAN_smter.add_value(self.loss_func.loss_G_GAN.detach().cpu())
        if self.model.feat_loss:
            self.G_GAN_Feat_smter.add_value(
                self.loss_func.loss_G_GAN_Feat.detach().cpu()
            )
        if self.loss_func.vgg_loss:
            self.G_VGG_smter.add_value(self.loss_func.loss_G_VGG.detach().cpu())

        if self.loss_func.l1_loss:
            self.G_l1_smter.add_value(self.loss_func.loss_G_l1.detach().cpu())

    def on_batch_end(self, last_input, last_output, **kwargs):
        self.G.zero_grad()
        fake_image = last_output[0].detach()
        input_label, real_image = last_input

        self._set_trainable(D=True)
        self.D.zero_grad()

        # Fake Detection and Loss
        self.pred_fake = self.discriminate(input_label, fake_image)
        self.pred_real = self.discriminate(input_label, real_image)

        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        self.D_fake_smter.add_value(self.loss_D_fake.detach().cpu())
        self.D_real_smter.add_value(self.loss_D_real.detach().cpu())

        loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        if self.learn.model.training == True:
            loss_D.backward()

        self.opt_D.step()
        self._set_trainable()

    def on_epoch_end(self, last_metrics, **kwargs):
        smter_list = [self.G_GAN_smter, self.D_fake_smter, self.D_real_smter]

        if self.model.feat_loss:
            smter_list.append(self.G_GAN_Feat_smter)
        if self.loss_func.vgg_loss:
            smter_list.append(self.G_VGG_smter)
        if self.loss_func.l1_loss:
            smter_list.append(self.G_l1_smter)

        return add_metrics(last_metrics, [s.smooth for s in smter_list])
