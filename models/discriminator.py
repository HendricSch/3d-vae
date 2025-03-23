import torch.nn as nn
import functools


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc: int, output_nc: int, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        norm_layer = nn.BatchNorm2d

        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) is functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            # output prediction map
            nn.Conv2d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class PatchGANDiscriminator(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, channels: int):
        super(PatchGANDiscriminator, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=channels * 2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 2,
                out_channels=channels * 2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 2,
                out_channels=channels * 4,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 4,
                out_channels=channels * 8,
                kernel_size=4,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 8,
                out_channels=channels * 8,
                kernel_size=4,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 8,
                out_channels=channels * 8,
                kernel_size=4,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=channels * 8,
                out_channels=out_channels,
                kernel_size=4,
                stride=1,
                padding=1
            ),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.out(x)
        return x
