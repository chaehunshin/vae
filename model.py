import torch
import torch.nn as nn

import math

class VAE(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim, output_dim=None):
        super(VAE, self).__init__()
        self._input_dim = input_dim
        self._output_dim = input_dim if output_dim is None else output_dim
        self._hidden_dim = hidden_dim
        self._z_dim = z_dim

        lrelu_slope=1e-2
        self._encoder = nn.ModuleList([
            nn.Conv2d(input_dim, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(lrelu_slope),
            nn.Conv2d(hidden_dim, hidden_dim*2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(lrelu_slope),
            nn.Conv2d(hidden_dim*2, hidden_dim*4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.LeakyReLU(lrelu_slope),
            nn.Conv2d(hidden_dim*4, hidden_dim*8, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim*8),
            nn.LeakyReLU(lrelu_slope),
            #nn.Conv2d(hidden_dim*8, hidden_dim*8, 4, 2, 1),
            #nn.BatchNorm2d(hidden_dim*8),
            #nn.LeakyReLU()
        ])

        self._mu = nn.Linear(hidden_dim*8*4*4, z_dim)
        self._logvar = nn.Linear(hidden_dim*8*4*4, z_dim)

        #self._decoder_pre = nn.Linear(z_dim, hidden_dim*8*2*4*4)
        self._decoder_pre = nn.Linear(z_dim, hidden_dim*8*4*4)
        self._decoder = nn.ModuleList([
            #nn.UpsamplingNearest2d(scale_factor=2),
            #nn.ReplicationPad2d(1),
            #nn.Conv2d(hidden_dim*8*2, hidden_dim*8, 3, 1),
            #nn.BatchNorm2d(hidden_dim*8, 1.e-3),
            #nn.LeakyReLU(),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReplicationPad2d(1),
            nn.Conv2d(hidden_dim*8, hidden_dim*4, 3, 1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.LeakyReLU(lrelu_slope),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReplicationPad2d(1),
            nn.Conv2d(hidden_dim*4, hidden_dim*2, 3, 1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(lrelu_slope),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReplicationPad2d(1),
            nn.Conv2d(hidden_dim*2, hidden_dim, 3, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(lrelu_slope),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReplicationPad2d(1),
            nn.Conv2d(hidden_dim, self._output_dim, 3, 1),
            nn.Sigmoid()
        ])

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.xavier_normal_(m.weight, math.sqrt(2./n))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                n = m.in_features + m.out_features
                nn.init.xavier_normal_(m.weight, math.sqrt(2./n))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        out = x
        for m in self._encoder:
            out = m(out)
        out = out.reshape(out.shape[0], -1)
        mu = self._mu(out)
        logvar = self._logvar(out)
        return mu, logvar


    def reparameterize(self, mu, logvar):
        eps = torch.rand_like(mu)
        std = torch.exp(0.5 * logvar)
        return eps*std + mu

    def decode(self, z):
        out = self._decoder_pre(z).reshape(z.shape[0], -1, 4, 4)
        for m in self._decoder:
            out = m(out)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)

        return out, mu, logvar
