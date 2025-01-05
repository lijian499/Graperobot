class PAN(nn.Module):

    def __init__(self, ch=[256, 256, 512], channel_outs=[256, 512, 512, 1024], version='s'):
        super(PAN, self).__init__()
        self.version = str(version)
        self.channels_outs = channel_outs

        if self.version.lower() in gains:
            self.gd = gains[self.version.lower()]['gd']
            self.gw = gains[self.version.lower()]['gw']
        else:
            self.gd = 0.33
            self.gw = 0.5

        self.re_channels_out()
        self.P3_size = ch[0]
        self.P4_size = ch[1]
        self.P5_size = ch[2]
        self.convP3 = Conv(self.P3_size,  self.channels_outs[0], 3, 2)
        self.P4 = C3(self.channels_outs[0] + self.P4_size, self.channels_outs[1], self.get_depth(3), False)
        self.convP4 = Conv(self.channels_outs[1], self.channels_outs[2], 3, 2)
        self.P5 = C3(self.channels_outs[2] + self.P5_size, self.channels_outs[3], self.get_depth(3), False)
        self.concat = Concat()
        self.out_shape = [self.P3_size, self.channels_outs[2], self.channels_outs[3]]

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for idx, channel_out in enumerate(self.channels_outs):
            self.channels_outs[idx] = self.get_width(channel_out)

    def forward(self, inputs):
        PP3, P4, P5 = inputs
        convp3 = self.convP3(PP3)
        concat3_4 = self.concat([convp3, P4])
        PP4 = self.P4(concat3_4)
        convp4 = self.convP4(PP4)
        concat4_5 = self.concat([convp4, P5])
        PP5 = self.P5(concat4_5)

        return PP3, PP4, PP5
