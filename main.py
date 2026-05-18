import torch
import torch.nn as nn
from custom.unet_dualstream import DualStreamEncBlock, LCSCSkipFuse3, ConvBlock


class DSPCNet(nn.Module):
    """
    DSPC-Net: Core Network Architecture for Inference
    """

    def __init__(
            self,
            in_channels=2,
            seg_out_channels=2,
            base_channels=64,
            n_blocks=4,
            k_struct=5,
            dil_struct=1,
            r_gate=4
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.in_channels = in_channels

        # ---- Encoder ----
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.enc_channels = []

        ch = in_channels
        out_ch = base_channels
        for i in range(n_blocks):
            self.enc_blocks.append(
                DualStreamEncBlock(ch, out_ch, k_struct=k_struct, dil=dil_struct, r_gate=r_gate)
            )
            self.enc_channels.append(out_ch)
            if i < n_blocks - 1:
                self.pools.append(nn.MaxPool2d(2))
            ch = out_ch
            out_ch *= 2

        # ---- Decoder & LCSC Module ----
        # The decoder consists of 3 upsampling stages to reconstruct the original resolution
        # corresponding to the 4-block encoder.
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.skip_fusers = nn.ModuleList()

        cur_ch = self.enc_channels[-1]
        for i in range(n_blocks - 1):
            same_ch = self.enc_channels[-(i + 2)]
            deep_ch = cur_ch
            idx = -(i + 3)
            deeper_ch = self.enc_channels[idx] if (i + 3) <= len(self.enc_channels) else same_ch

            self.up_convs.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.skip_fusers.append(LCSCSkipFuse3(same_ch, deep_ch, deeper_ch, c_out=same_ch))
            self.dec_blocks.append(ConvBlock(cur_ch + same_ch, same_ch))
            cur_ch = same_ch

        # ---- Output Heads ----
        self.seg_head = nn.Conv2d(base_channels, seg_out_channels, 1)
        self.bd_head = nn.Conv2d(base_channels, 1, 1)
        self.cav_head = nn.Conv2d(base_channels, 1, 1)

    def forward_encoder(self, x):
        skips = []
        for i, block in enumerate(self.enc_blocks):
            x = block(x)
            if i < self.n_blocks - 1:
                skips.append(x)
                x = self.pools[i](x)
        return x, skips

    def forward_decoder(self, bottleneck, skips):
        x = bottleneck
        for i, (up, fuser, block) in enumerate(zip(self.up_convs, self.skip_fusers, self.dec_blocks)):
            x = up(x)

            # skip_same maintains the highest resolution of the current layer.
            # It is directly used as the base size for feature fusion without upsampling.
            skip_same = skips[-(i + 1)]
            skip_deeper = skips[-(i + 2)] if i + 2 <= len(skips) else skip_same

            skip_fused = fuser(skip_same, x, skip_deeper)
            x = block(torch.cat([x, skip_fused], dim=1))
        return x

    def forward(self, x):
        bottleneck, skips = self.forward_encoder(x)
        feat = self.forward_decoder(bottleneck, skips)

        # Return necessary outputs for inference evaluation
        out = {
            "seg": self.seg_head(feat),
            "bd": self.bd_head(feat),
            "cav": self.cav_head(feat)
        }
        return out