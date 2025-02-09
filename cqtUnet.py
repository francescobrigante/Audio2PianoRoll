import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop
import torch.nn.init as init

# Ensure spatial size match by cropping
def crop_to_match(tensor, target_size):
    _, _, H, W = target_size.shape  # Target height & width
    return center_crop(tensor, (H, W))  # Apply cropping

def pad_to_match(tensor, target_size):
    _, _, H, W = target_size.shape  # Target height & width
    _, _, h, w = tensor.shape  # Current height & width
    
    # Calculate padding required on each side
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    
    # If the padding difference is odd, we need to adjust the padding on one side
    pad_h_top = pad_h
    pad_h_bottom = pad_h + (H - h) % 2
    pad_w_left = pad_w
    pad_w_right = pad_w + (W - w) % 2
    
    return F.pad(tensor, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom))  # Pad tensor


class UNetPianoRoll(nn.Module):
    def __init__(self, in_channels=1, out_channels=53, dropout_rate=0.3):
        """
        U-Net model for predicting a piano roll from a CQT spectrogram section.
        - Uses BatchNorm & Dropout to improve generalization.
        
        - Input shape: (Batch, 1, 156, 86) where 156 is the number of frequency bins and 86 is the number of time bins.  
        - Output shape: (Batch, 53, 86) where 53 is the number of MIDI channels.
        """
        super(UNetPianoRoll, self).__init__()

        # ---- Encoder ----
        self.enc1 = self.conv_block(in_channels, 32, dropout_rate)
        self.enc2 = self.conv_block(32, 64, dropout_rate)
        self.enc3 = self.conv_block(64, 128, dropout_rate)
        self.enc4 = self.conv_block(128, 256, dropout_rate)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling

        # ---- Bottleneck ----
        self.bottleneck = self.conv_block(256, 512, dropout_rate)

        # ---- Decoder ----
        self.upconv4 = self.upconv_block(512, 256)
        self.dec4 = self.conv_block(512, 256, dropout_rate)

        self.upconv3 = self.upconv_block(256, 128)
        self.dec3 = self.conv_block(256, 128, dropout_rate)

        self.upconv2 = self.upconv_block(128, 64)
        self.dec2 = self.conv_block(128, 64, dropout_rate)

        self.upconv1 = self.upconv_block(64, 32)
        self.dec1 = self.conv_block(64, 32, dropout_rate)

        # ---- Final Output ----
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)  # Reduce to 53 MIDI channels
        self.freq_reduction = nn.Conv2d(out_channels, out_channels, kernel_size=(156,1), groups=out_channels)
        self.sigmoid = nn.Sigmoid()
        
        #weights initialization
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        """Apply weight initialization to the model"""
        if isinstance(m, nn.Conv2d):
            # He initialization for convolution layers (for ReLU)
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            # Xavier initialization for transposed convolutions
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            # Xavier initialization for linear layers
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            # Initialize batch norm to have mean=0 and variance=1
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def conv_block(self, in_channels, out_channels, dropout_rate):
        """Conv Block: Conv2D -> BatchNorm -> ReLU -> Dropout -> Conv2D -> BatchNorm -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def upconv_block(self, in_channels, out_channels):
        """Upconvolution (Transposed Convolution for Upsampling)"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        """Forward pass"""
        # ---- Encoder ----
        e1 = self.enc1(x) # (Batch, 32, 156, 86)
        p1 = self.pool(e1) # (Batch, 32, 78, 43)

        e2 = self.enc2(p1) # (Batch, 64, 78, 43)
        p2 = self.pool(e2) # (Batch, 64, 39, 21)

        e3 = self.enc3(p2) # (Batch, 128, 39, 21)
        p3 = self.pool(e3) # (Batch, 128, 19, 10)

        e4 = self.enc4(p3) # (Batch, 256, 19, 10)
        p4 = self.pool(e4) # (Batch, 256, 9, 5)

        # ---- Bottleneck ----
        b = self.bottleneck(p4) # (Batch, 512, 9, 5)

        # ---- Decoder ----
        # Upconv4
        u4 = self.upconv4(b)  # (Batch, 256, 18, 10)
        u4_padded = pad_to_match(u4, e4)  # pad `u4` to match `e4` (Batch, 256, 19, 10)
        d4 = self.dec4(torch.cat([u4_padded, e4], dim=1))  # Concatenate and apply conv block (Batch, 256, 19, 10)

        # Upconv3
        u3 = self.upconv3(d4)  # (Batch, 128, 36, 20)
        u3_padded = pad_to_match(u3, e3)  # pad `u3` to match `e3` (Batch, 128, 39, 21)
        d3 = self.dec3(torch.cat([u3_padded, e3], dim=1))  # Concatenate and apply conv block (Batch, 128, 39, 21)

        # Upconv2
        u2 = self.upconv2(d3)  # (Batch, 64, 78, 42)
        u2_padded = pad_to_match(u2, e2) # (Batch, 64, 78, 43)
        d2 = self.dec2(torch.cat([u2_padded, e2], dim=1)) # (Batch, 64, 78, 43)

        # Upconv1
        u1 = self.upconv1(d2) # (Batch, 32, 156, 86)
        d1 = self.dec1(torch.cat([u1, e1], dim=1)) # (Batch, 32, 156, 86)

        # ---- Final Output ----
        out = self.final_conv(d1)  # (Batch, 53, 156, 86)
        out = self.freq_reduction(out)     # (Batch, 53, 1, 86)
        out = out.squeeze(2)               # (Batch, 53, 86)
        out = self.sigmoid(out) # (Batch, 53, 86)

        return out