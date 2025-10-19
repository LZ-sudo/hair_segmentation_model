"""
BiSeNet model for hair segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBNReLU(nn.Module):
    """Basic convolution block with BatchNorm and ReLU."""
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, 
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BiSeNetOutput(nn.Module):
    """Output layer for BiSeNet."""
    def __init__(self, in_chan, mid_chan, num_classes):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class AttentionRefinementModule(nn.Module):
    """Attention Refinement Module."""
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out


class ContextPath(nn.Module):
    """Context Path using ResNet backbone."""
    def __init__(self):
        super(ContextPath, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32 = self.get_resnet_features(x)
        
        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, feat32.size()[2:], mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, feat16.size()[2:], mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, feat8.size()[2:], mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up

    def get_resnet_features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        feat8 = self.resnet.layer2(x)
        feat16 = self.resnet.layer3(feat8)
        feat32 = self.resnet.layer4(feat16)
        return feat8, feat16, feat32


class FeatureFusionModule(nn.Module):
    """Feature Fusion Module."""
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class BiSeNet(nn.Module):
    """BiSeNet model for semantic segmentation."""
    def __init__(self, num_classes):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, num_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, num_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, num_classes)

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)

        return feat_out


class HairSegmentationModel:
    """Wrapper class for BiSeNet hair segmentation."""
    
    def __init__(self, model_path, num_classes=19, device='cuda'):
        """
        Initialize the hair segmentation model.
        
        Args:
            model_path: Path to pretrained weights
            num_classes: Number of segmentation classes
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Initialize model
        self.model = BiSeNet(num_classes=num_classes)
        self.load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def load_weights(self, model_path):
        """Load pretrained weights."""
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded weights from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {model_path}: {e}")
            print("Using randomly initialized weights")
    
    def predict(self, input_tensor):
        """
        Run inference on input tensor.
        
        Args:
            input_tensor: Preprocessed image tensor [1, 3, H, W]
            
        Returns:
            Segmentation output [1, num_classes, H, W]
        """
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
        return output
    
    def extract_hair_mask(self, output, hair_class_idx=17):
        """
        Extract hair class from segmentation output.
        
        Args:
            output: Model output [1, num_classes, H, W]
            hair_class_idx: Index of hair class
            
        Returns:
            Binary mask [H, W] with hair pixels
        """
        # Get class predictions
        pred = output.squeeze(0).cpu().numpy()
        pred = pred.argmax(0)  # [H, W]
        
        # Extract hair mask
        hair_mask = (pred == hair_class_idx).astype('uint8')
        return hair_mask