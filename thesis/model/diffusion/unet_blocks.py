import torch
import torch.nn as nn
import torch.nn.functional as F

class DownSample1D(nn.Module):
    def __init__(self, dim):
        """Down samples the action sequence by 2 with a stride 2 conv

        Args:
            Dim: num of channels, num in and out channels are the same
        """
        super().__init__()
        
        # conv layer
        in_channels = dim
        out_channels = dim
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        # return convolved output
        return self.conv(x)
        

class UpSample1D(nn.Module):
    """Up samples the action sequence by 2 with a stride 2 conv transpose

    Args:
        Dim: num of channels, num in and out channels are the same
    """
    def __init__(self, dim):
        super().__init__()
        
        # transpose conv layer
        in_channels = dim
        out_channels = dim
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        # return transpose convolved output
        return self.conv(x)
    

class Conv1DBlock(nn.Module):
    """: Learn features with conv. Size of action sequence stays the same.

    Args:
       in_channels: num of input channels
       out_channels: num of output channels
       kernel_size: conv kernel size
       n_groups: num of groups for group norm
    """
    def __init__(self, in_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        
        # tensor shape: (batch_size, num_channels, seq_len)
        
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            # normalizes activations to make training more stable and faster
            nn.GroupNorm(n_groups, out_channels), 
            # add nonlinearity to the output of the conv layer
            nn.Mish(),
        )
        
    def forward(self, x):
        return self.block(x)

if __name__ == "__main__":
    def test():
        conv_block = Conv1DBlock(256, 128, kernel_size=3)
        x = torch.zeros((1, 256, 10)) # (batch_size, num_channels, seq_len)
        out = conv_block(x)
        print(out.shape)
    
    test()