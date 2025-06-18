import torch
import torch.nn as nn
# import torch.nn.functional as F # F will be unused


class LayerNorm32(nn.LayerNorm): # Renamed back
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reverted to original implementation
        return super().forward(x.float()).type(x.dtype)
    

class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)
    
    
class ChannelLayerNorm32(LayerNorm32): # Ensure it inherits from LayerNorm32
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x
    
