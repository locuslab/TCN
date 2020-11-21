import torch.nn.functional as F
from torch import nn
from TCN.tcan import TemporalConvNet


class TCAN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCAN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size,   num_sub_blocks = 2, temp_attn=True, nheads=1, en_res=True, conv=True, key_size = 25, dropout=dropout, visual=False)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)
    
    
    #self, emb_size, num_channels, num_sub_blocks, temp_attn, nheads, en_res,
                #conv, key_size, kernel_size, visual, vhdropout=0.0, dropout=0.2