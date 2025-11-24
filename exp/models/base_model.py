import torch.nn as nn

class BaseModel(nn.Module):
    def forward(self, x_imf, x_raw, x_ex):
        raise NotImplementedError
