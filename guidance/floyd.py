
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from diffusers import IFPipeline

# class Floyd(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.dtype = torch.float16

#         model = IFPipeline("", variant="", torch_dtype=self.dtype)