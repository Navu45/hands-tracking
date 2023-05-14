import torch

fcn = torch.jit.load('data/models/torch-script/moganet.pt').cuda()
fcn = fcn.eval()
print(fcn(torch.rand(16, 3, 256, 256).cuda()))