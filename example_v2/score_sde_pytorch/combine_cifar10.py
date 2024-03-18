import os
import torch
import torchvision
import numpy as np
from PIL import Image 
# root = "./dataset"
# train_ds = torchvision.datasets.CIFAR10(root=os.path.join(root, 'CIFAR10'),
                                                
#                                                 train=True,
#                                                 download=False)

# arr = []
# for img,_ in iter(train_ds):
#     arr.append(np.array(img))
# arr = np.stack(arr)
# np.savez("cifar10_all.npz",arr)
# root = "./multistage/multistage_gn16_v2_old_interval/eval_1/ckpt_9_host_0"
# root = "./meme_128/test/ckpt_3_host_0"
root = "/scratch/qingqu_root/qingqu1/shared_data/multistage/edm/edm_multistage_cifar10/fid-tmp2"
arr = []
for i in os.listdir(root)[:50000]:
    s = Image.open(os.path.join(root,"{}".format(i))).convert('RGB')
    arr.append(np.array(s))
arr = np.stack(arr)
print(arr.shape)
print(arr.max())
np.savez("edm-mul.npz",arr)