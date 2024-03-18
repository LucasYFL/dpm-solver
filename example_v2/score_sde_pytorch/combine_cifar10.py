import os
import torch
import torchvision
import numpy as np
# root = "./dataset"
# train_ds = torchvision.datasets.CIFAR10(root=os.path.join(root, 'CIFAR10'),
                                                
#                                                 train=True,
#                                                 download=False)

# arr = []
# for img,_ in iter(train_ds):
#     arr.append(np.array(img))
# arr = np.stack(arr)
# np.savez("cifar10_all.npz",arr)
root = "./multistage/multistage_gn16_v2_old_interval/eval_1/ckpt_9_host_0"
arr = []
for i in range(49):
    s = np.load(os.path.join(root,"samples_{}.npz".format(i)))['samples']
    arr.append(s)
arr = np.concatenate(arr)
print(arr.shape)
np.savez("mul.npz",arr)