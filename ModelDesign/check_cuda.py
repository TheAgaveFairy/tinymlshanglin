#!/usr/bin/python3

import torch

print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())
print("torch.cuda.current_device:", torch.cuda.current_device())
count = torch.cuda.device_count()
device = torch.cuda.current_device()
for c in range(count):
    t = "torch.cuda.device: " + str(c)
    dev = torch.cuda.device(c)
    print(t,dev)
    print("torch.cuda.get_device_name:",c, torch.cuda.get_device_name(dev))
