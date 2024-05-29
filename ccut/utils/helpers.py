import torch
import torch.nn as nn


def get_device(device=None):
    if device:
        return device
    else:
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        return device


def get_device_count():
    return torch.cuda.device_count()


def print_cuda_devices():
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)
