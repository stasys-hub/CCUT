import torch

def get_device(device=None):
    """
    Determines the device to be used for PyTorch operations.

    This function returns the specified device if provided. If no device is 
    specified, it checks for CUDA availability and returns the first CUDA 
    device ("cuda:0") if available. If CUDA is not available, it returns the 
    CPU device ("cpu").

    Parameters:
    ----------
    device : str, optional
        The device to be used for PyTorch operations. If None, the function 
        will check for CUDA availability. Default is None.

    Returns:
    -------
    str
        The device to be used for PyTorch operations, either "cuda:0" or "cpu".
    """
    if device:
        return device
    else:
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        return device


def get_device_count():
    """
    Returns the number of available CUDA devices.

    This function returns the count of CUDA-capable devices available on the 
    current system.

    Returns:
    -------
    int
        The number of available CUDA devices.
    """
    return torch.cuda.device_count()


def print_cuda_devices():
    """
    Prints the properties of each available CUDA device.

    This function iterates over all available CUDA devices and prints the 
    properties (e.g., name) of each device.
    """
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)
