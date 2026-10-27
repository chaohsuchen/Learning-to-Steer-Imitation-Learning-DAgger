import torch
import argparse

# selct device automatically：
#   M series Mac (Apple Silicon)  -> mps (Apple 的 GPU backend)
#   NVIDIA GPU              -> cuda
#   Others                       -> cpu
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using mps...")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU...")


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
