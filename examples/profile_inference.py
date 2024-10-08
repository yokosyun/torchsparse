import time
import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.cuda
import torch.nn as nn
import torch.optim

NUM_PC_CHANNELS = 4
VOXEL_SIZE = (0.1, 0.1, 0.1)
MAX_X = 200
MAX_Y = 200
MAX_Z = 4
NUM_PC = MAX_X * MAX_Y * MAX_Z

import torchsparse
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.tensor_cache import TensorCache


def generate_random_point_cloud(size, voxel_size):
    pc = np.fromfile(
        "data/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin",
        dtype=np.float32,
    ).reshape(-1, 5)[:, :4]
    labels = np.random.choice(10, size)

    if use_sparse:
        coords, feats = pc[:, :3], pc
        coords -= np.min(coords, axis=0, keepdims=True)

        torch.cuda.synchronize()
        start_time = time.time()
        coords, indices = sparse_quantize(coords, voxel_size, return_index=True)
        torch.cuda.synchronize()
        end_time = time.time()
        print("quantization=", (end_time - start_time) * 1000, "[ms]")

        torch.cuda.synchronize()
        start_time = time.time()
        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats[indices], dtype=torch.float)
        labels = torch.tensor(labels[indices], dtype=torch.long)
        input = SparseTensor(coords=coords, feats=feats)
        label = SparseTensor(coords=coords, feats=labels)
        torch.cuda.synchronize()
        end_time = time.time()
        print("SparseTensor=", (end_time - start_time) * 1000, "[ms]")
    else:
        feats = torch.zeros(NUM_PC_CHANNELS,
                            MAX_X,
                            MAX_Y,
                            MAX_Z,
                            dtype=torch.float)
        labels = torch.zeros(MAX_X, MAX_Y, MAX_Z, dtype=torch.long)
        input = feats
        label = labels

    feed_dict = {"input": input, "label": label}

    return feed_dict


def generate_batched_random_point_clouds(size, voxel_size, batch_size=1):
    batch = []
    for _ in range(batch_size):
        batch.append(generate_random_point_cloud(size, voxel_size))
    return sparse_collate_fn(batch)


def dummy_train_3x3(device):
    torch.cuda.set_device(0)
    if False:
        model = nn.Sequential(
            spnn.Conv3d(4, 16, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            # spnn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            # spnn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            # spnn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1, **kargs2),
            # spnn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1, **kargs2),
            # spnn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1, **kargs2),
            # spnn.Conv3d(32, 10, kernel_size=3, stride=1, padding=1, **kargs2),
        )
    elif True:
        model = nn.Sequential(
            spnn.Conv3d(4, 16, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),

            # spnn.MaxPool3d(kernel_size=2),
            spnn.Conv3d(16, 32, kernel_size=2, stride=2, padding=0),
            # spnn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            spnn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(32, 64, kernel_size=2, stride=2, padding=0),
            # spnn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(64, 128, kernel_size=2, stride=2, padding=0),
            # spnn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            spnn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            spnn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),

            # spnn.Conv3d(128, 128, kernel_size=, stride=1, padding=1),
        )
    elif False:
        model = nn.Sequential(
            spnn.Conv3d(4, 16, kernel_size=7, stride=1, padding=1),
            spnn.Conv3d(16, 16, kernel_size=7, stride=1, padding=1),
            spnn.Conv3d(16, 16, kernel_size=7, stride=1, padding=1),
            spnn.Conv3d(16, 16, kernel_size=7, stride=1, padding=1),
            spnn.Conv3d(16, 16, kernel_size=7, stride=1, padding=1),
            spnn.MaxPool3d(kernel_size=2),
            spnn.Conv3d(16, 64, kernel_size=5, stride=1, padding=1),
            spnn.Conv3d(64, 64, kernel_size=5, stride=1, padding=1),
            spnn.Conv3d(64, 64, kernel_size=5, stride=1, padding=1),
            spnn.Conv3d(64, 64, kernel_size=5, stride=1, padding=1),
            spnn.Conv3d(64, 64, kernel_size=5, stride=1, padding=1),
        )
    else:
        from torchsparse.backbones.resnet import SparseResNet21D
        from torchsparse.backbones.unet import SparseResUNet42
        model = SparseResNet21D()
        # model = SparseResUNet42()

    model = model.to(device)
    model.eval()

    feed_dict = generate_batched_random_point_clouds(size=NUM_PC,
                                                     voxel_size=VOXEL_SIZE)
    print(feed_dict["input"].feats.shape)
    print(feed_dict["input"].coords.shape)

    with torch.no_grad():
        if False:
            with torch.profiler.profile(
                    profile_memory=True,
                    use_cuda=True,
                    schedule=torch.profiler.schedule(wait=1,
                                                     warmup=2,
                                                     active=10,
                                                     repeat=2),
            ) as prof:
                with profiler.record_function("model_inference"):
                    for _ in range(100):
                        inputs = feed_dict["input"].to(device)
                        inputs._caches = TensorCache()
                        model(inputs)
                        prof.step()

            print(prof.key_averages().table(sort_by="cuda_time_total",
                                            row_limit=10))
            prof.export_chrome_trace("trace_dummy_3x3.json")

        else:
            inputs = feed_dict["input"].to(device)

            warmup_iter = 10
            for _ in range(warmup_iter):
                inputs._caches = TensorCache()
                model(inputs)

            time_buffer = []

            active_iter = 100

            for iter in range(active_iter):
                print("------active_iter---------", iter)
                start_time = time.time()
                inputs._caches = TensorCache()
                outputs = model(inputs)
                print(outputs.coords.shape, outputs.stride,
                      outputs.spatial_range)
                end_time = time.time()
                duration = (end_time - start_time) * 1000

                time_buffer.append(duration)

            forward_time = np.median(time_buffer)
            print(f"forward_time= {forward_time:.2f} [ms]")


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    use_sparse = True
    import torchsparse.nn as spnn

    kargs2 = {"transposed": True}
    dummy_train_3x3(device)
