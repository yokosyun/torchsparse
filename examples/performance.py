from datetime import datetime

import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.cuda
import torch.nn as nn
import torch.optim


NUM_PC_CHANNELS = 4
VOXEL_SIZE = 1.0
MAX_X = 200
MAX_Y = 200
MAX_Z = 4
NUM_PC = MAX_X * MAX_Y * MAX_Z


from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize


def generate_random_point_cloud(size=100000, voxel_size=0.2):
    pc = np.random.randn(size, NUM_PC_CHANNELS)
    pc[:, :3] = pc[:, :3] * 10
    labels = np.random.choice(10, size)

    if use_sparse:
        coords, feats = pc[:, :3], pc
        coords -= np.min(coords, axis=0, keepdims=True)
        coords, indices = sparse_quantize(coords, voxel_size, return_index=True)

        # print(coords.shape[0] / feats.shape[0] * 100, "[%]")

        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats[indices], dtype=torch.float)
        labels = torch.tensor(labels[indices], dtype=torch.long)
        input = SparseTensor(coords=coords, feats=feats)
        label = SparseTensor(coords=coords, feats=labels)
    else:
        feats = torch.zeros(NUM_PC_CHANNELS, MAX_X, MAX_Y, MAX_Z, dtype=torch.float)
        labels = torch.zeros(MAX_X, MAX_Y, MAX_Z, dtype=torch.long)
        input = feats
        label = labels

    feed_dict = {"input": input, "label": label}

    return feed_dict


def generate_batched_random_point_clouds(size=100000, voxel_size=0.2, batch_size=2):
    batch = []
    for _ in range(batch_size):
        batch.append(generate_random_point_cloud(size, voxel_size))
    return sparse_collate_fn(batch)


def dummy_train_3x3(device):

    model = nn.Sequential(
        spnn.Conv3d(4, 32, kernel_size=3, stride=1, **kargs1),
        spnn.Conv3d(32, 64, kernel_size=3, stride=1, **kargs1),
        spnn.Conv3d(64, 128, kernel_size=3, stride=1, **kargs1),
        spnn.Conv3d(128, 256, kernel_size=3, stride=1, **kargs1),
        spnn.Conv3d(256, 128, kernel_size=3, stride=1, **kargs1, **kargs2),
        spnn.Conv3d(128, 64, kernel_size=3, stride=1, **kargs1, **kargs2),
        spnn.Conv3d(64, 32, kernel_size=3, stride=1, **kargs1, **kargs2),
        spnn.Conv3d(32, 10, kernel_size=3, stride=1, **kargs1, **kargs2),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)

    print("Starting dummy_train_3x3...")
    time = datetime.now()
    with profiler.profile(profile_memory=True, use_cuda=True) as prof:
        with profiler.record_function("model_inference"):
            for _ in range(10):
                feed_dict = generate_batched_random_point_clouds(
                    size=NUM_PC, voxel_size=VOXEL_SIZE
                )
                inputs = feed_dict["input"].to(device)
                if use_sparse:
                    targets = feed_dict["label"].F.to(device).long()
                else:
                    targets = feed_dict["label"].to(device).long()
                outputs = model(inputs)
                # optimizer.zero_grad()
                # if use_sparse:
                #     loss = criterion(outputs.F, targets)
                # else:
                #     loss = criterion(outputs, targets)
                # loss.backward()
                # optimizer.step()
                # print('[step %d] loss = %f.'%(i, loss.item()))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("trace_dummy_3x3.json")

    time = datetime.now() - time
    print("Finished dummy_train_3x3 in ", time)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    use_sparse = False
    import torch.nn as spnn

    kargs1 = {"padding": 1}
    kargs2 = {}
    dummy_train_3x3(device)

    use_sparse = True
    import torchsparse.nn as spnn

    kargs1 = {}
    kargs2 = {"transposed": True}
    dummy_train_3x3(device)
