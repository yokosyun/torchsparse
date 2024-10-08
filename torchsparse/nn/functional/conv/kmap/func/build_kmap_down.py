import torch
from torch import Tensor
from torchsparse.nn.modules.flat_coord import bxyz2flat, generate_rand_input, flat2bxyz
"""
<my kmap>
full -1
transform
unique is just build hashmap
scatter


<torchsparse kmap>
full -1
transform
build hashmap
trasform
8times lookup for hash
scatter
"""


def build_kernel_map_downsample(in_coords: Tensor,
                                coords_min: Tensor,
                                coords_max: Tensor,
                                kernel_size: int = 2):
    kernel_volume = kernel_size ** 3

    down_coords = torch.cat(
        (in_coords[:, 0:1], in_coords[:, 1:] // kernel_size), dim=1)

    kernel_offset = in_coords[:, 1:] % 2
    kernel_offset = kernel_offset[:,
                                  2] + kernel_offset[:,
                                                     1] * 2 + kernel_offset[:,
                                                                            0] * 2 ** 2
    enc_coords = bxyz2flat(down_coords, coords_min, coords_max)

    enc_unique_coords, out_idx = torch.unique(enc_coords,
                                              sorted=False,
                                              return_inverse=True,
                                              return_counts=False,
                                              dim=0)

    scatter_index = out_idx * kernel_volume + kernel_offset
    divisor = 128
    n_out_coords = (len(enc_unique_coords) + divisor - 1) // divisor * divisor
    out_in_map = torch.full([n_out_coords * kernel_volume],
                            -1,
                            dtype=in_coords.dtype,
                            device=in_coords.device)

    in_idx = torch.arange(len(in_coords),
                          device=out_in_map.device,
                          dtype=out_in_map.dtype)

    out_in_map = out_in_map.scatter_(0, scatter_index, in_idx)

    out_coords = flat2bxyz(enc_unique_coords, coords_min, coords_max)

    return out_in_map.reshape(-1, kernel_volume), out_coords


def main():
    _, in_coords = generate_rand_input()
    out_in_map = build_kernel_map_downsample(in_coords)
    print(out_in_map)


if __name__ == "__main__":
    main()
