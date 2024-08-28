import torch
from max_pool import encode_coordinate

BXYZ_DIM = 4
XYZ_DIM = 3
L = 4
kernel_size = 2
kernel_volume = kernel_size ** 3

in_coords = torch.randint(1, 3, (L, BXYZ_DIM))

kernel_offset = in_coords[:, 1:] % 2
kernel_offset = kernel_offset[:,
                              0] + kernel_offset[:,
                                                 1] * 2 + kernel_offset[:,
                                                                        2] * 2 ** 2

enc_coords, min_coords, max_coords = encode_coordinate(in_coords)

enc_unique_coords, inv_idx = torch.unique(enc_coords,
                                          sorted=False,
                                          return_inverse=True,
                                          return_counts=False,
                                          dim=0)

print(enc_coords, enc_unique_coords)
print(inv_idx, kernel_offset)

scatter_index = inv_idx * kernel_volume + kernel_offset
out_in_map = torch.zeros(len(enc_unique_coords) * kernel_volume,
                         dtype=enc_unique_coords.dtype)
out_in_map = out_in_map.scatter_(0, scatter_index, enc_coords)

print(out_in_map.reshape(-1, kernel_volume))
