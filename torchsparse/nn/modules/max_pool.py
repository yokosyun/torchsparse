import torch
from torch import Tensor


def encode_coordinate(coords: Tensor):
    """_summary_

    Args:
        coords (Tensor): [L, 1 + COORD]

    Return
        [L]
    """
    L, COORDS = coords.shape

    coords = coords.clone()
    min_coords, _ = torch.min(coords, dim=0)
    coords = coords - min_coords
    max_coords, _ = torch.max(coords, dim=0)
    max_coords += 1

    encoded_coords = torch.zeros(L, dtype=coords.dtype, device=coords.device)
    for idx in range(COORDS - 1):
        encoded_coords += coords[:, idx]
        encoded_coords *= max_coords[idx + 1]
    encoded_coords += coords[:, -1]

    return encoded_coords, min_coords, max_coords


def decode_coordinate(encoded_coords, min_coords, max_coords):
    COORDS = 4

    L = len(encoded_coords)

    out_coords = torch.zeros(L,
                             COORDS,
                             dtype=encoded_coords.dtype,
                             device=encoded_coords.device)

    for idx in range(COORDS - 1, -1, -1):
        remain = encoded_coords % max_coords[idx]
        out_coords[:, idx] = min_coords[idx] + remain
        encoded_coords -= remain
        encoded_coords = encoded_coords / max_coords[idx]

    return out_coords


def sparse_max_pool_1d(in_feats: Tensor, in_coords: Tensor,
                       scale_factor: float) -> Tensor:
    """
        in_feats: [L C]
        in_coords: [L, COORD]
    """

    out_coords = in_coords.clone()
    out_coords[:, 1:] = in_coords[:, 1:] // scale_factor

    print(in_feats, out_coords)

    encoded_coords, min_coords, max_coords = encode_coordinate(out_coords)

    print(encoded_coords, min_coords, max_coords)

    encoded_coords, inv_idx = torch.unique(encoded_coords,
                                           sorted=False,
                                           return_inverse=True,
                                           return_counts=False,
                                           dim=0)

    out_feats = torch.empty(len(encoded_coords), in_feats.shape[1])

    # reduce same coordinate
    out_feats.index_reduce_(0, inv_idx, in_feats, 'amax', include_self=False)

    out_coords = decode_coordinate(encoded_coords, min_coords, max_coords)

    return out_feats, out_coords


def generate_rand_input():
    B = 2
    C = 2
    L = 3
    COORDS = 3  # [X,Y,Z]

    feats_list = []
    coords_list = []

    for b_idx in range(B):
        feats = torch.rand(L, C)
        coords = torch.randint(0, 4, (L, COORDS))
        feats_list.append(feats)

        batch_idx = torch.full((L, 1),
                               b_idx,
                               dtype=coords.dtype,
                               device=coords.device)
        coords_list.append(torch.cat((batch_idx, coords), dim=1))

    return torch.cat(feats_list, dim=0), torch.cat(coords_list, dim=0)


in_feats, in_coords = generate_rand_input()
# print(in_feats.shape, in_coords.shape)
# print(in_feats, in_coords)
out_feats, out_coords = sparse_max_pool_1d(in_feats=in_feats,
                                           in_coords=in_coords,
                                           scale_factor=2.0)
print(out_feats.shape, out_coords.shape)
print(out_feats, out_coords)
