from typing import Tuple
import torch
from torch import Tensor, nn
from torchsparse import SparseTensor

__all__ = ["MaxPool3d"]

XYZ_DIM = 3  # [X,Y,Z]
BXYZ_DIM = 4  # [B,X,Y,Z]


def encode_coordinate(coords: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """ (((b) * X_SIZE + y) * Y_SIZE + z ) * Z_SIZE + z
    Args:
        coords (Tensor): [L, BXYZ_DIM]

    Return
        Tensor: encoded coordinate. shape[L]
        Tensor: minimum coordiante. shppe[BXYZ_DIM]
        Tensor: maximum coordinate. shape[BXYZ_DIM]
    """

    assert coords.shape[1] == BXYZ_DIM

    coords = coords.clone()
    min_coords, _ = torch.min(coords, dim=0)
    coords = coords - min_coords
    max_coords, _ = torch.max(coords, dim=0)
    max_coords += 1

    enc_coords = torch.zeros(coords.shape[0],
                             dtype=coords.dtype,
                             device=coords.device)
    for idx in range(BXYZ_DIM - 1):
        enc_coords += coords[:, idx]
        enc_coords *= max_coords[idx + 1]
    enc_coords += coords[:, -1]

    return enc_coords, min_coords, max_coords


def decode_coordinate(enc_coords: Tensor, min_coords: Tensor,
                      max_coords: Tensor) -> Tensor:
    """
    Args:
        enc_coords (Tensor): encoded coordinate. shape[L]
        min_coords (Tensor): minimum coordinate. shape[BXYZ_DIM]
        max_coords (Tensor): maximum coordinate. shape[BXYZ_DIM]

    Returns:
        Tensor: decoded coordinate [L, BXYZ_DIM]
    """

    out_coords = torch.zeros(len(enc_coords),
                             BXYZ_DIM,
                             dtype=enc_coords.dtype,
                             device=enc_coords.device)

    for idx in range(BXYZ_DIM - 1, -1, -1):
        remain = enc_coords % max_coords[idx]
        out_coords[:, idx] = min_coords[idx] + remain
        enc_coords -= remain
        enc_coords = enc_coords / max_coords[idx]

    return out_coords


def sparse_max_pool_1d(in_feats: Tensor, in_coords: Tensor,
                       kernel_size: int) -> Tuple[Tensor, Tensor]:
    """_summary_

    Args:
        in_feats (Tensor): shape[L, C]
        in_coords (Tensor): shape[L, BXYZ_DIM]
        kernel_size (int): downsample ratio

    Returns:
        Tuple[Tensor, Tensor]: out_feats[L*, C], out_coords[L*, BXYZ_DIM]
    """

    out_coords = in_coords.clone()
    out_coords[:, 1:] = in_coords[:, 1:] // kernel_size

    enc_coords, min_coords, max_coords = encode_coordinate(out_coords)

    enc_coords, inv_idx = torch.unique(enc_coords,
                                       sorted=False,
                                       return_inverse=True,
                                       return_counts=False,
                                       dim=0)

    out_feats = torch.empty(len(enc_coords),
                            in_feats.shape[1],
                            dtype=in_feats.dtype,
                            device=in_feats.device)

    # reduce same coordinate
    out_feats.index_reduce_(0, inv_idx, in_feats, 'amax', include_self=False)

    out_coords = decode_coordinate(enc_coords, min_coords, max_coords)

    return out_feats, out_coords


def generate_rand_input(batch_size=2,
                        channel_size=2,
                        length=3) -> Tuple[Tensor, Tensor]:
    feats_list = []
    coords_list = []

    for b_idx in range(batch_size):
        feats = torch.rand(length, channel_size)
        coords = torch.randint(0, 4, (length, XYZ_DIM))
        feats_list.append(feats)

        batch_idx = torch.full((length, 1),
                               b_idx,
                               dtype=coords.dtype,
                               device=coords.device)
        coords_list.append(torch.cat((batch_idx, coords), dim=1))

    return torch.cat(feats_list, dim=0), torch.cat(coords_list, dim=0)


class MaxPool3d(nn.Module):

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        assert isinstance(kernel_size, int), "currently support only integer"
        self.kernel_size = kernel_size

    def forward(self, input: SparseTensor) -> SparseTensor:
        out_feats, out_coords = sparse_max_pool_1d(in_feats=input.feats,
                                                   in_coords=input.coords,
                                                   kernel_size=self.kernel_size)

        output = SparseTensor(
            coords=out_coords,
            feats=out_feats,
            stride=[stride * self.kernel_size for stride in input.stride],
            spatial_range=input.spatial_range,
        )

        output._caches = input._caches

        return output


def main():
    in_feats, in_coords = generate_rand_input()
    print(in_feats.shape, in_coords.shape)
    print(in_feats, in_coords)
    out_feats, out_coords = sparse_max_pool_1d(in_feats=in_feats,
                                               in_coords=in_coords,
                                               kernel_size=2)
    print(out_feats.shape, out_coords.shape)
    print(out_feats, out_coords)


if __name__ == "__main__":
    main()
