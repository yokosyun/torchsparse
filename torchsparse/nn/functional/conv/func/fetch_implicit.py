from typing import Dict

import torch
from torch.autograd import Function

import torchsparse
import torchsparse.backend

__all__ = ["FetchImplicitConvolutionFuntion"]


class FetchImplicitConvolutionFuntion(Function):

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        kmap: Dict,
        config: Dict,
        transposed: bool = False,
    ) -> torch.Tensor:
        """if transposed:
            input_nbmaps = kmap["nbmaps"][1, :]
            output_nbmaps = kmap["nbmaps"][0, :]
        else:
            input_nbmaps = kmap["nbmaps"][0, :]
            output_nbmaps = kmap["nbmaps"][1, :]

        M = nbmaps.size(0)
        nbmaps_t = torch.zeros((2, M),
            dtype=torch.int, device=input.device, requires_grad=False)
        for l in range(M):
            nbmaps_t[0, l] = nbmaps[l, 0]
            nbmaps_t[1, l] = nbmaps[l, 1]"""

        nbmaps = kmap["nbmaps"]
        nbsizes = kmap["nbsizes"]
        nbaddrs = kmap["nbaddrs"]
        qnbaddrs = kmap["qnbaddrs"]
        sizes = kmap["sizes"]
        qmapsize = kmap["qmapsize"]

        mapsize = nbmaps.size(1)

        input = input.contiguous()
        weight = weight.contiguous()

        if not input.device.type == "cuda":
            if not transposed:
                output = torch.zeros(sizes[1],
                                     weight.size(-1),
                                     dtype=input.dtype,
                                     device=input.device)
            else:
                # TODO(Haotian): ensure the original, upsampled size to be the same.
                output = torch.zeros(sizes[0],
                                     weight.size(-1),
                                     dtype=input.dtype,
                                     device=input.device)

        if input.device.type == "cuda":
            if torch.float16 in [input.dtype, weight.dtype]:
                input = input.to(torch.float16)
                weight = weight.to(torch.float16)

            weight = torch.roll(weight, shifts=9, dims=0)
            weight_fod = weight[:18, :, :]
            weight_implicit = weight[18:, :, :]

            if config["FOD_fusion"] == True:
                fod_output = torchsparse.backend.conv_forward_fetch_on_demand_cuda(
                    input,
                    weight_fod.contiguous(),
                    nbmaps,
                    mapsize,
                    nbaddrs,
                    qnbaddrs,
                    sizes[1] if not transposed else sizes[0],
                    qmapsize,
                    transposed,
                    torchsparse.backends.allow_tf32,
                    torchsparse.backends.allow_fp16,
                )
            else:
                fod_output = (torchsparse.backend.
                              conv_forward_fetch_on_demand_no_fusion_cuda(
                                  input,
                                  weight_fod,
                                  nbmaps,
                                  nbsizes.cpu(),
                                  mapsize,
                                  sizes[1] if not transposed else sizes[0],
                                  transposed,
                                  torchsparse.backends.allow_tf32,
                                  torchsparse.backends.allow_fp16,
                              ))

            if not transposed:
                out_in_map = kmap["out_in_map"]
                reorder_out_in_map = kmap["reorder_out_in_map"]
                reduced_sorted_mask = kmap["reduced_sorted_mask"]
                reorder_loc = kmap["reorder_loc"]
            else:
                out_in_map = kmap["out_in_map_t"]
                reorder_out_in_map = kmap["reorder_out_in_map_t"]
                reduced_sorted_mask = kmap["reduced_sorted_mask_t"]
                reorder_loc = kmap["reorder_loc_t"]

            ifsort = config["ifsort"]
            num_out_feats = sizes[1] if not transposed else sizes[0]
            num_out_channels = weight_implicit.shape[-1]

            if not ifsort:
                assert False, "ifsort = False is not supported"
            else:
                output_implicit = torchsparse.backend.conv_forward_implicit_gemm_sorted_cuda(
                    input,
                    weight_implicit.contiguous(),
                    reorder_out_in_map.contiguous(),
                    reduced_sorted_mask,
                    reorder_loc,
                    num_out_feats,
                    num_out_channels,
                    torchsparse.backends.allow_tf32,
                    torchsparse.backends.allow_fp16,
                )
        else:
            raise NotImplementedError

        output = fod_output + output_implicit

        ctx.for_backwards = (input, weight, nbmaps, nbsizes, transposed)
        return output.to(weight.dtype)
