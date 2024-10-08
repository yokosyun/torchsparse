import unittest
from torchsparse.nn import functional as F
from python import (
    test_single_layer_convolution_forward,
    test_to_dense_forward,
)


class SparseConvTestCase(unittest.TestCase):

    def test_single_layer(self):
        # kernel_sizes = [2, 3, 5]
        kernel_sizes = [3]
        strides = [1]

        config = F.conv_config.get_default_conv_config()
        config.kmap_mode = "hashmap_on_the_fly"
        print("----------------------------")
        print(config)

        acc_adiff = 0.0
        acc_rdiff = 0.0
        count = 0
        # hashmap mode by default
        for kernel_size in kernel_sizes:
            for stride in strides:
                mean_adiff, max_rdiff = test_single_layer_convolution_forward(
                    kernel_size=kernel_size, stride=stride)
                acc_adiff += mean_adiff
                acc_rdiff += max_rdiff
                count += 1
                print("kernel_size=", kernel_size, "stride=", stride,
                      "mean_adiff=", mean_adiff, "acc_rdiff=", acc_rdiff)

        self.assertLessEqual(acc_adiff / count, 1e-3)
        self.assertLessEqual(acc_rdiff / count, 1e-2)

        # switch to hashmap_on_the_fly
        # config = F.conv_config.get_default_conv_config()
        # config.kmap_mode = "hashmap"
        # F.conv_config.set_global_conv_config(config)
        # print("----------------------------")
        # print(config)

        # acc_adiff = 0.0
        # acc_rdiff = 0.0
        # count = 0
        # for kernel_size in kernel_sizes:
        #     for stride in strides:
        #         mean_adiff, max_rdiff = test_single_layer_convolution_forward(
        #             kernel_size=kernel_size, stride=stride)
        #         acc_adiff += mean_adiff
        #         acc_rdiff += max_rdiff
        #         count += 1
        #         print("kernel_size=", kernel_size, "stride=", stride,
        #               "mean_adiff=", mean_adiff, "acc_rdiff=", acc_rdiff)

        # self.assertLessEqual(acc_adiff / count, 1e-3)
        # self.assertLessEqual(acc_rdiff / count, 1e-2)


class ToDenseTestCase(unittest.TestCase):

    def test_to_dense(self):
        max_adiff = test_to_dense_forward()
        self.assertLessEqual(max_adiff, 1e-5)


if __name__ == "__main__":
    unittest.main()
