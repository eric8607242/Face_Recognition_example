# block_type, kernel_size, se, activation, kwargs
MOBILEFACENET_CFG = {
    # block_type, in_channels, out_channels, stride, kernel_size, group, activation, se, kwargs
    "first": [["Conv", 3, 64, 2, 3, 1, "prelu", False, {}],
              ["Conv", 64, 64, 1, 3, 64, "prelu", False, {}]],

    # block_type, in_channels, out_channels, stride, kernel_size, group, activation, se, kwargs
    "stage": [["Mobile", 64, 64, 2, 3, 1, "prelu", False, {"expansion_rate": 2}],
              ["Mobile", 64, 64, 1, 3, 1, "prelu",
                  False, {"expansion_rate": 2}],
              ["Mobile", 64, 64, 1, 3, 1, "prelu",
                  False, {"expansion_rate": 2}],
              ["Mobile", 64, 64, 1, 3, 1, "prelu",
                  False, {"expansion_rate": 2}],
              ["Mobile", 64, 64, 1, 3, 1, "prelu",
                  False, {"expansion_rate": 2}],
              ["Mobile", 64, 128, 2, 3, 1, "prelu",
                  False, {"expansion_rate": 4}],
              ["Mobile", 128, 128, 1, 3, 1, "prelu",
                  False, {"expansion_rate": 2}],
              ["Mobile", 128, 128, 1, 3, 1, "prelu",
                  False, {"expansion_rate": 2}],
              ["Mobile", 128, 128, 1, 3, 1, "prelu",
                  False, {"expansion_rate": 2}],
              ["Mobile", 128, 128, 1, 3, 1, "prelu",
                  False, {"expansion_rate": 2}],
              ["Mobile", 128, 128, 1, 3, 1, "prelu",
                  False, {"expansion_rate": 2}],
              ["Mobile", 128, 128, 1, 3, 1, "prelu",
                  False, {"expansion_rate": 2}],
              ["Mobile", 128, 128, 2, 3, 1, "prelu",
                  False, {"expansion_rate": 4}],
              ["Mobile", 128, 128, 1, 3, 1, "prelu",
                  False, {"expansion_rate": 2}],
              ["Mobile", 128, 128, 1, 3, 1, "prelu", False, {"expansion_rate": 2}]],

    # block_type, in_channels, out_channels, stride, kernel_size, group, activation, se, kwargs
    "last": [["Conv", 128, 512, 1, 1, 1, "prelu", False, {}],
             ["Conv", 512, 512, 1, 7, 512, None, False, {}],
             ["Conv", 512, 128, 1, 1, 1, None, False, {}],
             ["global_average", 0, 0, 0, 0, 0, 0, False, {}]]
}
