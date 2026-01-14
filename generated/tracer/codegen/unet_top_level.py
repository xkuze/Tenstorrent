def top_level_module():
    variable_0 = torch.randn(
        1, 3, 256, 256
    )  # shapes: (1, 3, 256, 256), dtypes: torch.float32; duration: 1.2 ms
    variable_1 = (
        torch.as_tensor(
            [
                -0.5958576202392578,
                0.947610080242157,
                -2.073986768722534,
                -1.0039725303649902,
                0.3877097964286804,
                -1.715579867362976,
                -0.27635255455970764,
                0.03884531185030937,
            ],
            ...,
        )
        .reshape((1, 3, 256, 256))
        .to(torch.float32)
    )  # shapes: (1, 3, 256, 256), dtypes: torch.float32
    variable_2 = UNetVGG19(
        config, x=variable_1, parameters=parameters
    )  # shapes: (1, 1, 256, 256), dtypes: torch.float32; duration: 211.7 ms
