def UNetVGG19(config, x, *, parameters):
    x = Sequential(
        config, input=x, parameters=parameters.enc1
    )  # shapes: (1, 64, 256, 256), dtypes: torch.float32; duration: 14.3 ms
    variable_0 = MaxPool2d(
        config, input=x, parameters=parameters.pool1
    )  # shapes: (1, 64, 128, 128), dtypes: torch.float32; duration: 3.3 ms
    variable_0 = Sequential(
        config, input=variable_0, parameters=parameters.enc2
    )  # shapes: (1, 128, 128, 128), dtypes: torch.float32; duration: 12.3 ms
    variable_1 = MaxPool2d(
        config, input=variable_0, parameters=parameters.pool2
    )  # shapes: (1, 128, 64, 64), dtypes: torch.float32; duration: 1.6 ms
    variable_1 = Sequential(
        config, input=variable_1, parameters=parameters.enc3
    )  # shapes: (1, 256, 64, 64), dtypes: torch.float32; duration: 23.3 ms
    variable_2 = MaxPool2d(
        config, input=variable_1, parameters=parameters.pool3
    )  # shapes: (1, 256, 32, 32), dtypes: torch.float32; duration: 709.8 µs
    variable_2 = Sequential(
        config, input=variable_2, parameters=parameters.enc4
    )  # shapes: (1, 512, 32, 32), dtypes: torch.float32; duration: 21.8 ms
    variable_3 = MaxPool2d(
        config, input=variable_2, parameters=parameters.pool4
    )  # shapes: (1, 512, 16, 16), dtypes: torch.float32; duration: 456.1 µs
    variable_3 = Sequential(
        config, input=variable_3, parameters=parameters.enc5
    )  # shapes: (1, 512, 16, 16), dtypes: torch.float32; duration: 12.9 ms
    variable_4 = MaxPool2d(
        config, input=variable_3, parameters=parameters.pool4
    )  # shapes: (1, 512, 8, 8), dtypes: torch.float32; duration: 245.6 µs
    variable_4 = DoubleConv(
        config, x=variable_4, parameters=parameters.bridge
    )  # shapes: (1, 1024, 8, 8), dtypes: torch.float32; duration: 14.3 ms
    variable_5 = UpBlock(
        config, x=variable_4, skip=variable_3, parameters=parameters.up5
    )  # shapes: (1, 512, 16, 16), dtypes: torch.float32; duration: 14.5 ms
    variable_6 = UpBlock(
        config, x=variable_5, skip=variable_2, parameters=parameters.up4
    )  # shapes: (1, 256, 32, 32), dtypes: torch.float32; duration: 14.2 ms
    variable_7 = UpBlock(
        config, x=variable_6, skip=variable_1, parameters=parameters.up3
    )  # shapes: (1, 128, 64, 64), dtypes: torch.float32; duration: 14.8 ms
    variable_8 = UpBlock(
        config, x=variable_7, skip=variable_0, parameters=parameters.up2
    )  # shapes: (1, 64, 128, 128), dtypes: torch.float32; duration: 19.5 ms
    variable_9 = UpBlock(
        config, x=variable_8, skip=x, parameters=parameters.up1
    )  # shapes: (1, 64, 256, 256), dtypes: torch.float32; duration: 38.6 ms
    variable_10 = Conv2d(
        config, input=variable_9, parameters=parameters.final
    )  # shapes: (1, 1, 256, 256), dtypes: torch.float32; duration: 2.3 ms
    return variable_10
