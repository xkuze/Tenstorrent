def UNetVGG19(config, x, *, parameters):
    x = Sequential(config, input=x, parameters=parameters.enc1)    # shapes: (1, 64, 256, 256), dtypes: torch.float32; duration: 19.3 ms
    variable_0 = MaxPool2d(config, input=x, parameters=parameters.pool1)    # shapes: (1, 64, 128, 128), dtypes: torch.float32; duration: 3.6 ms
    variable_0 = Sequential(config, input=variable_0, parameters=parameters.enc2)    # shapes: (1, 128, 128, 128), dtypes: torch.float32; duration: 14.4 ms
    variable_1 = MaxPool2d(config, input=variable_0, parameters=parameters.pool2)    # shapes: (1, 128, 64, 64), dtypes: torch.float32; duration: 1.7 ms
    variable_1 = Sequential(config, input=variable_1, parameters=parameters.enc3)    # shapes: (1, 256, 64, 64), dtypes: torch.float32; duration: 25.5 ms
    variable_2 = MaxPool2d(config, input=variable_1, parameters=parameters.pool3)    # shapes: (1, 256, 32, 32), dtypes: torch.float32; duration: 1.0 ms
    variable_2 = Sequential(config, input=variable_2, parameters=parameters.enc4)    # shapes: (1, 512, 32, 32), dtypes: torch.float32; duration: 26.1 ms
    variable_3 = MaxPool2d(config, input=variable_2, parameters=parameters.pool4)    # shapes: (1, 512, 16, 16), dtypes: torch.float32; duration: 540.0 µs
    variable_3 = Sequential(config, input=variable_3, parameters=parameters.enc5)    # shapes: (1, 512, 16, 16), dtypes: torch.float32; duration: 15.5 ms
    variable_4 = MaxPool2d(config, input=variable_3, parameters=parameters.pool4)    # shapes: (1, 512, 8, 8), dtypes: torch.float32; duration: 301.1 µs
    variable_4 = DoubleConv(config, x=variable_4, parameters=parameters.bridge)    # shapes: (1, 1024, 8, 8), dtypes: torch.float32; duration: 20.4 ms
    variable_5 = UpBlock(config, x=variable_4, skip=variable_3, parameters=parameters.up5)    # shapes: (1, 512, 16, 16), dtypes: torch.float32; duration: 17.6 ms
    variable_6 = UpBlock(config, x=variable_5, skip=variable_2, parameters=parameters.up4)    # shapes: (1, 256, 32, 32), dtypes: torch.float32; duration: 17.9 ms
    variable_7 = UpBlock(config, x=variable_6, skip=variable_1, parameters=parameters.up3)    # shapes: (1, 128, 64, 64), dtypes: torch.float32; duration: 19.0 ms
    variable_8 = UpBlock(config, x=variable_7, skip=variable_0, parameters=parameters.up2)    # shapes: (1, 64, 128, 128), dtypes: torch.float32; duration: 23.7 ms
    variable_9 = UpBlock(config, x=variable_8, skip=x, parameters=parameters.up1)    # shapes: (1, 64, 256, 256), dtypes: torch.float32; duration: 53.1 ms
    variable_10 = Conv2d(config, input=variable_9, parameters=parameters.final)    # shapes: (1, 1, 256, 256), dtypes: torch.float32; duration: 3.7 ms
    return variable_10


def top_level_module():
    variable_0 = torch.randn(1, 3, 256, 256)    # shapes: (1, 3, 256, 256), dtypes: torch.float32; duration: 1.4 ms
    variable_1 = torch.as_tensor([-0.6618398427963257, 0.9280802011489868, -1.1047552824020386, 0.9945057034492493, 1.2822483777999878, 0.07357272505760193, -0.3302072286605835, -1.7969377040863037], ...).reshape((1, 3, 256, 256)).to(torch.float32)    # shapes: (1, 3, 256, 256), dtypes: torch.float32
    variable_2 = UNetVGG19(config, x=variable_1, parameters=parameters)    # shapes: (1, 1, 256, 256), dtypes: torch.float32; duration: 277.5 ms
