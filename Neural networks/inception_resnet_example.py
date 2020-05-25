def build_model(inputs):

    bn = BatchNormalization()(inputs)
    conv0 = Conv2D(128, kernel_size=1, strides=1, padding='same', activation='relu')(bn)
    conv1 = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(bn)
    concat = concatenate([conv0, conv1], axis=3)

    bn = BatchNormalization()(concat)
    conv0 = Conv2D(64, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
    conv1 = Conv2D(64, kernel_size=5, strides=1, padding='same', activation='relu')(bn)
    concat = concatenate([concat, conv0, conv1], axis=3)

    bn = BatchNormalization()(concat)
    conv0 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(bn)
    conv1 = Conv2D(32, kernel_size=7, strides=1, padding='same', activation='relu')(bn)
    concat = concatenate([concat, conv0, conv1], axis=3)

    for i in range(10):
        bn = BatchNormalization()(concat)
        conv0 = Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu')(bn)
        conv1 = Conv2D(16, kernel_size=7, strides=1, padding='same', activation='relu')(bn)
        concat = concatenate([concat, conv0, conv1], axis=3)

    bn = BatchNormalization()(concat)
    outputs = Conv2D(1, kernel_size=1, strides=1, padding='same', activation='relu')(bn)

    model = Model(inputs=inputs, outputs=outputs)

    return model

input_layer = Input((40, 40, 10))
model = build_model(input_layer)
