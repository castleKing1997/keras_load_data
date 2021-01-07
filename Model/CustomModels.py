from keras.models import Model
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPool2D, ZeroPadding2D, Dropout, Activation, BatchNormalization
from keras.regularizers import l2

def FCnet(input_shape,n_class):
    ipt = Input(shape=(input_shape),name="input")
    layer = Flatten(name="flatten")(ipt)
    layer = Dense(128,activation=None,name="fc1")(layer)
    layer = Dense(n_class,activation="softmax",name="output")(layer)
    model = Model(ipt,layer)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def AlexNet(input_shape,num_classes,l2_reg=0.0, weights=None):
    ipt = Input(shape=input_shape,name="input")
    layer = ZeroPadding2D(padding=(2, 2))(ipt)
    layer = conv_block(layer, filters=96, kernel_size=(11, 11),
                   strides=(4, 4), padding="valid", l2_reg=l2_reg, name='Conv_1_96_11layer11_4')
    layer = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_1_3x3_2")(layer)

    # Layer 2
    layer = conv_block(layer, filters=256, kernel_size=(5, 5),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_2_256_5layer5_1")
    layer = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_2_3x3_2")(layer)

    # Layer 3
    layer = conv_block(layer, filters=384, kernel_size=(3, 3),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_3_384_3x3_1")

    # Layer 4
    layer = conv_block(layer, filters=384, kernel_size=(3, 3),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_4_384_3x3_1")

    # Layer 5
    layer = conv_block(layer, filters=256, kernel_size=(3, 3),
                   strides=(1, 1), padding="same", l2_reg=l2_reg, name="Conv_5_256_3x3_1")
    layer = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid", name="maxpool_3_3x3_2")(layer)

    # Layer 6
    layer = Flatten()(layer)
    layer = Dense(units=4096)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    # Layer 7
    layer = Dense(units=4096)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    # Layer 8
    layer = Dense(units=num_classes)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation("softmax")(layer)

    if weights is not None:
        layer.load_weights(weights)
    model = Model(ipt, layer, name="AlelayerNet")
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def conv_block(layer, filters, kernel_size=(3, 3), strides=(1, 1), padding='valid', l2_reg=0.0, name=None):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               kernel_regularizer=l2(l2_reg),
               kernel_initializer="he_normal",
               name=name)(layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

