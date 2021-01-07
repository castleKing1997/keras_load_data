from keras.models import Model
from keras.layers import Dense, Flatten, Input


def FCnet(input_shape,n_class):
    ipt = Input(shape=(input_shape),name="input")
    layer = Flatten(name="flatten")(ipt)
    layer = Dense(128,activation=None,name="fc1")(layer)
    layer = Dense(n_class,activation="softmax",name="output")(layer)
    model = Model(ipt,layer)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

