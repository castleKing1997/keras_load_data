from keras.preprocessing.image import ImageDataGenerator
from Model.CustomModels import *


data_dir = "./data/mnist/images"

imagegen = ImageDataGenerator(rescale=1/255.0)
imagegen = imagegen.flow_from_directory(data_dir,target_size=(28,28),color_mode="grayscale",batch_size=500)

model = FCnet((28,28,1),10)
# Train model on dataset
model.fit_generator(generator=imagegen,epochs=10)

