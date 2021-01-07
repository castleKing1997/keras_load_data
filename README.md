# keras数据读取 


## ImageDataGenerator

参考:https://blog.csdn.net/mrr1ght/article/details/90902639

应用场景：数据以图片的形式保存在本地目录，不同类的数据在不同子目录中

keras.preprocessing.image.ImageDataGenerator()是一个类，可进行实时数据增强，生成一个batch的数据。


```python
def __init__(self,
                featurewise_center=False,#使数据集去中心化，使均值为0。统计整个数据集的均值
                samplewise_center=False, #使每个样本去中心化，只统计当前样本的均值
                featurewise_std_normalization=False,#除以数据集的标准差完成标准化，统计整个数 
                                                    #据集的标准差
                samplewise_std_normalization=False, #除以当前样本的标准差完成标准化，统计当前 
                                                    #样本的标准差
                zca_whitening=False,      #对数据进行zca白化
                zca_epsilon=1e-6,
                rotation_range=0.,
                width_shift_range=0.,
                height_shift_range=0.,
                shear_range=0.,
                zoom_range=0.,
                channel_shift_range=0.,
                fill_mode='nearest',
                cval=0.,
                horizontal_flip=False,
                vertical_flip=False,
                rescale=None,
                preprocessing_function=None,
                data_format=None):
```

## CustomGenerator

参考：https://blog.csdn.net/ordream/article/details/107182781

应用场景：数据以图片的形式保存在本地目录，每张图片由ID标识，给出不同ID对应的类别文档。

## LoadAllImage

