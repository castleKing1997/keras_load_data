from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
import numpy as np

class ImageLoader():
    def __init__(self, prefix, list_IDs, labels, dim=(32,32), n_channels=1,
                 n_classes=10, shuffle=True, one_hot=True):
        'Initialization'
        self.dim = dim
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.prefix = prefix
        self.one_hot = one_hot
        self.on_epoch_end()
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def load_data(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        N = len(self.indexes)
        X = np.empty((N,)+(self.dim)+(self.n_channels,))
        y = np.empty((N,), dtype=int)
        list_IDs_temp = [self.list_IDs[k] for k in self.indexes]
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.prefix[-4:] == ".npy":
                X[i,] = np.load(self.prefix.format(*ID))
            elif self.prefix[-4:] == ".png":
                grayscale = True if self.n_channels == 1 else False
                X[i,] = img_to_array(load_img(self.prefix.format(*ID),target_size=self.dim+(self.n_channels,),grayscale=grayscale))
            # Store class
            y[i] = self.labels[ID]
        X = X/X.max()
        y = to_categorical(y, num_classes=self.n_classes) if self.one_hot else y
        return X, y

