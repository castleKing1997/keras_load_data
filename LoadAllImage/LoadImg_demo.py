from LoadAllImage import *
from Model.CustomModels import *


if __name__ == '__main__':

    import os,glob
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Parameters
    params = {'n_channels': 1,
              'shuffle': True,
              'dim': (224,224),
              'n_classes': 10,
              'one_hot': True}

    # Datasets
    # partition =  # IDs
    # labels =  # Label
    prefix = "./data/mnist/images/*/*.png" 
    paths = glob.glob(prefix)
    paths = paths[::10]
    x = {}
    x['train'] = [(path[:-4].split("/")[-2],path[:-4].split("/")[-1]) for path in paths]
    N = len(x['train'])
    y = {}
    y['train'] = {(path[:-4].split("/")[-2],path[:-4].split("/")[-1]):int(path[:-4].split("/")[-2]) for path in paths}
    prefix = prefix.replace("*","{}")
    # Generators
    image_loader = ImageLoader(prefix,x['train'],y['train'],**params)
    X,y = image_loader.load_data()
    # Design model
    model = AlexNet((params["dim"]+(params["n_channels"],)),params["n_classes"])
    # Train model on dataset
    model.fit(X,y,batch_size=128,epochs=20,validation_split=0.3)




