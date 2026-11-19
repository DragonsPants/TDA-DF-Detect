from multiprocessing.managers import SharedMemoryManager

import torch;
from source.image_dataset import ImageDataSet, get_dataloaders
from source.tda import calc_topo_features
from joblib import parallel_backend
import os
from source.models import train_model, test_model
from source.models import TopoHead, xNetHead
import numpy as np
from math import inf

celeb_df = {
    "real_dir": os.path.join("Celeb-DF-v2","Celeb-real"),
    "fake_dir": os.path.join("Celeb-DF-v2","Celeb-synthesis"),
    "dataset_name": "Celeb-DF-v2",
}

def main():
    source_dir = celeb_df

    train_loader, test_loader = get_dataloaders(source_dir)

    device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
    print(device)
    #model = xNetHead()
    model = TopoHead( num_topo_features = train_loader.dataset.num_topo_features())
    model.to(device)
    model = train_model(model, train_loader, device, num_epochs = 500)
    test_model(model, test_loader, device)

main()


#add grayscale branch piece
#add landscape of gradient???

# exploding gradient
    # test for gradient
    # weight initialization
    # input values not normalized
    # no activation function
    # learning rate to high?
    # residual connections?