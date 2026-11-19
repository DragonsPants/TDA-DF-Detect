import torchvision.transforms as transforms
from source.image_handler import ImageHandler
from torch.utils.data import DataLoader
import numpy as np
import glob
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from source.tda import calc_topo_features, calc_landscape
import torch
import random
from joblib import Parallel, delayed
import gtda.diagrams as gtdad

def get_video_paths( directory_name : str , num_paths=-1 ):
    video_paths = sorted(glob.glob(os.path.join(directory_name, "*.mp4")))
    if num_paths > 0:
        video_paths = video_paths[:num_paths]
    return video_paths

def split_dataset( data :list, indices : np.ndarray ):
    train_data = []
    test_data = []
    data.sort()
    for i in range(len(data)):
        if i in indices:
            train_data.append(data[i])
        else:
            test_data.append(data[i])
    return train_data, test_data

IMG_SIZE = 299
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_transform = transforms.Compose( [ transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(mean=mean,std=std)] )
filtrations = [ "dilate", "erode", "fft", "gradient", "gray", "lbp" ]

class ImageDataSet(Dataset):
    def num_topo_features(self):
        return len( self.topo_features[0][0] )

    def process_image(self, ih:ImageHandler):
        topo = []
        for filter in filtrations:
            topo.extend(calc_topo_features( ih.get_ph_diagram_path( filter ) ) )
        return [topo] 

    def customNormalize( self, input:torch.Tensor, mean:torch.Tensor, std:torch.Tensor) -> torch.Tensor:
        output = torch.zeros(input.shape)
        for ch_id in range(input.shape[0]):
                output[ch_id] = (input[ch_id] - mean[ch_id]) / std[ch_id]
        return output

    def __init__(self, real_paths, fake_paths, dataset_name, transform = None):

        self.num_real_images = len(real_paths)
        self.video_frames = []
        self.topo_features = []
        self.lbp_landscapes = []

        for path in tqdm(real_paths):
            ih = ImageHandler( path, is_real=True, dataset_name = dataset_name)
            self.video_frames.append( transform( ih.get_PIL_image() ) )
            self.topo_features.append( self.process_image(ih) )
            self.lbp_landscapes.append( calc_landscape( ih.get_ph_diagram_path( "lbp" ) ) )

        for path in tqdm(fake_paths):
            ih = ImageHandler( path, is_real=False, dataset_name = dataset_name)
            self.video_frames.append( transform( ih.get_PIL_image() ) )
            self.topo_features.append( self.process_image(ih) )
            self.lbp_landscapes.append( calc_landscape( ih.get_ph_diagram_path( "lbp" ) ) )
        #for path in tqdm(fake_paths):
        #    self.process_image( ImageHandler( path, is_real=False, dataset_name = dataset_name) )

        self.topo_features = torch.tensor( np.array(self.topo_features) ).float()    
        topo_mean = torch.mean( self.topo_features, dim=[0,2] )
        topo_std = torch.std( self.topo_features, dim=[0,2] )
        for t_id in range(self.topo_features.shape[0]):
            self.topo_features[t_id] = self.customNormalize( self.topo_features[t_id], topo_mean, topo_std )

        self.lbp_landscapes = torch.tensor( np.array(self.lbp_landscapes) ).float()
        land_mean = torch.mean( self.lbp_landscapes, dim=[0,2] )
        land_std = torch.std( self.lbp_landscapes, dim=[0,2] )
        for ls_id in range(self.lbp_landscapes.shape[0]):
            self.lbp_landscapes[ls_id] = self.customNormalize( self.lbp_landscapes[ls_id], land_mean, land_std )


    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, index):
        frame = self.video_frames[index]
        label = torch.tensor( 0 if index < self.num_real_images else 1 )
        #grad_image = self.grad_images[index]
        topo_vector = self.topo_features[index]
        landscape = self.lbp_landscapes[index]

        return frame, label, topo_vector, landscape

def get_dataloaders(source_dir:dict):
    FTR_RATIO = 3
    real_video_paths = get_video_paths( source_dir["real_dir"] )
    fake_video_paths = get_video_paths( source_dir["fake_dir"] )
    ri_path = os.path.join("SampleIndices",f"real_indices_{FTR_RATIO}.csv")
    fi_path = os.path.join("SampleIndices",f"fake_indices_{FTR_RATIO}.csv")

    print(len(real_video_paths))
    print(len(fake_video_paths))

    real_indices = np.loadtxt( ri_path, delimiter="," )
    #real_indices = np.asarray( random.sample( range( 0, len( real_video_paths ) ), int(len(real_video_paths) ) ) )
    #np.save( ri_path, real_indices )

    fake_indices = np.loadtxt( fi_path, delimiter="," )
    #fake_indices = np.asarray( random.sample( range( 0, len( fake_video_paths ) ), int(len(fake_video_paths) * FTR_RATIO ) ) )
    #np.save( fi_path, fake_indices )

    #load data
    real_train, real_test = split_dataset( real_video_paths , real_indices ) 
    fake_train, fake_test = split_dataset( fake_video_paths , fake_indices )

    fake_train = fake_train[:len(real_train) * FTR_RATIO]
    fake_test = fake_test[:len(real_test) * FTR_RATIO]

    train_dataset = ImageDataSet( real_train, fake_train, source_dir["dataset_name"], img_transform )
    train_dataloader = DataLoader( train_dataset, batch_size=32, shuffle=True)

    test_dataset = ImageDataSet( real_test, fake_test, source_dir["dataset_name"], img_transform )
    test_dataloader = DataLoader( test_dataset, batch_size=32, shuffle= False )

    return train_dataloader, test_dataloader