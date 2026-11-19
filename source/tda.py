import gtda.diagrams as gtdad
import numpy as np
import os
import joblib

def calc_landscape( diagram_path ):
    save_path = os.path.join( "landscape" , diagram_path )
    try:
        landscape = np.load( save_path )
    except:
        if not os.path.exists( os.path.dirname(save_path)):
            os.makedirs( os.path.dirname(save_path) )
        landscape = gtdad.PersistenceLandscape().fit_transform( np.load( diagram_path ) )
        np.save( save_path, landscape )
    landscape = landscape[0][:2]
    if landscape.shape != (2,100):
        print(f"landscape has incorrect shape {landscape.shape}")
    return landscape

def calc_topo_features( diagram_path ):
    feature_path = os.path.join( "features" , diagram_path )
    try:
        return np.load( feature_path )
    except:
        if not os.path.exists( os.path.dirname(feature_path)):
            os.makedirs( os.path.dirname(feature_path) )
        features = _calc_topo_features( np.load(diagram_path) )
        np.save( feature_path, features )
        return features
    
#perhaps change heat kernel metric_params
def _calc_topo_features( diagram ):
    #landscape_handler = gtdad.PersistenceLandscape(n_bins=100)

    with joblib.parallel_backend("loky"):
        amps = []
        amps.extend(gtdad.Amplitude(metric='wasserstein',metric_params={ 'p': 1 }).fit_transform(diagram)[0][:2])
        amps.extend(gtdad.Amplitude(metric='wasserstein',metric_params={ 'p': 2 }).fit_transform(diagram)[0][:2])
        amps.extend(gtdad.Amplitude(metric='bottleneck').fit_transform(diagram)[0][:2])
        amps.extend(gtdad.Amplitude(metric='betti',metric_params={ 'p': 1 }).fit_transform(diagram)[0][:2])
        amps.extend(gtdad.Amplitude(metric='betti',metric_params={ 'p': 2 }).fit_transform(diagram)[0][:2])
        amps.extend(gtdad.Amplitude(metric='landscape', metric_params={ 'p': 1, 'n_layers': 1 }).fit_transform(diagram)[0][:2])
        amps.extend(gtdad.Amplitude(metric='landscape', metric_params={ 'p': 2, 'n_layers': 1 }).fit_transform(diagram)[0][:2])
        amps.extend(gtdad.Amplitude(metric='landscape', metric_params={ 'p': 1, 'n_layers': 2 }).fit_transform(diagram)[0][:2])
        amps.extend(gtdad.Amplitude(metric='landscape', metric_params={ 'p': 2, 'n_layers': 2 }).fit_transform(diagram)[0][:2])
        amps.extend(gtdad.Amplitude(metric='heat', metric_params={ 'p': 1, 'sigma':0.1, 'n_bins':20 }).fit_transform(diagram)[0][:2])
        amps.extend(gtdad.Amplitude(metric='heat', metric_params={ 'p': 2, 'sigma':0.1, 'n_bins':20 }).fit_transform(diagram)[0][:2])

        amps.extend(gtdad.Amplitude(metric='silhouette', metric_params={ 'p': 1 }).fit_transform(diagram)[0][:2])
        amps.extend(gtdad.Amplitude(metric='silhouette', metric_params={ 'p': 2 }).fit_transform(diagram)[0][:2])
        amps.extend(gtdad.Amplitude(metric='persistence_image', metric_params={ 'p': 1, 'sigma':0.1, 'n_bins':20 }).fit_transform(diagram)[0][:2])
        amps.extend(gtdad.Amplitude(metric='persistence_image', metric_params={ 'p': 2, 'sigma':0.1, 'n_bins':20 }).fit_transform(diagram)[0][:2])

        entropy = gtdad.PersistenceEntropy().fit_transform(diagram)[0][:2]
        amps.extend(entropy)

    features = amps
    #all_features = np.concatenate( (betti_curve[0][:2], landscape[0][:2] ), axis = 0 )
    return features

#def visualize_prediction