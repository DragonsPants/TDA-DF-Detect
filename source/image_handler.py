
import os
import numpy as np
from PIL import Image
import cv2
import gtda.homology.cubical as cubical
import glob

#REMEMBER -- handle how gradient works

#possible filtrations
#dilate, erode, gabor, local binary pattern, frequency domain, intensity?

class ImageHandler(object):
    def __init__(self, video_path, is_real, dataset_name = ''):
        self.image = self.path_to_image(video_path)
        self.name = os.path.splitext(os.path.basename( video_path ))[0]
        self.dataset_name = dataset_name
        self.is_real = is_real

    def path_to_image(self, video_path):
        capture = cv2.VideoCapture( video_path )
        if capture.isOpened():
            fIndex, fImage = capture.read()
            capture.release()
            return fImage
        else:
            print("Capture could not be opened, path:" + video_path)
            capture.release()
            return None

    def get_PIL_image(self):
        return Image.fromarray(self.image)

    def get_grad_PIL( self, grayscale = False ):
        grad_image = self.get_grad_image()
        if grayscale:
            return Image.fromarray(grad_image)
        grad_image = cv2.cvtColor(grad_image, cv2.COLOR_GRAY2BGR)
        return Image.fromarray(grad_image)

    def get_grad_image( self ):
        out_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        out_image = cv2.Laplacian( out_image, cv2.CV_16S , ksize=3 )
        out_image = cv2.convertScaleAbs(out_image)

        return out_image

    def imshow( self ):
        while True:
            cv2.imshow(self.name, self.image)
            if cv2.waitKey(17) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    def get_ph_diagram_path(self, filter = 'gray'):
        phd_path = os.path.join( "Homology", self.dataset_name, "real" if self.is_real else "fake" )
        if filter is not None:
            phd_path = os.path.join(phd_path, filter)
        phd_path = os.path.join( phd_path, self.name ) + ".npy"

        if not os.path.exists( phd_path ):
            self._calc_ph_diagram( output_path = phd_path, filter = filter )
        return phd_path

    def _calc_ph_diagram(self, output_path, filter = 'gray' ):
        #ph_input = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #if filter is not None:
        ph_input = np.load( os.path.join( "..","image-filtration", "Real" if self.is_real else "Fake", filter, self.name) + ".npy" )

        cc = cubical.CubicalPersistence(homology_dimensions=[0, 1],n_jobs=-1)
        phDiagram = cc.fit_transform([ph_input])
        np.save( output_path, phDiagram )
        return phDiagram

    #dilate, erode, gabor, local binary pattern, frequency domain