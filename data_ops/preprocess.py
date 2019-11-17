import os
import sys
import scipy.ndimage
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pydicom
import pydicom.pixel_data_handlers.gdcm_handler as gdcm_handler
pydicom.config.image_handlers = [None, gdcm_handler]

def normalize(image, MIN_B=-1024.0, MAX_B=3072.0):
    # https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    image = (image - MIN_B) / (MAX_B - MIN_B)
    return image

def denormalize(image, MIN_B=-1024.0, MAX_B=3072.0):
    image *= (MAX_B - MIN_B)
    image += MIN_B
    return image

def get_pixels_from_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    z = scan[0].SliceThickness
    x, y = scan[0].PixelSpacing
    spacing = np.array([z,x,y], dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image#, new_spacing

def split_data(self, directory, ratios=(.8,.1,.1)):
    '''
    directory - Directory with unstructured paired ct images.
    ratios    - List or tuple of ratios for train, val and test sets
    '''
    train_ratio, val_ratio, test_ratio = ratios
    
    dirs  = ['train', 'val', 'test']
    files = []
    for file in os.listdir(directory):
        files.append(os.path.join(directory, file))
    
    #shuffle the data
    random.shuffle(files)
    
    # define the indices of data split
    train_idx = int( len(files) * train_ratio )  
    val_idx   = int( len(files) * val_ratio + train_idx )
    
    # split the data
    data_split = {
        'train': files[:train_idx],
        'val':   files[train_idx:val_idx],
        'test':  files[val_idx:]
    }
    # move each image to the set folder to which it belongs (train, val, test)
    for split_dir in dirs:
        name       = split_dir
        split_dir  = os.path.join(directory, split_dir)
        if not os.path.isdir(split_dir):
            os.mkdir(split_dir)
        for img in data_split[name]:  # e.g. data_split['train']
            shutil.move(img, split_dir)


class DataPreparation:
    def __init__(self, dicoms_dir, destination_dir, labels_csv ,HU_normalization=True):
        self.dicoms_dir        = dicoms_dir
        self.destination_dir   = destination_dir
        self.labels_csv = pd.read_csv(labels_csv)
        self.HU_normalization  = HU_normalization
        # default values
        self.min_HU = -1024.0
        self.max_HU = 3072.0

    def process(self):
        if self.HU_normalization: 
            self.find_min_max_HU()
        self.do()

    def do(self):
        if not os.path.isdir(self.destination_dir):
            os.mkdir(self.destination_dir)
        # walk through directories and subdirectories
        for dir_name, subdir_list, file_list in os.walk(self.dicoms_dir):
            # read all dicom files
            volume = [pydicom.read_file(os.path.join(dir_name, file)) for file in file_list if '.dcm' in file.lower()]
            # sort according to slice position
            volume = self._sort_slices(volume)
            # check if slice contains anything
            if volume:
                label = self.get_label(volume[0].PatientID)
                npy_volume = self.dicom_to_npy_volume(volume)
                save_path = os.path.join(self.destination_dir, 
                                         '{}_{}'.format(volume[0].PatientID, 
                                                        volume[0].SeriesInstanceUID))
                np.save(save_path, np.array([npy_volume, label]))

    def _sort_slices(self, slices):
        # sort the slices of a series in the same order in which they were scanned
        try:
            slices.sort(key=lambda x: x.InstanceNumber)
        # series that don't contain scans are deleted
        except AttributeError:
            print('--> Exception: The dicom series under name', slices[0].SeriesInstanceUID, 'has been \
            removed from the dataset for not having any scans in it. It might be the mask file.')
            return
        return slices

    def find_min_max_HU(self):
        self.min_HU = sys.maxsize
        self.max_HU = 1 - sys.maxsize
        for dir_name, subdir_list, file_list in os.walk(self.dicoms_dir):
            # read all dicom files
            volume = [pydicom.read_file(os.path.join(dir_name, file)) for file in file_list if '.dcm' in file.lower()]
            # sort according to slice position
            volume = self._sort_slices(volume)
            # check if volume contains anything
            if volume:
                min_here = get_pixels_from_hu(volume).min()
                max_here = get_pixels_from_hu(volume).max()
                if min_here < self.min_HU:
                    self.min_HU = min_here
                if max_here > self.max_HU:
                    self.max_HU = max_here
        print('Min_HU:', self.min_HU, 'Max_HU:', self.max_HU)
    
    def dicom_to_npy_volume(self, dicoms):
        image     = get_pixels_from_hu(dicoms)
        resampled = resample(image, dicoms)
        return normalize(resampled)
    
    def get_label(self, patientID):
        row = self.labels_csv.loc[self.labels_csv['name'] == patientID]
        #return ICH, Fracture, MassEfect, leave out the name
        return row.values[0][1:]

if __name__ == "__main__":
    dicoms_dir = '../dataset/scans/'
    destination_dir = '../dataset/processed/'
    labels_csv = '../dataset/simplified_labels.csv'
    HU_normalization = True
    dataset = DataPreparation(dicoms_dir, destination_dir, labels_csv, HU_normalization)
    #dataset.min_HU = -1200
    #dataset.max_HU = 6000
    dataset.process()