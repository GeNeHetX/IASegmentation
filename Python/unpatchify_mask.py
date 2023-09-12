import numpy as np
import cv2, os
import shutil
from tqdm import tqdm
import rasterio.features
from rasterio.transform import Affine
import numpy as np
from scipy import ndimage
import json
import cython_extension/unpatchify_mask as um

RESULT_DIR="result"
DATA_DIR=["./", "./"]
down_sample=2.0
size=512
threshold_proba = 127

def labels_to_features(lab: np.ndarray, object_type='annotation', connectivity: int=4, 
                      transform: Affine=None, mask=None, downsample: float=2.0, include_labels=False,
                      classification=None):
    """
    Create a GeoJSON FeatureCollection from a labeled image.
    """
    features = []
    print(lab.dtype)
    # Ensure types are valid
    if lab.dtype == bool:
        mask = lab
        lab = lab.astype(rasterio.int16)
    else:
        mask = (lab > 0).astype(rasterio.uint8)
    lab.astype(rasterio.int16)
    # Create transform from downsample if needed
    if transform is None:
        transform = Affine.scale(downsample)
    
    # Trace geometries
    for s in rasterio.features.shapes(lab.astype(rasterio.int16), mask=mask, 
                                      connectivity=connectivity, transform=transform):

        # Create properties
        props = dict(object_type=object_type)
        if include_labels:
            props['measurements'] = [{'name': 'Label', 'value': s[1]}]
            
        # Just to show how a classification can be added
        if classification is not None:
            props['classification'] = classification
        
        # Wrap in a dict to effectively create a GeoJSON Feature
        po = dict(type="Feature", geometry=s[0], properties=props)

        features.append(po)
    
    return features

for data_dir in DATA_DIR:

    #to move image in folder for each lame
    for img in tqdm(os.listdir(data_dir)):
        img_dir=os.path.join(data_dir, img.split(" ")[0])
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        shutil.move(os.path.join(data_dir, img), img_dir)


    print(data_dir)
    result_dir=os.path.join(data_dir, RESULT_DIR)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    for dir in os.listdir(data_dir):
        end_img=0
        image_binarize=0
        if dir!="result" and dir!=RESULT_DIR:
            folder_dir=os.path.join(data_dir, dir)
            down_sample=float(os.listdir(folder_dir)[0].split("=")[1][:-2])
            print(dir)
            print(down_sample)
            list_name=os.listdir(folder_dir)
            end_img=um.process_images(folder_dir + "/", list_name, down_sample, size)
            
            print("End_ img recu pret pour export")
            end_img=np.array(end_img)
            print(end_img.shape)

            cv2.imwrite(os.path.join(result_dir, dir+".png"), end_img)

            image_binarize=np.where(end_img>=threshold_proba, False,True)
            
            # Create a labeled image (or just use a binary one)
            # Here, bw is the numpy bool array
            lab, n = ndimage.label(image_binarize)

            # Create GeoJSON-like version
            print("export geojson")
            features = labels_to_features(lab, downsample=down_sample, object_type='annotation', classification="Stroma")
            with open(os.path.join(result_dir, dir+".geojson"), "w") as f:
                json.dump(features, f)
