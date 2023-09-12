import numpy as np
import cv2, os
import rasterio.features
from rasterio.transform import Affine
from scipy import ndimage
import json
import cython_extension/unpatchify_mask as um
from tqdm import tqdm
import torch
import albumentations as album
import segmentation_models_pytorch as smp
import warnings
warnings.filterwarnings("ignore")
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(r'C:/openslide-win64/bin'):
        import openslide as ops
else:
    import openslide as ops

ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'

DATA_DIR="C:/Users/Hajar/Documents/multicentrique_select/"
TILES_DIR="C:/Users/Hajar/Documents/tiles_select_256/"
MODEL_DIR=""
RESULT_DIR=""
IMAGE_NAME=""

if not os.path.exists(TILES_DIR):
    os.makedirs(TILES_DIR)

size=(512, 512)
down_sample=4
overlap=256
threshold_tissue=220
part_tissue_min=0.2
threshold_prediction=0


def test_tissue(image, threshold=0, size=(512, 512), part_tissue_min=0):
    """
    Test if the tile contain enough tissue (return bool)

    Args:
        image (nparray): image need to test
        threshold (int): maximal intensity to declare a pixel as a tissue
        size (tuple): image size of the tile
        part_tissue_min (float) : part of the minimal tissue proportion to keep the tile

    """
    image=np.asarray(image)
    image_transform=np.sum(image, axis=2)
    image_binarize=np.where((image_transform/3)<=threshold, 1,0)
    sum_tissue=np.sum(image_binarize)
    ammount_tissue=sum_tissue/(size[0]*size[1])
    # print(sum_tissue)
    if ammount_tissue>=part_tissue_min:
        return True
    
    # print(np.asarray(image))
    return False

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

def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)

def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

#keep to have image_dimension
def export_prediction_tiles(image_name, overlap, size, down_sample, threshold_tissue, part_tissue_min, best_model, preprocessing_fn):
    """
    Export tiles for the image and do preditions. returning image_dimension
    Args:
        images_name (str): name of the image file
        overlap (int): part of the image overlapping
        size (tuple): size of the tiles format (x,y)
        down_sample (int): reduction size of the image
        threshold_tissue (int): maximal intensity to declare a pixel as a tissue (in gray scale image)
        part_tissue_min (float) : part of minimal tissue proportion to keep the tile
        best_model (object): The model load to prediction
        Preprocessing_fn (object) : The SMP normalisation for dataset and pretrained model

    """
    print(image_name)
    lame_name=os.path.splitext(image_name)[0]
    wsi_image=ops.OpenSlide(os.path.join(DATA_DIR ,image_name))

    x_wsi, y_wsi=wsi_image.dimensions

    size_x_tiles_export= (size[0]-overlap)*down_sample
    size_y_tiles_export= (size[1]-overlap)*down_sample

    size_region=(size[0]*down_sample, size[1]*down_sample)


    for x in tqdm(range(0, x_wsi, size_x_tiles_export)):

        for y in (range(0, y_wsi, size_y_tiles_export)):
            image=wsi_image.read_region((x, y), 0, size_region)
            image=image.resize(size)
            if test_tissue(image, threshold_tissue, size, part_tissue_min):

                name= lame_name + " [d=" + str(down_sample) + ",x=" + str(x) + ",y=" + str(y) + ",w=" + str(size[0]*down_sample) + ",h=" + str(size[1]*down_sample) + "].png"

                path_name=os.path.join(TILES_DIR, name)

                preprocessing=get_preprocessing(preprocessing_fn=preprocessing_fn)
                x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            
                # Predict test image
                image = np.array(image)
                image=preprocessing(image=image)
                pred_mask = best_model(x_tensor)
                pred_mask = pred_mask.detach().squeeze().cpu().numpy()
                
                # Do the pred mask image
                tumor_proba_mask=pred_mask[1]
                tumor_proba_mask_img=(tumor_proba_mask*255).astype(int)

                cv2.imwrite(path_name, tumor_proba_mask_img)
    return wsi_image.dimensions

def prediction_loop(tiles_dir, result_tiles_dir, best_model, preprocessing_fn):
    """
    Doing the prediction in grayscale tiles to show probability of segmentation
    Args:
        tiles_dir (str): The directory of the tiles
        result_tiles_dir (str):The directory to put the result tiles with probability.
        best_model (object): The model load to prediction
        Preprocessing_fn (object) : The SMP normalisation for dataset and pretrained model

    """

    for img in tqdm(os.listdir(tiles_dir)):
        preprocessing=get_preprocessing(preprocessing_fn=preprocessing_fn)
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
       
        # Predict test image
        image = cv2.cvtColor(cv2.imread(os.path.join(tiles_dir, img)), cv2.COLOR_BGR2RGB)
        image=preprocessing(image=image)
        pred_mask = best_model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        
        # Do the pred mask image
        tumor_proba_mask=pred_mask[1]
        tumor_proba_mask_img=(tumor_proba_mask*255).astype(int)

        cv2.imwrite(os.path.join(result_tiles_dir, img+".png"), tumor_proba_mask_img)

def unpatchify_mask(result_prediction_dir, result_dir, threshold_prediction, down_sample, size, x_max, y_max):
    """
    Merge all tiles into a png and a geajson with annotations.
    Args:
        result_prediction_dir (str): The directory of the tiles predicted in gayscale
        result_dir (str):The directory to put the geojson and png
        threshold_prediction (int): The minimum of intensity to predict tumor in geojson
        down_sample (int) : the down sample to reconstruct geojson
        size (tuple): size of the prediction tiles
        x_max (int) : x size of the wsi
        y_max (int) : y size of the WSI
    """
    end_img=0
    image_binarize=0
    list_name=os.listdir(result_prediction_dir)
    end_img=um.process_images(result_prediction_dir + "/", list_name, down_sample, size[0], x_max, y_max)
    
    print("End_img recu pret pour export")
    end_img=np.array(end_img)
    print(end_img.shape)

    cv2.imwrite(os.path.join(result_dir, dir+".png"), end_img)

    image_binarize=np.where(end_img>=threshold_prediction, False, True)
    # Create a labeled image (or just use a binary one)
    # Here, bw is the numpy bool array
    lab, _ = ndimage.label(image_binarize)

    # Create GeoJSON-like version
    print("export geojson")
    features = labels_to_features(lab, downsample=down_sample, object_type='annotation', classification="Stroma")
    with open(os.path.join(result_dir, dir+".geojson"), "w") as f:
        json.dump(features, f)
    # print(x_max, y_max)

def main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(MODEL_DIR+'best_model.pth'):
            best_model = torch.load(MODEL_DIR+'best_model.pth', map_location=DEVICE)
            print('Loaded DeepLabV3+ model from a previous commit.')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


    print("export et prédictions")
    x_max, y_max=export_prediction_tiles(IMAGE_NAME, overlap, size, down_sample, threshold_tissue, part_tissue_min, best_model, preprocessing_fn)

    print("fusion")
    unpatchify_mask(TILES_DIR, RESULT_DIR, threshold_prediction, down_sample, size, x_max, y_max)

    print("Done")

if __name__ == '__main__':
    main()
