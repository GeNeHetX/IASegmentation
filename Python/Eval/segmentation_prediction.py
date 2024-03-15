import os, cv2
import numpy as np
import random, tqdm
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import tifffile as tf
from tqdm import tqdm
#%matplotlib inline
import torch
import torch.nn as nn
import albumentations as album
import segmentation_models_pytorch as smp
import warnings
warnings.filterwarnings("ignore")

# Indicate the path where your files are located
DATA_DIR = "/"
RESULT_DIR="/result_adj_ttneo_25NM_cree_epoch10_10/"

x_test_dir = os.path.join(DATA_DIR+'Train_adj_ttneo_25NM_10/', 'image')

class_name = ['Background', 'Tumor']
rgb_values = [[255, 255, 255], [200, 0, 0]]
select_class_name = ['Background', 'Tumor']

# helper function for data visualization
def visualize(namefile, **images):
    # """
    # Plot images in one row
    # """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.savefig(namefile)
    plt.close('all')
    # plt.show()

# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map
    
# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
   
    x = np.argmax(image, axis = -1)

    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    

    return x

class BuildingsDataset(torch.utils.data.Dataset):

    """Massachusetts Buildings Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
        self, 
        images_dir, 
        augmentation=None,
        class_rgb_values=None,  
        preprocessing=None,
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.augmentation = augmentation
        self.class_rgb_values = class_rgb_values
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read images and masks
        
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
            # I have changed  channel color image in order to have goor color, must to replace their with other traind model
        # image=cv2.imread(self.image_paths[i])
        # mask=cv2.imread(self.mask_paths[i])

        # one-hot-encode the mask
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image= sample['image']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image= sample['image']
            
        return image
        

    def __len__(self):
        # return length of 
        return len(self.image_paths)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

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

# Center crop padded image / mask to original image dims
def crop_image(image, target_image_dims=[512,512,3]):
    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,
    ]

def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)

def main():
    i=0
    #Pour chaque lame test
    for lame_test in os.listdir(DATA_DIR+'Test_overlay/'):
            i=i+1
            if i >0:
                x_test_dir = os.path.join(DATA_DIR+'Test_overlay/'+lame_test+'/', 'image')
                ENCODER = 'resnet18'
                ENCODER_WEIGHTS = 'imagenet'
                predict_dir=os.path.join(RESULT_DIR, lame_test )

                preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

                # Set device: `cuda` or `cpu`
                DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Récupere modèle entrainé
                if os.path.exists(RESULT_DIR+'best_model_cree_final.pth'):
                    best_model = torch.load(RESULT_DIR+'best_model_cree_final.pth', map_location=DEVICE)
                    print('Loaded DeepLabV3+ model from a previous commit.')

                # create test dataloader (with preprocessing operation: to_tensor(...))
                predict_dataset = BuildingsDataset(
                    x_test_dir,  
                    augmentation=get_validation_augmentation(), 
                    preprocessing=get_preprocessing(preprocessing_fn),
                    class_rgb_values=rgb_values,
                )

                # test dataset for visualization (without preprocessing transformations)
                predict_dataset_vis = BuildingsDataset(
                    x_test_dir, 
                    augmentation=get_validation_augmentation(),
                    class_rgb_values=rgb_values,
                )

                # get a random test image/mask index
                random_idx = random.randint(0, len(predict_dataset_vis)-1)
                image= predict_dataset_vis[random_idx]

                #Dossier où enregistrer les predictions
                heat_pred_folder= predict_dir+'/tile_proba'

                print(heat_pred_folder)

                if not os.path.exists(heat_pred_folder):
                    os.makedirs(heat_pred_folder)
                
                list_name=[os.path.splitext(filename)[0] for filename in sorted(os.listdir(x_test_dir))]

                # Pour chaque tuile
                for idx in tqdm(range(len(predict_dataset))):
                    image= predict_dataset[idx]
                    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
                    # Predict test image
                    pred_mask = best_model(x_tensor)
                    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
                    tumor_proba_mask=pred_mask[1]
                    tumor_proba_mask_img=(tumor_proba_mask*255).astype(int)

                    # Enregistrement du mask
                    cv2.imwrite(os.path.join(heat_pred_folder, list_name[idx]+".png"), tumor_proba_mask_img)

if __name__ == '__main__':
    main()