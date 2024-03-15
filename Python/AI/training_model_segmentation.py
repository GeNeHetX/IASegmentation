##### MODEL SEGMENTATION ####

##################### IMPORT ##################
import os
import cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tifffile as tf
from tqdm import tqdm

#%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album

from stardist.models import StarDist2D
from csbdeep.utils import normalize
import re
import statistics
from skimage import io

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from PIL import Image

#################################################

############# Parametre à modifier #################

TRAINING = True # Pour entrainer le model 
PREDICTION_LOOP= False # Pour predir des lames

# Set num of epochs
EPOCHS = 10

#Chemin du dossier qui contient dataset
DATA_DIR="/" 

#Chemin du dossier qui contiendra les resultats 
RESULT_DIR = "/result_adj_ttneo_25NM_cree_epoch10test2_10/" 

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
        
# Données normalisés 
BATH_SIZE=30
LEARNING_RATE=0.00001
IMAGE_SIZE=512

# Recuperation dataset
x_train_dir = os.path.join(DATA_DIR+'Train_adj_ttneo_25NM_10/', 'image')
y_train_dir = os.path.join(DATA_DIR+'Train_adj_ttneo_25NM_10/', 'label')
x_valid_dir = os.path.join(DATA_DIR+'Valid_adj_ttneo_10/', 'image')
y_valid_dir = os.path.join(DATA_DIR+'Valid_adj_ttneo_10/', 'label')

class_name = ['Background', 'Tumor']
rgb_values = [[255, 255, 255], [200, 0, 0]]
select_class_name = ['Background', 'Tumor']

##################### FONCTION ####################

################ Fonction entrainement ##############

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
# Permet de segmenter l'image en fonction des différents colour 
# Il garde la même taille et change le nombre de profondeur en fonction du nombre de couleur que l'on veut observer 
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

def _threshold(x, threshold=None): # Utiliser dans specifiity 
    if threshold is not None:
        return (x > threshold).type(x.dtype) 
    else:
        return 
    
def _take_channels(*xs, ignore_channels=None): # Utiliser dans specificity 
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs

# functions for specificity
#  permet d'évaluer sa capacité à éviter de classer à tort des éléments négatifs comme positifs
def specificity(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    tn = torch.sum((1 - gt) * (1 - pr))  # True negative
    fp = torch.sum((1 - gt) * pr) #false positive

    score = (tn + eps) / (tn + fp + eps)

    return score

class BuildingsDataset(torch.utils.data.Dataset):

    """Massachusetts Buildings Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): couleur values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
        self, 
        images_dir, 
        masks_dir, 
        class_rgb_values=None, 
        augmentation=None, 
        preprocessing=None,
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        masks_dir=masks_dir #ligne non cohérente 

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read images and masks
        
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask =cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
    
    def get_class(self, i):
        #must to change th e treshold to have over class
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2GRAY)
        mask_binarize=np.where(mask!=255, 1,0)
        sum_mask=np.sum(mask_binarize)
        if sum_mask==0:
            class_sum=0
        else:
            class_sum=1
        return class_sum
        

    def __len__(self):
        # return length of 
        return len(self.image_paths)

class ArgMax(nn.Module): # Utiliser pour activation 
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)
    
class Clamp(nn.Module):  # Utiliser pour activation 
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)
    
class Activation(nn.Module): # UTilise que dans specificity 
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)
    
#Add a smp pmetrics to have specificity
# CLASS qui permet de sauvegarder resultats specificité 
# Associé avec les autres métrique 
class Specificity(smp.utils.base.Metric): 
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return specificity(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
    
def get_training_augmentation(): #permet de augmenter le dataset en changant la position bougeant la saturation... 
    train_transform =[
        album.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, always_apply=True),
        album.Compose([
            album.OneOf([
                album.RandomBrightnessContrast(p=1),
                album.ColorJitter(p=1, brightness=(0.7, 1.3), contrast=(0.7, 1.3), saturation=(0.7, 1.3), hue=(-0.3, 0.1)),
                album.RandomGamma(gamma_limit=(60, 140)),
                album.HueSaturationValue(hue_shift_limit=(-30, 10), sat_shift_limit=(-30, 30), val_shift_limit=(-30, 30))
            ], p=0.9),
            album.OneOf([
                album.GaussNoise(var_limit=(10.0, 250.0)),
                album.Blur(blur_limit=(3,8))
            ], p=0.9),
            album.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), interpolation=1, border_mode=2, p=0.3),
            album.OneOf([
                album.HorizontalFlip(p=1.0),
                album.VerticalFlip(p=1.0),
                album.RandomRotate90(p=1),
            ], p=0.9),
            
        ], p=0.75)
    ]

    return album.Compose(train_transform)

def get_validation_augmentation():   
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

# pretraitement des données avant d'etre envoyé au model de prediction 
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
def crop_image(image, target_image_dims=[IMAGE_SIZE,IMAGE_SIZE,3]): # NON UTILISER DANS LE CODE 
    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,
    ]


###################################################

################ MAIN ###############################

###################################################

def main():
    dataset = BuildingsDataset(x_train_dir, y_train_dir, class_rgb_values=rgb_values)
    random_idx = random.randint(0, len(dataset)-1)
    image, mask = dataset[2]

    #Augmente donnée test 
    augmented_dataset = BuildingsDataset(
    x_train_dir, y_train_dir, 
    augmentation=get_training_augmentation(),
    class_rgb_values=rgb_values,
    )

    random_idx = random.randint(0, len(augmented_dataset)-1)

    # Different augmentations on a random image/mask pair (256*256 crop)
    for i in range(3):
        image, mask = augmented_dataset[random_idx]
        

    #Création du model à 0
    ENCODER = 'resnet18'
    ENCODER_WEIGHTS = 'imagenet'
    # CLASSES = class_names
    CLASSES = class_name
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

    # create segmentation model with pretrained encoder
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    #Création des dataset train et validation 
    # Get train and val dataset instances
    train_dataset = BuildingsDataset(
        x_train_dir, y_train_dir, 
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        # class_rgb_values=select_class_rgb_values,
        class_rgb_values=rgb_values,
    )
    valid_dataset = BuildingsDataset(
        x_valid_dir, y_valid_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        # class_rgb_values=select_class_rgb_values,
        class_rgb_values=rgb_values,
    )

    # Get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=30, drop_last=True, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): 
        print("Utilisation GPU")
    else :
        print("Utilisation cpu")
    #Matric prediction
    # define loss function
    loss = smp.utils.losses.DiceLoss()

    # define metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Accuracy(threshold=0.5),
        #sensibility
        smp.utils.metrics.Recall(threshold=0.5),
        #specificity
        Specificity(threshold=0.5),
    ]

    # define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    # define learning rate scheduler (not used in this NB)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    # load best saved model checkpoint from previous commit (if present)
    

    ########################################################################################
    
    # Si déja model entrainer mettre fichier dans le dossier où les resultats seront enregistrer
    if os.path.exists(RESULT_DIR+'best_model_cree_pre0.pth'):
        model = torch.load(RESULT_DIR+'best_model_cree_pre0.pth', map_location=DEVICE)
        print('Utilisation du model pre entrainé')

    ########################################################################################

    print("premier best model")

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )
    
    print(TRAINING)
    stop=0
    if TRAINING:

        best_iou_score = 0.0
        train_logs_list = []
        valid_logs_list = []

        for i in range(0, EPOCHS):
            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            
            valid_logs = valid_epoch.run(valid_loader)
            
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)
            print(train_logs)

            # Save model if a better val IoU score is obtained
            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                torch.save(model, RESULT_DIR+'best_model_cree_'+str(i)+'.pth') # permet de savoir quel epoch est la meilleur 
                print('Model saved!')
                stop=0
            else:
                stop=stop+1

            if stop ==5:
                #break  
                print("aurait du stop ")

        torch.save(model, RESULT_DIR+'best_model_cree_final.pth')
        print("on est sorti")
        train_logs_df = pd.DataFrame(train_logs_list)
        valid_logs_df = pd.DataFrame(valid_logs_list)
        train_logs_df.T
        train_logs_df.to_csv("train_logs_df.csv", sep = ",")
        valid_logs_df.to_csv("valid_logs_df.csv", sep = ",")

        print("csv enregistrer")
        '''
        plt.figure(figsize=(20,8))
        plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label = 'Entraînement')
        plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label = 'Validation')
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.ylim(0, 1)
        plt.title('Fonction de perte', fontsize=20)
        plt.legend(loc='best', fontsize=16)
        plt.grid()
        plt.savefig('dice_loss_plot.png')

        print("figure enregistrer ")
        # plt.show()
        '''
    ########################################################################################

    # load best saved model checkpoint from the current run
    if os.path.exists(RESULT_DIR+'best_model_cree_final.pth'):
        model = torch.load(RESULT_DIR+'best_model_cree_final.pth', map_location=DEVICE)
        print('Model Saved used.')

    # load best saved model checkpoint from previous commit (if present)
    elif os.path.exists(RESULT_DIR+'best_model.pth'):
        model = torch.load(RESULT_DIR+'best_model.pth', map_location=DEVICE)
        print('model from a previous commit.')

    ########################################################################################


    print("deuxieme best model")
    # create test dataloader (with preprocessing operation: to_tensor(...))


    if PREDICTION_LOOP:    
        i=0
        for lame_test in os.listdir(DATA_DIR+'Test/'):
            i=i+1
            if i >0:
                x_test_dir = os.path.join(DATA_DIR+'Test/'+lame_test+'/', 'image')
                y_test_dir = os.path.join(DATA_DIR+'Test/'+lame_test+'/', 'label')

                test_dataset = BuildingsDataset(
                    x_test_dir, 
                    y_test_dir, 
                    augmentation=get_validation_augmentation(), 
                    preprocessing=get_preprocessing(preprocessing_fn),
                    class_rgb_values=rgb_values,
                )

                test_dataloader = DataLoader(test_dataset)

                # test dataset for visualization (without preprocessing transformations)
                test_dataset_vis = BuildingsDataset(
                    x_test_dir, y_test_dir, 
                    augmentation=get_validation_augmentation(),
                    class_rgb_values=rgb_values,
                )

                # get a random test image/mask index
                random_idx = random.randint(0, len(test_dataset_vis)-1)
                image, mask = test_dataset_vis[random_idx]

            #Indicate the path where you want to put the file with the prediction results
                lame_folder = RESULT_DIR +lame_test+'/'
                
                #heat_pred_folder= RESULT_DIR + 'heat_pred/'
                if not os.path.exists(lame_folder):
                    os.makedirs(lame_folder)
                    sample_preds_folder=lame_folder+'sample_predictions__tumor2/'
                    os.makedirs(sample_preds_folder)
                    tile_folder= lame_folder + 'tile_prediction/'
                    os.makedirs(tile_folder)
                else : 
                    sample_preds_folder=lame_folder+'sample_predictions__tumor2_15epoch/'
                    os.makedirs(sample_preds_folder)
                    tile_folder= lame_folder + 'tile_prediction_15epoch/'
                    os.makedirs(tile_folder)
            
                #if not os.path.exists(heat_pred_folder):
                    #os.makedirs(heat_pred_folder)
                
                total_confu=[[0,0],[0,0]]
                list_name=[os.path.splitext(filename)[0] for filename in sorted(os.listdir(x_test_dir))]

                for idx in tqdm(range(len(test_dataset))):
                    image, gt_mask = test_dataset[idx]
                    image_vis = crop_image(test_dataset_vis[idx][0].astype('uint8'))
                    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
                    # Predict test image
                    pred_mask = model(x_tensor)
                    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
                    # Convert pred_mask from `CHW` format to `HWC` format
                    
                    pred_mask = np.transpose(pred_mask,(1,2,0))

                    # Get prediction channel corresponding to building
                    pred_tumor_heatmap = pred_mask[:,:,select_class_name.index('Tumor')]
                    pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), rgb_values))
                    # Convert gt_mask from `CHW` format to `HWC` format
                    gt_mask = np.transpose(gt_mask,(1,2,0))
                    gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), rgb_values))
                        
                    cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), np.hstack([image_vis, gt_mask, pred_mask])[:,:,::-1])
                
                    cv2.imwrite(os.path.join(tile_folder, f"Labels_{list_name[idx]}.png"), pred_mask)


    # visualize(
    #     original_image = image_vis,
    #     ground_truth_mask = gt_mask,
    #     predicted_mask = pred_mask,
    #     predicted_tumor_heatmap = pred_tumor_heatmap
    # )



    test_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    valid_logs = test_epoch.run(valid_loader)
    print(valid_logs.keys())
    print(type(valid_logs))
    print("Evaluation on Test Data: ")
    print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
    print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")
    print(f"Mean Accuracy: {valid_logs['accuracy']:.4f}")
    print(f"Sensibility : {valid_logs['recall']:.4f}")
    print(f"Specificity : {valid_logs['specificity']:.4f}")
  
if __name__ == '__main__':

    main()
