import os, cv2
import numpy as np
import pandas as pd
import tqdm
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
#%matplotlib inline
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segmentation_models_pytorch import utils
import albumentations as album
import segmentation_models_pytorch as smp
from typing import Iterator
warnings.filterwarnings("ignore")

RESULT_DIR="C:/Users/Hajar/Documents/result24/"
if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

TRAINING = False
PREDICTION_LOOP=False
RELOAD_TRAINNING=False

class_name = ['Background', 'Tumor']
rgb_values = [[255, 255, 255], [200, 0, 0]]
# rgb_values = [[255, 255, 255], [85, 85, 85]]
select_class_name = ['Background', 'Tumor']

ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
# CLASSES = class_names
CLASSES = class_name
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

TRAIN_PARAMETERS=[
                    # [Batch size, Epochs number, Learning Rate, Dataset Dir, Image Size, Reload Traiing Directory], 
                ]

IMAGE_SIZE=512

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
        masks_dir, 
        class_rgb_values=None, 
        augmentation=None, 
        preprocessing=None,
    ):
        
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        masks_dir=masks_dir

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
        
    def __len__(self):
        # return length of 
        return len(self.image_paths)

#Add a smp pmetrics to have specificity
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
    
# functions for specificity
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

class Activation(nn.Module):
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
    
class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)

class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

def get_training_augmentation():
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
def crop_image(image, target_image_dims=[IMAGE_SIZE,IMAGE_SIZE,3]):
    target_size = target_image_dims[0]
    image_size = len(image)
    padding = (image_size - target_size) // 2

    return image[
        padding:image_size - padding,
        padding:image_size - padding,
        :,
    ]

def main():

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    print("Trainning : ", TRAINING)
    if TRAINING:
        for j, train_parameter in enumerate(TRAIN_PARAMETERS):

            #get folder
            folder=RESULT_DIR + "train" + str(j) + "/"
            if not os.path.exists(folder):
                os.makedirs(folder)

            x_train_dir = os.path.join(train_parameter[3], 'train')
            y_train_dir = os.path.join(train_parameter[3], 'train_labels')

            x_valid_dir = os.path.join(train_parameter[3], 'val')
            y_valid_dir = os.path.join(train_parameter[3], 'val_labels')

            #Create model or load a previous model
            if RELOAD_TRAINNING and os.path.exists(train_parameter[5]+'best_model.pth'):
                print("load a previous model for train")
                model = torch.load(train_parameter[5]+'best_model.pth', map_location=DEVICE)
            else:
                print("creating a new model")
                model = smp.DeepLabV3Plus(
                    encoder_name=ENCODER, 
                    encoder_weights=ENCODER_WEIGHTS, 
                    classes=len(CLASSES), 
                    activation=ACTIVATION,
                )

            #Building dataset
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
            train_loader = DataLoader(train_dataset, batch_size=train_parameter[0], drop_last=True, shuffle=True, num_workers=16, pin_memory=True)
            valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
        
            # define loss function
            loss = smp.utils.losses.DiceLoss()

            # define metrics
            metrics = [
                smp.utils.metrics.IoU(threshold=0.5),
                smp.utils.metrics.Accuracy(threshold=0.5),
                smp.utils.metrics.Recall(threshold=0.5),
                Specificity(threshold=0.5),
            ]

            # define optimizer
            optimizer = torch.optim.Adam([ 
                dict(params=model.parameters(), lr=train_parameter[2]),
            ])

            # define learning rate scheduler (not used in this NB)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=1, T_mult=2, eta_min=5e-5,
            )

            # load best saved model checkpoint from previous commit (if present)
            if os.path.exists('../input/deeplabv3-efficientnetb4-frontend-using-pytorch/best_model.pth'):
                model = torch.load('../input/deeplabv3-efficientnetb4-frontend-using-pytorch/best_model.pth', map_location=DEVICE)

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

            print("Training for : " + str(j) )
            
            best_iou_score = 0.0
            train_logs_list, valid_logs_list = [], []

            for i in range(0, train_parameter[1]):
                # Perform training & validation
                print('\nEpoch: {}'.format(i))
                train_logs = train_epoch.run(train_loader)
                valid_logs = valid_epoch.run(valid_loader)
                train_logs_list.append(train_logs)
                valid_logs_list.append(valid_logs)
                print(train_logs)
                print(valid_logs)
                train_logs_df = pd.DataFrame(train_logs_list)
                valid_logs_df = pd.DataFrame(valid_logs_list)
                train_logs_df.T
                train_logs_df.to_csv(folder +"train_logs_df.csv", sep = ",")
                valid_logs_df.to_csv(folder +"valid_logs_df.csv", sep = ",")

                # Save model if a better val IoU score is obtained
                if best_iou_score < valid_logs['iou_score']:
                    best_iou_score = valid_logs['iou_score']
                    torch.save(model, folder + 'best_model.pth')
                    print('Model saved!')

            train_logs_df = pd.DataFrame(train_logs_list)
            valid_logs_df = pd.DataFrame(valid_logs_list)
            train_logs_df.T
            train_logs_df.to_csv(folder +"train_logs_df.csv", sep = ",")
            valid_logs_df.to_csv(folder +"valid_logs_df.csv", sep = ",")
            
            plt.figure(figsize=(20,8))
            plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label = 'Entraînement')
            plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label = 'Validation')
            plt.xlabel('Epochs', fontsize=20)
            plt.ylabel('Loss', fontsize=20)
            plt.ylim(0, 1)
            plt.title('Fonction de perte', fontsize=20)
            plt.legend(loc='best', fontsize=16)
            plt.grid()
            plt.savefig(folder + 'dice_loss_plot.png')
            # plt.show()

    for  j, train_parameter in enumerate(TRAIN_PARAMETERS):
        
        folder=os.path.join(RESULT_DIR, "train" + str(j))

        # load best saved model checkpoint from the current run
        if os.path.exists(folder +'best_model.pth'):
            best_model = torch.load(folder+'best_model.pth', map_location=DEVICE)
            print('Loaded DeepLabV3+ model from the train : ' + str(j))

        x_test_dir = os.path.join(train_parameter[3], 'test')
        y_test_dir = os.path.join(train_parameter[3], 'test_labels')

        # create test dataloader (with preprocessing operation: to_tensor(...))
        test_dataset = BuildingsDataset(
            x_test_dir, 
            y_test_dir, 
            augmentation=get_validation_augmentation(), 
            preprocessing=get_preprocessing(preprocessing_fn),
            class_rgb_values=rgb_values,
        )

        test_dataloader = DataLoader(test_dataset, num_workers=16,  pin_memory=True)

        #Indicate the path where you want to put the file with the prediction results
        heat_pred_folder= os.path.join(folder, 'prediction/')

        if not os.path.exists(heat_pred_folder):
            os.makedirs(heat_pred_folder)
        
        print("prediction loop : ", PREDICTION_LOOP)

        if (PREDICTION_LOOP):

            list_name=[os.path.splitext(filename)[0] for filename in sorted(os.listdir(x_test_dir))]
            
            for idx in tqdm(range(len(test_dataset))):
                image, gt_mask = test_dataset[idx]
                x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

                pred_mask = best_model(x_tensor)
                pred_mask = pred_mask.detach().squeeze().cpu().numpy()

                tumor_proba_mask=pred_mask[1]
                tumor_proba_mask_img=(tumor_proba_mask*255).astype(int)

                cv2.imwrite(os.path.join(heat_pred_folder, list_name[idx]+".png"), tumor_proba_mask_img)
                
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

        test_epoch = smp.utils.train.ValidEpoch(
            best_model,
            loss=loss, 
            metrics=metrics, 
            device=DEVICE,
            verbose=True,
        )

        valid_logs = test_epoch.run(test_dataloader)
        print(valid_logs.keys())
        print(type(valid_logs))

        print("Evaluation on Test Data: ")
        print(f"Mean IoU Score : {valid_logs['iou_score']:.4f}")
        print(f"Mean Dice Loss : {valid_logs['dice_loss']:.4f}")
        print(f"Mean Accuracy : {valid_logs['accuracy']:.4f}")
        print(f"Sensibility : {valid_logs['recall']:.4f}")
        print(f"Specificity : {valid_logs['specificity']:.4f}")
        valid_logs_df = pd.DataFrame([valid_logs])
        valid_logs_df.to_csv(folder +"test_logs_df.csv", sep = ",")

if __name__ == '__main__':
    main()