# Segmentation d'images avec Python

## Les données d'entrée 

Vous avez des patchs comportant des tuiles et des masques. Les tuiles doivent être au format tiff et les masques au format png. 
Dans class_name, indiquer le nom de votre classe, puis, dans rgb_values la couleur correspondante à vos annotations. Enfin dans select_class_name, il faut 
indiquer la classe sur laquelle vous allez entraîner le modèle.

## Les données de sortie

Dans la partie de prédiction du test (ligne 396), vous avez 3 variables : image_vis correspond à vos tuiles, gt_mask correspond à vos masques et pred_mask correspond aux prédictions sur le fichier test. 
En sortie vous avez une image png avec la tuile, le masque et la prédiction correspondante. Le chemin vers le dossier de prédiction est à indiquer à la ligne 391 (sample_preds_folder).

## Les modifications à faire

- Le chemin vers les dossiers d'entraînement, de validation et de test à la ligne 25
- Indiquer la taille de vos patchs à la ligne 174, 190 et 215
- Si vous souhaitez appliquer la fonction d'augmentation, retirer '=None' à la ligne 132
- Si vous avez plusieurs classes, modifier la fonction d'activation sigmoid par softmax2d à la ligne 251
- Vous pouvez modifier le nom du modèle entrainé à la ligne 353. ATTENTION laisser le .pth
- Une fois le modèle entraîné, vous voulez appliquer le modèle sur un nouveau jeu de donnée uniquement pour la prédiction :
  * Changer le dossier test dans votre explorateurs de fichier, vous pouvez mettre vos tuiles dans le dossier Images et des masques blancs dans Labels
  * Indiquer TRAINING = False à la ligne 285
  * Vous pouvez modifier la sortie en retirant le masque à la ligne 417


