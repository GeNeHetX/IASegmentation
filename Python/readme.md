# Segmentation d'images avec Python

## Segmentation Sampler

### Les données d'entrée 

Il prend en entrée : 
- RESULT_DIR (str) : Chemin du dossier de destination des résultats
- TRAINNING (bool) : S'il faut fiare un entrainement
- PREDICTION_LOOP (bool) : S'il faut créer les prédictions sous forme de carte de proba
- RELOAD_TRAINNING (bool) : S'il faut réutiliser un précédent modèles
- class_name (list) : Liste contenant le nom des classes
- rgb_value (list) : Valeur des couleurs de nos classes
- ENCODER (str) : Modèle utilisé pour la segmentation, voir les compatibilité pour la libraire SMP
- ENCODER_WEIGHTS (str) : Fine tunning utilisé pour le modèle, voir  la compatibilité pour la libraire SMP
- TRAIN_PARAMETERS (list) : Paramètres pour l'entrainement et le test (effectué séqunetiellement par élément de la liste). Il suit cet ordre  : [ batch_size (int), nombre d'époque (int), learning rate (float), chemin du dataset (str), taille des images (int), chemin du modèle à réutiliser]

### Les données de sortie

Le code ressort plusieurs  données toutes contenu ans le dossier indiqués dans le résultat. Pour chauqe modèle, il crée un dossier nommé train{i} avec le i correspondant au numéro de l'entrainement dans l'ordre spécifié au dessus (en commençant par 0). Chacun de ces dossier comprends :
- Le modèle en .pth
- Les métriques de validations en .CSV 
- Les métriques d'entrainement en .CSV
- Les métriques de test en .CSV
- Un graphe représentant l'évolution des métriques (peu utile en soit)
- Un dossier comprenant les prédictions (si la prédicitons loop est activé) c'est à dire des tuiles (images PNG) représenant la probabilités de tumeurs entre 0 et 255. Le même nom que les images initales est gardé.

### Information sur le code :

Organisation générale du code :
- La première partie regroupe la déclaration des entrées qui ont été le plus possible regroupés au début du code, elles sont détaillés dans les entrées du code. (Ligne 20 à 43)
- La deuxième la déclaration de fonction nécessaire pour fonctionner (Ligne 45 à 348)

A la ligne 285, il est indiqué l'augmentation qui est souhaité pour l'entrainement

A la ligne 311 celle pour la validation et le test

Les dossier dans lequels doivent être les images sont dans le dossier dataset sous le nom suivant :
- train et train_labels pour l'ensemble d'entrainement respectivement pour les images et pour les masques.
- val et val_labels pour l'ensemble de validation respectivement pour les images et pour les masques
- test et test_labels pour l'ensemble de test respectivement pour les images et pour les masques

Ligne 422 il y a une ébauche pour le learning rate scheduler qui n'est pas fonctionnel en l'etat.



## Segmentation Prédiction

Ce script permet de faire les prédictions à travers un modèle donnée et des tuiles exportées.

### Paramètre

Il prend en entrée :
- DATA_DIR (str): Localisation des tuiles (mettre le fichier)
- PREDICT_DIR (str): Nom du dossier avec les prédictions
- PREDICTION_SEQUENCE (list): Chemin du dossier du modèle pour lesquels on fait les prédictions

### Sortie

Le code écrit dans le dossier du modèle les prédictions dans un nouveau dossier qui a le nom donnée dans prédict_dir des tuiles en niveau de gris représentant la probabilité de prédiction pour la classe.

### Remarques

En l'état actuel du code, le model doit se nomme best_model.pth dans le dossier.

## Move Train

Il permet de splitter l'export en trainet validation aléatoirement.

### paramètres

- image_folder (str) : Dossier avec les images
- label_folder (str) : Dossier avec les labels
- train_image_folder (str) : Dossier de direction des images de train
- train_label_folder (str) : Dossier de direction des labels de train
- validation_image_folder (str) : Dossier de direction des images de validation
- validation_label_folder (str) : Dossier de direction des labels de validation
- validation_split (float) : Part des tuiles allant dans la validation

Il faut que dans les dossier les noms des labels soient strictement identique au nom des images en .tif pour les images et .png pour les labels.

## Unpatchify_mask

Permet de fusionner les tuiles selon la moyenne de probabilités. Utilise cython, l'extension unpatchify mask doit être dans le même dossier que le code.

### Paramètres

 - RESULT_DIR (str) : Nom du dossier créé pour y mettre les resultats
 - DATA_DIR (list) : liste des dossier contenant les prédictions (chacune est faite un par un)
 - size (int) : taille des nos tuiles (elles doivent être carrés)
 - threshold_proba (int) : seuil de niveau de gris nécessaire pour classifier positif (entre 0 et 255)

### Sortie 

Le script trie les tuiles selon leur noms et met chaque lame dans un dossier
Ensuite il fusionne toutes les tuiles et enregistre une image de notre tuiles entièrene png et exporte le geojson correspondant aux annotations selon le seuil prédéfini. Attention le geojson prédit les négatifs et pas les positifs.

### Remarque

Attention au formatage du nom de l'image il est nécessaire pour retrouver les valeurs de x et de y

## Export Tiles

Permet d'exporter des tuiles à partir de WSI
Utilise open slide

### Paramètres
- RESULT_DIR (str): Dossier dans lequel on met les tuiles exportés
- DATA_DIR (str) : Dossier dans lequel se trouve les lames
- size (tuple) : Taille des tuiles format (x, y)
- down_sample (int) : valeur du downsample
- overlap (int) : Valeur du chevauchement, doit être inférieur à la taille de l'image.
- threshold (int) : Valeur en dessous de la quelle le pixel en niveau de gris est considéré comme du tissue
- part_tissue_min (str) : Proportion de tissue minimale pour garder la tuiles

### Sortie
Les tuiles sont enregistrés dans le fichier prédéfini dans RESULT_DIR. Le script boucle sur tout les élément du dossier indiqués dans DATA_DIR

Les tuiles dont la preoportion de tissue est inférieur au seuil ne sont pas enregistré.

## Prediction all

Permet de faire des prédiction en donnant un modèle et une WSI d'entrée. Retourne un GeoJSON et une image de masque.

### Paramètre
- ENCODER (str) : Nom du modèle dans la librairie SMP pour notre modèle
- ENCODER_WEIGHTS (str) : Nom du dataset utilisé pour le fine tunning dans la librairie SMP pour notre modèle.
- DATA_DIR (str) : Chemin du fichier de l'image entière. (format autorisé par openslide pour l'image)
- TILES_DIR (str) : Dossier de localisation du dossier temporaire contenant les tuiles de probabilité prédites
- MODEL_DIR (str) : Chemin du fichier où se situe le modèle (en .pth)
- RESULT_DIR (str) : Dossier dans lequel veulent être mis les résultats
- IMAGE_NAME (str) : Nom de la lames qui doit être mis dans les fichier résultats
- size (tuple) : Dimension des tuiles utilisés pour les prédictions
- down_sample (int) : Redimensionnement de l'image
- overlap (int) : Taille du chevauchmeent entre les images
- threshold_tissue (int) : Valeur en dessous de la quelle le pixel en niveau de gris est considéré comme du tissue
- part_tissue_min (float) : Pourcentage de tissue minimum pour faire la prédiction sur les tuiles
- thresold_prediction (int) : Valeur minimale de la probabilité sur une echelle de 0 à 255 nécessaire pour classifier un pixel comme positif


### Sortie

Retourne dans le dossier résultat un masque de l'image entière en probabilité et iun geojson indiquant les zones non tumorales.

Le dossier temporaire avec les tuiles peut ensuite être supprimé.

### Remarques

**Le code n'a jamais été testé donc est-ce qu'il marche ????**

Il faut que le code cython du module unpatchify mask soit dans le même dossier

Il est nécessaire d'indiquer l'encoder et les poids pour permettre d'effectuer la même normalisation sur les images de prédiction qu'au moment de l'entrainement.

## Pour la partie cython utilisé dnas unpatchify_mask et prediction_all

Les fichier sont compilé en utilisant cette commande :

```python setup.py build_ext --inplace```
Le code setup.py indiquant la localisation du code à compliler

Le code en tant que tel est dans le fichier ```unpatchify_mask.pyx```

### Module et librairie nécessaire

J'ai du installer build_essential :```sudo apt install build_essential```

ET la librairie cython : ```pip install cython```

Je n'ai jamais réussi à le faire marché sur windows

