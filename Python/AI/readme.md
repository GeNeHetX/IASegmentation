# AI

Après avoir récupéré l'ensemble de nos tuiles, nous pouvons entraîner notre modèle. Nous avons effectués les étapes suivants :
- [Algorithme de segmentation](#1)
    - [training_model_segmentation.py](#training)
    - [best_model_seg.pth](#model)

- [Algorithme de classification](#2)
    - [training_model_classification.py](#class_train)
    - [best_model_cal.pth](#model_cla)
    - [predict_model_classification.ipynb](#pred)

<div id='1'/>

# Algorithme de segmentation 
<div id='training'/>

Pour pouvoir entrainer notre modèle nous utilisons le code : `training_model_segmentation.py`.

### Les données d'entrée : 

- `TRAINING` : est une variable binaire afin d'indiquer si nous voulons effectuer l'entraînement.
- `PREDICTION_LOOP` : est une variable binaire afin d'indiquer si nous voulons effectuer les prédictions sur la partie Test.
- `EPOCH` : correspond au nombre total d'epochs que nous voulons effectuer, généralement 10.
- `DATA_DIR` : est le chemin du dossier où se trouvent l'ensemble des dossiers Train, Valid et Test.
- `RESULT_DIR` : est le chemin du dossier où nous voulons enregistrer les résultats.
- `BATCH_SIZE` : correspond à la taille des lots pour l'entraînement, mis à 30.
- `LEARNING_RATE` : est le taux d'apprentissage, mis à 0.00001.
- `IMAGE_SIZE` : correspond à la taille de nos images, soit 512.

### Les données de sortie : 

Le programme enregistre le modèle en **.pth** dès que son `iou_score` est meilleur que celui de l'epoch précédent.

Lorsque `PREDICTION_LOOP` est **True** : le programme enregistre un dossier pour chaque lame testée contenant deux dossiers `sample_predictions_tumor2` qui permet de comparer le label et la prédiction pour chaque tuile, et `tile_prediction` qui contient l'intégralité des tuiles prédites.

### Information sur le code : 

Il faut faire attention aux lignes **60** à **63** pour vérifier que les noms des fichiers `Train` et `Valid` sont corrects.

Définir le nom du modèle **.pth** si l'on veut le ré-entraîner ligne **634** et **635**.

Vérifier que l'on récupère le bon modèle pour la prédiction ligne **718** et **719**.

Vérifier que le nom du dossier `Test` est correct ligne **736** à **740**.

### Organisation générale du code :

- La première partie regroupe la déclaration des entrées qui ont été le plus possible regroupés au début du code, elles sont détaillés dans les entrées du code. (Ligne 20 à 43)
- La deuxième la déclaration de fonction nécessaire pour fonctionner (Ligne 45 à 348)

A la ligne 285, il est indiqué l'augmentation qui est souhaité pour l'entrainement

A la ligne 311 celle pour la validation et le test


Ligne 422 il y a une ébauche pour le learning rate scheduler qui n'est pas fonctionnel en l'etat.


### Résultats obtenus avec le modèle de segmentation :

Nous avons lancé le modèle sur plus de 7 bases de données différentes afin de trouver le meilleur entraînement possible, étape regroupées dans `Résultats_diff_seg_BDD.pdf`, donnant les résultats suivant : 

- **Précision** = 0.9718
- **Sensibilité** = 0.6492
- **Spécificité** = 0.9901
- **VPP** = 0.7480

<div id='model'/>

Nous avons donc continué les entraînements avec cette base de données, créant le modèle : `best_model_seg.pth`.



<div id='2'/>

# Algorithme de classification
<div id='class_train'/>

J'ai initié la création d'un modèle de classification, mais il faudra fortement continuer à l'améliorer. J'ai utilisé le code `training_model_classification.py` :

### Les données d'entrée :

Le programme a besoin des fichiers `Train` et `Valid` pour s'entraîner, qu'il faut spécifier aux lignes **28** et **40**.

### Les données de sortie :

Le programme retourne le modèle en **.pth**.

### Information sur le code :
<div id='model_cla' />

La précision maximale obtenue avec ce modèle est de 78% enregistré dans `best_model_cla.pth`.

<div id='pred'/>

Pour tester les lames, il faut utiliser un second code : `predict_model_classification.ipynb`.

### Les données d'entrée :

- `model` et `checkpoint` : sont les chemins d'accès au modèle.
- `TEST_DIR` : est le chemin d'accès au dossier `Test`.
- `RESULT_DIR` : est le chemin où l'on veut enregistrer les prédictions.
- `test_data` : est le chemin d'accès aux données de test pour une seule lame.
- `output_dir_tumoral` : est le chemin où l'on enregistre les prédictions du lit tumoral pour une seule lame.
- `output_dir_non_tumoral` : est le chemin où l'on enregistre les prédictions du background pour une seule lame.

### Les données de sortie :

Le modèle retourne les tuiles en les classant dans deux dossiers : soit `lit` soit `background` suivant les résultats de la prédiction.

### Information sur le code :

Nous pouvons lancer le code pour prédire le fichier test en entier ou seulement une lame.