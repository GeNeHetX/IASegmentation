# Evaluation des modèles

Afin d'évaluer correctement les performances de nos modèles de segmentation nous effectuons les étapes suivantes :

- [Visualisation des résultats](#1)
    - [See_result.py](#see)
    - [unpatchify_mask.py](#unpatchify)
    - [See_result_classification.py](#cla)
- [Calcul des métriques](#2)
    - [compter_noyauxV2.py](#stardist)
    - [identifier_noyaux.py](#metric)
- [Probabilité prédictions](#3)
    - [segmentation_prediction.py](#segpred)

<div id='1'/>

# Visualisation des résultats :

Le facteur le plus important pour nous est la composante visuelle, permettant de voir et de comprendre les cellules sur lesquelles notre programme commet le plus d'erreurs.

<div id='see'/>

J'ai ainsi créé le code Python `See_result.py`.

### Les données d'entrée :

- `RESULT_DIR` : chemin d'accès au dossier contenant les résultats de `training_model_segmentation.py`
- `IMAGE_DIR` : chemin du dossier contenant les lames entières
- `LABEL_DIR` : chemin du dossier contenant les tuiles avec les labels rangées par lame 
- `SAVE_GEOJSON` : variable booléenne indiquant si l'on souhaite enregistrer la prédiction au format GeoJSON

### Les données de sortie :

Le code renvoie une image de la lame avec la superposition des étiquettes et des prédictions, ainsi qu'un fichier GeoJSON contenant les prédictions (si `SAVE_GEOJSON` est **True**).

Sur l'image **.png** enregistrée, on peut facilement distinguer :
- les `FN` en **vert**
- les `TP` en **jaune-marron**
- les `FP` en **rouge**

### Informations sur le code :

- Les lignes **111** à **120** créent les listes `image_nom` et `label_nom` permettant d'accéder au bon dossier pour chaque lame. Cela peut varier en fonction de vos dossiers.
- La variable `test` à la ligne **144** spécifie si nous voulons effectuer la visualisation pour plusieurs fichiers résultat en même temps. Attention, le programme ne regardera que les lames présentes également dans le chemin spécifié dans `RESULT_DIR`.
- Nous pouvons effectuer la visualisation uniquement sur les étiquettes ou uniquement sur les prédictions en commentant soit les lignes **168** à **172**, soit les lignes **161** à **166**.
- Pour enregistrer les prédictions dans le GeoJSON, il faut utiliser le code `unpatchify_mask.py`.

<div id="unpatchify"/>

## Informations sur le code `unpatchify_mask.py` :

### Les données d'entrée :

- `RESULT_DIR` : Nom du dossier créé pour stocker les résultats
- `DATA_DIR` : liste des dossiers contenant les prédictions (chacune est générée une par une)
- `size` : taille de nos tuiles (elles doivent être carrées)
- `threshold_proba` : seuil de niveau de gris nécessaire pour classifier positif (entre 0 et 255)

### Les données de sortie :

On obtient un GeoJSON faisant la moyenne des probabilités selon le seuil prédéfini. Attention, le GeoJSON prédit les négatifs et non les positifs.

### Informations sur le code :

Pour utiliser le code `unpatchify_mask.py`, il faut d'abord enregistrer le dossier `cython_extension`, puis télécharger les différents modules nécessaires :

Pour compiler :
```shell
python setup.py build_ext --inplace
```
Installation build_essential
```shell
sudo apt install build_essential
```
Librairie Cython 
```shell
pip install cython
```
## Visualisation pour classification : 
<div id='cla' />

Nous pouvons également visualiser les résultats pour notre modèle de classification grace au code `See_result_classification.py`

### Les données d'entrée : 

- `IMAGE_DIR` : est le chemin du dossier contenant les lames
- `image_nom` : est le nom du fichier **.svs** sur lequel nous voulons effectuer la visualisation
- `PREDICT_LIT` : est le chemin du dossier où ce trouve les tuiles prédites comme lit tumoral pour notre lame 
- `LABEL_LIT` : est le chemin d'acces du dossier contenant les tuiles du lit tumoral pour notre lame
- `RESULT_DIR` : est le chemin du dossier où nous voulons enregistrer les résultats 

### Les données de sortie : 

Le programme retourne la lame en **.png** avec les : 
- `TP` en **vert**
- `FP` en **rouge**
- `FN` en **bleu**

<div id='2'/>

# Calcul des métriques

Le calcul des métriques est automatiquement effectué par les différents modèles en prenant en compte les pixels. Seulement, dans le cadre de notre projet, nous voulons identifier les cellules tumorales. Savoir que la cellule est correctement identifiée à 80% ne nous importe pas ; il nous faut seulement savoir si oui ou non la cellule est bien identifiée.

Nous utilisons alors `Stradist` qui permet d'identifier les noyaux des cellules et ainsi changer notre calcul des métriques.

Nous utilisons tout d'abord le code [compter_noyauxV2.py](#stardist) afin de pouvoir enregistrer dans un `fichier CSV` les coordonnées des noyaux d'une lame. Ensuite, nous utilisons [identifier_noyaux.py](#metric) qui permet de savoir si les coordonnées enregistrées sont dans la zone segmentée comme tumorale ou non, et ainsi calculer les métriques.   

*********
<div id='stardist' />

Le code `compter_noyauxV2.py`: 

### Les données d'entrées : 

- `LAME_DIR` est le chemin du dossier où se trouve les lames entieres 
- `LABEL_DIR` est le chemin du dossier contenant les tuiles rangées par lame
- `RESULT_DIR` est le cehmin du dossier où enregistrer les csv

### Les données de sorties : 

Le code retourne un CSV par lame contenant les coordonnées du centre du noyau ainsi que les coordonnées du polygone formant le tour du noyaux.


### Information sur le code 

Le code `compter_noyauxV2.py` est optimal et devra être effectué sur toutes les nouvelles lames scannées pour ensuite avoir les coordonnées enregistrées en cas de besoin.

J'ai fait tourner ce code sur l'ensemble des lames Néoadjuvantes et adjuvantes.

*********
<div id='metric' />

Le code `identifier_noyaux.py` :

### Les données d'entrées : 

- `LABEL_DIR` : chemin du dossier contenant les tuiles avec les labels rangées par lame 
- `label_nom` : liste contenant le nom des lames où nous voulons calculer les métriques 
- `PREDICT_DIR` : chemin du dossier contenant les tuiles predites 
- `CSV` : chemin du dossier où sont enregistré les csv contenant les coordonnées des noyaux 

### Les données de sorties : 

Le code affiche pour chaque lame et pour chaque dossier testés les métriques : `Accuracy`, `Sensibilité`, `Spécificité` et `PPV` ainsi que la table de contingence.


### Informations sur le code : 

- Les lignes **193** à **199** créent la liste `label_nom` permettant d'accéder au bon dossier pour chaque lame. Cela peut varier en fonction de vos dossiers.
- La variable `predict_nom` à la ligne **205** spécifie si nous voulons effectuer la visualisation pour plusieurs fichiers résultat en même temps.

<div id='3' />

# Probabilité prédictions

Afin d'éviter les erreurs de notre modèle nous pouvons évaluer les performances sur la moyennes des prédictions. 

Pour cela j'ai re extrait les tuiles de nos lames tests afin que chaque partis soit prédite 4 fois. Puis j'ai lancer la prédictions avec le code `training_model_segmentation.py`. 

On a remarqué que nos visualisations était largement plus satifaisante mais nos métriques restent équivalentes. J'ai ainsi ensuite utilisé le code de raphael `segmentation_prediction.py`

<div id ='segpred'/>

### Les données d'entrée : 

- `DATA_DIR`: Localisation des tuiles (mettre le fichier)
- `PREDICT_DIR` : Nom du dossier avec les prédictions
- `PREDICTION_SEQUENCE`: Chemin du dossier du modèle pour lesquels on fait les prédictions

### Les données de sortie : 

Le code écrit dans le dossier du modèle les prédictions dans un nouveau dossier qui a le nom donnée dans prédict_dir des tuiles en niveau de gris représentant la probabilité de prédiction pour la classe.

### Information sur le code : 

Nous pouvons ensuite utiliser le code [`unpatchify_mask.py`](#unpatchify) pour visualiser les résultats obtenues en nuance de gris. 

On remarque que les résultats obtenues restent équivalent à ce obtenus directement avec `training_model_segmentation.py`.  