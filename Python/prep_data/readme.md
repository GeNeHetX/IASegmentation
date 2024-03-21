# Préparation des données

Le processus pour préparer les données et obtenir les tuiles pour les algorithmes suit les étapes suivantes : 
- [VALIS : Alignement des images HES et IHC](#alignement)
    - [register_valis.py](#register)
- [QuPATH : Identification des cellules épithéliales](#pixelclassifier)
    - [Installation de QuPath](#instal)
    - [Initialisation de QuPath](#init)
        - [Pixel_Classifier.json](#pixel)
- [VALIS/ QuPATH/ PYTHON : Transferer les annotations](#annotation)
    - [alignement_annot_valis.py](#align)
    - [aggrandir_annote.py](#agg)
- [QuPATH : Extraction des tuiles](#tuile)
    - [export_tile_classification.groovy](#groovy)
    - [export_tile_segmentation.groovy](#groovy)
- [PYTHON : Nettoyage des données](#nettoyage)
    - [preparation_donner.ipynb](#prep)
    - [Tips](#tips)
- [PYTHON : Création de la base de donnée](#bdd)
    - [creation_dataset.ipynb](#bdd_seg)
    - [classifier_label.ipynb](#class_label)
    - [creation_dataset_classification.ipynb](#bdd_class)
- [ PYTHON : Création des nouveaux labels](#7)
    - [modification_mask.py](#modifmask)

<div id='alignement'/>

# Alignement des images HES et PanCK 


Pour pouvoir aligner les images HES et IHC on utilise le code `register_valis.py`

`Valis` utilise un docker il faut donc tout d'abord vérifier qu'il soit bien activé en effectuant la commande suivante sur le terminal :
```shell
docker
```

Si la commande `docker` n'est pas reconnue, vérifiez que votre Docker Desktop est bien installé et actif

<div id='register'/>

## Explication sur l'utilisation du code `register_valis.py` :
### Les données d'entrée :

- `slide_src_dir` : est le chemin d'accès du dossier où se trouvent les deux images .svs à aligner. (Attention qu'il n'y ait bien que les deux images de type SVS et rien d'autre)
- `results_dst_dir` : est le chemin du dossier où vous voulez enregistrer les images alignées 
- `target_img_f` : est le chemin de l'image utilisée comme référence lors de l'alignement

### Les données de sortie :

Le code retourne les deux images alignées dans le dossier mis en chemin dans `results_dst_dir` avec le format .ome.tiff

### Information sur le code :

Il faut lancer le programme sur le terminal avec la commande suivante : 
``` shell 
sudo docker run --memory=20g  -v $(pwd):$(pwd) cdgatenbee/valis-wsi:latest python3 /mnt/chemin/acces/au/fichier/register_valis.py
```
<div id='pixelclassifier'/>

# Identification des cellules épithéliale 

Pour pouvoir effectuer un pixel classifier sur une image, il faut utiliser `QuPath`.

QuPath est un logiciel qui évolue beaucoup et change radicalement d'une version à l'autre. Toutes mes explications s'appliquent à la version `0.4.4`.

<div id='instal'/>

## Installation de QuPath : 

* Windows :

Ce logiciel est vraiment très simple à installer !

Aller sur le site suivant : https://qupath.github.io/  (s’adapte au système d’exploitation)

Configuration du logiciel : 

* Installation sur Ubuntu :

La première étape est la même. Ensuite, ouvrir le terminal et exécuter les commandes suivantes :

 &rarr; cd/Downloads/QuPath/bin
 &rarr; sudo apt update
 &rarr; apt list --upgradable
 &rarr; chmod +x QuPath
 &rarr; ./QuPath

<div id='init' />

## Initialisation de QuPath :

- Pour créer un nouveau projet sur QuPath, il faut créer un dossier vide sur l'ordinateur qui sera le fichier source de votre projet. 

- Ensuite allez sur l'application QuPath, selectionnez :
``` shell
File --> Project--> Create project
```
puis indiquer le fichier vide que vous venez de créer 

- Importez ensuite les images que vous souhaitez en glissant l'image directement sur la fenêtre de QuPath

- Double-cliquez ensuite sur l'image souhaitée sur QuPath et indiquez sur le popup le type d'image (HES correspond à HE, IHC correspond à H- DAB)

<div id='pixel'/>

### Pixel Classifier 


Pour pouvoir annoter les cellules d'une zone spécifique de la lame : 

- Téléchargez le fichier `Pixel_Classifier.json`
- Selectionnez la zone souhaitée sur la lame avec l'aide des différents outils présents en haut à droite
- Puis appuyez sur 
``` shell
Classify --> Pixel classification --> Load pixel classifier 
```
- Importez ensuite le model à l'aide des trois petits points et selectionnez `Import from files`. Un popup va apparaitre vous demandant si vous voulez copier le modèle dans le projet, selectionnez oui pour ne pas avoir à reimporter le model à chaque utilisation. 
- Selectionnez sur le menu roulant de `Choose model` le fichier `Pixel_Classifier.json`, puis validez en appuyant sur `Create objects`
- Un popup va apparaitre, selectionnez bien `Current selection` puis appuyez sur `OK`, rappuyez sur `OK` sur le popup suivant
- Une fois que les cellules de la zone selectionnée sont devenues grasse vous pouvez fermer la page `Load pixel classifier`

Les annotations sont maintenant effectuées, comme vous pouvez le voir dans l'onglet `Annotations`.

### Enregistrement des annotations

Dans le cadre de nos données, nous effectuons ce pixel classifier sur les lames IHC. Nous avons ainsi toutes les cellules épithéliales de la zone tumorale annotées. 

Pour enregistrer ses annotations il suffit de double-cliquez sur `Annotations(Tumor)` dans l'onglet `Annotations` puis faire :
``` shell
File --> Exports objects as GeoJSON
```
Selectionnez ensuite `selected objects` validez sur `OK` puis indiquez le chemin où enregistrer le fichier geojson

J'ai déja enregistré l'ensemble des annotations effectuées par les anapaths, sur les images Néoadjuvantes et adjuvantes

<div id='annotation' />

# Transferer les annotations

Une fois que la detection des cellules tumorales sur les lames IHC est effectuée, il faut transférer ses annotations sur la lame HES. 

<div id='geojson' />

Pour transférer ses annotations il faut :

- Importer l'image HES dans le projet QuPath ( Faites attention qu'il s'agit bien de la même coupe que celle où le geojson IHC a été enregistré)
- Double-cliquez sur l'image pour l'ouvrir dans QuPath et selectionnez le type de l'image (HE) sur le popup
- Puis selectionnez : 
``` shell
File --> Import objects from file
```
et selectionnez le fichier geojson de l'IHC

Si les images IHC et HES sont de même dimension et correctement alignées, l'annotation devrait parfaitement tomber sur l'image. Si ce n'est pas le cas nous pouvons corriger cela soit directement sur QuPath soit avec Valis soit à l'aide de code python. 

<div id='qupath' />

### **QuPath**
*******
Si l'annotation est de bonne taille mais mal possitionnée nous pouvons régler cela via QuPath. 

- Selectionnez l'onglet `Annotations` et double-cliquez sur `Annotation (Tumor)`
- Appuyez ensuite sur les touches `CTRL` + `L`, un popup va s'afficher 
- Tapez dans `Search all commands` :
``` shell
Transform annotations
```
- Doucle-cliquez dessus, le popup se fermera et un cadre gris apparaitra sur l'image 
- Vous pouvez maintenant à l'aide de la souris bouger l'annotation sur l'image (attention le temps de calcul peut être long il faut donc aller doucement lorsque l'on bouge l'annotation). Vous pouvez faire des modifications plus précises en appuyant sur `Ctrl` et sur les flèches.  
- Pour tourner les annotations il faut appuyer sur `CTRL` + `Shift` + sur les flèches
- Une fois les annotations bien replacées, double-cliquez en dehors du carré gris.
- Un popup apparait appuyez sur `All objects`

Vos annotations sont maintenant bien placées vous pouvez enregistrer le geojson comme montré [précédemment](#geojson).  


### **Valis** 
*****************
Si l'annotation n'est pas de bonne taille ou si elle est mal positionnée on peut utiliser Valis. 

Valis permet d'aligner les annotations sur l'image souhaitée. Il est ainsi utile si l'alignement entre HES et IHC n'a pas été effectué avant les annotations où si les deux images n'ont pas les même dimensions (une en 20x l'autre en 40x par exemple).
<div id='align'/>

Pour cela on utilise le code `alignement_annot_valis.py`

`Valis` utilise un docker il faut donc tout d'abord verifier qu'il soit bien activé en effectuant la commande suivante sur le terminal :
```shell
docker
```

## Explication sur l'utilisation du code `alignement_annot_valis.py` :

### Les données d'entrées :

- `slide_src_dir` : est le chemin d'accès du dossier où se trouvent les deux images .svs à aligner. (Attention qu'il n'y ai bien que les deux images de type SVS et rien d'autre)
- `results_dst_dir` : est le chemin du dossier où vous voulez enregistrer les images alignées 
- `annotation_img_f` : est le chemin de l'image à aligner
- `target_img_f` : est le chemin de l'image utilisée comme référence lors de l'alignement
- `annotation_geojson_f`: est le chemin du geojson de l'image à aligner 
- `warped_geojson_annotation_f`: est le chemin d'accès au geojson qui sera enregistré

### Les données de sortie : 

Le programme retourne un ensemble de dossier, il nous suffit uniquement de récuperer le géojson aligné 

### Information sur le code : 

Il faut ensuite lancer le programme sur le terminal avec la commande suivante : 
``` shell 
sudo docker run --memory=20g  -v $(pwd):$(pwd) cdgatenbee/valis-wsi:latest python3 /mnt/chemin/acces/au/fichier/alignement_annot_valis.py
```

J'ai utilisé ce code pour l'alignement des annotations sur les données néoadjuvantes, cependant, je n'ai pas trouvé les résultats satisfaisants. Les annotations étaient bien redimensionnées, mais non parfaitement alignées ; il fallait les repositionner à la main sur QuPath. J'ai donc préféré effectuer l'ensemble des alignements avec mes codes Python et QuPath.


### **Python** 
**************

<div id='agg'/>

Si l'annotation n'est pas de bonne taille on peut utiliser le code python que j'ai créé `aggrandir_annote.py` :

### Les données d'entrées : 

- `DATA_DIR` est le chemin d'acces où se trouvent le geojsons à redimensionner
- `RESULT_DIR` est le chemin du dossier où vous voulez enregistrer le geojson redimensionné
- `fichier` est le nom du geojson à modifier 
- `scaling_factor` est le coefficient de grossissement. Pour mes différents fichiers, les deux facteurs étaient **0,531** s'il fallait rétrécir l'annotation et **1,883** s'il fallait l'agrandir. 

### Les données de sortie : 

Vous obtenez le geojson redimensionné. Vous pouvez maintenant transférer les annotations sur votre image HES en utilisant [QuPath](#qupath). Il faudra surement repositionner les annotations sur QuPath avant de pouvoir enregistrer le geojson. 
<div id='tuile' />

# Extraction des tuiles 
<div id='groovy'/>

Une fois les lames HES correctement annotées, nous pouvons extraire les tuiles. J'ai deux codes pour effectuer cela `export_tile_classification.groovy` et ` export_tile_segmentation.groovy`. Les deux codes s'effectuent de la même manière, se sont les classes selectionnées qui varient : la segmentation prend les cellules tumorales identifiées par le pixel classifier tandis que la classification utilise le lit tumoral identifié par les anapaths. 

## Pour exporter les tuiles sur QuPath: 
- Seletionnez 
``` shell 
Automate --> Show scritp editor
```
- Une fenetre s'affiche, ouvrez maintenant le script groovy souhaité en faisant : 
``` shell
File --> Open 
```  

Vous pouvez maintenant exécuter le code en appuyant sur `Run`. Si vous voulez faire tourner le code pour plusieurs images, appuyez sur les trois petits points à côté de `Run` et sélectionnez `Run for project`. Vous sélectionnez ensuite les lames qui vous intéressent et validez avec `OK`.

### Les données d'entrées : 

Si l'on regarde le script groovy de plus près, les seules informations qui dependent de nos données sont : 

- `classNames` représente le nom des différentes classes d'interêt ( ligne qui varie entre `export_tile_classification.groovy` et ` export_tile_segmentation.groovy`)
- `downsample` représente le grossisement. Celui-ci varie entre les lames de 20 ou 40x. On veut des tuiles de 10x donc on met un downsample de 2 pour les lames de 20x et un downsample de 4 pour les lames de 40x. ( Pour connaitre la résolution de la lame il faut aller dans l'onglet `Image` et c'est écrit à ` Magnification`)
- `patchSize` représente la taille des tuiles créées, on le met à 512 
- `pixelOverlap` représente la supperposition entre les différentes tuiles initialement à 128

### Les données de sortie : 

Une fois l'ensemble des paramètres bien définit, on peut appuyer sur `Run`. Les résultats seront enregistrés dans le dossier associé aux projets QuPAth dans le sous dossier `tiles`. 


<div id='nettoyage' />

# Nettoyage des données 

<div id='prep'/>

Une fois nos tuiles obtenues il faut préparer les données. J'ai ainsi créé un code `preparation_donner.ipynb` capable de supprimer les images qui ne sont pas du tissu. 

### Les données d'entrées :

- `DATA_DIR` est le chemin du dossier contenant les dossiers des tuiles rangées par lame à préparer
- `IMAGE_DIR` est le chemin du dossier où sont enregistrées les images des lames entières ( Important pour l'utilisation de stradist)
- `SUP_DIR` est le chemin du dossier qui va contenir les tuiles à supprimer
- `LABEL_DIR` est le chemin du dossier contenant les labels des tuiles rangées par lame  

### Les données de sortie : 

Les tuiles considérées comme "limite" par le programme seront copiée dans le fichier `SUP_DIR` avant d'etre supprimées avec leur label associé de `DATA_DIR` et `LABEL_DIR`.

### Information sur le code : 

Le code fonctionne en trois parties qui peuvent etre effectuées séparement. 
- La premiere étape permet une première visualisation des tuiles en fonction de la coloration. En dessous d'un certain seuil de couleur, les images sont copiées dans le fichier `SUP_DIR`
- La deuxième étape utilise stradist. Il va compter le nombre de noyaux sur les tuiles enregistrer dans le dossier `SUP_DIR`. En dessus d'un certain nombre de noyaux, la tuile est enlevé du fichier `SUR_DIR`. (Cette étape est plus longue mais les résultats sont plus efficace) 
- La troisème étape supprime les tuiles et les labels. Le code va parcourir le fichier `SUP_DIR` et pour chaque tuile présente dans le fichier, il va les supprimer dans `DATA_DIR` ainsi que le label associé à la tuile dans `LABEL_DIR`. Attention à bien vérifier les tuiles présentes dans `SUP_DIR` avant de lancer le programme pour ne pas effacer une tuile importante. Pour empêcher sa suppression il suffit de l'enlever du fichier `SUP_DIR`.

<div id='tips' />

## TIPS :

Pour utiliser ce code ils faut que les tuiles soit rangées par lame. Si ce n'est pas le cas vous pouvez effectuer le code suivant qui le fera automatiquement : 

``` shell
for img in tqdm(os.listdir(DATA_DIR)):
    img_dir=os.path.join(DATA_DIR, img.split(" ")[0])
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    shutil.move(os.path.join(DATA_DIR, img), img_dir)
```
`DATA_DIR` correspond au chemin du dossier où ce trouve toutes vos tuiles à ranger. 

<div id='bdd' />

# Création de la base de donnée

Une fois toutes nos données prêtes, on peut créer les bases de données. J'ai effectué deux datasets différents : un pour la segmentation `creation_dataset.ipynb` et un pour la classification `creation_dataset_classification.ipynb`.

## **Pour la segmentation** :
<div id='bdd_seg'/>

### Les données d'entrées : 

- `LAME_DIR_F`/`LAME_DIR_G`  est le chemin d'accès aux tuiles rangées par lame 
- `LABEL_DIR_F`/`LABEL_DIR_G` est le chemin d'accès aux labels rangées par lame 
- `Test` est le chemin où l'on veut enregistrer nos lames tests
- ` Train` est le chemin où l'on veut enregistrer nos lames train
- ` Valid` est le chemin où l'on veut enregistrer nos lames valid

### Les données de sorties : 

Le code retourne trois dossier : `Train`, `Valid` et `Test` contenant les images et les labels  

### Informations sur le code : 

Le programme va enregistrer 10% des lames dans le dossier `Test`. Sur les 90% des lames restantes le code selectionne aléatoirement 80% des tuiles qu'il enregistre dans le dossier `Train` et 20% dans le dossier `Valid`.  

## **Pour la classification** : 
<div id='class_label' />


Avant de pouvoir utiliser `creation_dataset_classification.ipynb` il faut tout d'abord classer les images entre lit et background. Pour cela j'ai créé le code `classifier_label.ipynb`.

### Les données d'entrées : 

- `LAME_DIR` est le chemin d'acces aux tuiles rangées par lame 

### Les données de sorties: 

Le code retourne les tuiles rangées entre deux dossiers : `Lit` et `Background` pour chaque lame. Les labels sont supprimés car ils ne sont pas utiles dans la classification. 

### Informations sur le code : 

Le code va utiliser les labels des tuiles pour pouvoir trier s'il s'agit d'une tuile située dans le lit tumoral ou non. Il crée les deux dossiers et supprime les anciens.
****
<div id='bdd_class' />

Nous pouvons maintenant utiliser `creation_dataset_classification.ipynb` : 


### Les données d'entrées : 

- `LAME_DIR_F`/`LAME_DIR_G` est le chemin d'accès aux tuiles triées en lit et background rangées par lame. 
- `Test` est le chemin où l'on veut enregistrer nos lames tests
- ` Train` est le chemin où l'on veut enregistrer nos lames train
- ` Valid` est le chemin où l'on veut enregistrer nos lames valid 

### Les données de sorties : 

Le code retourne trois dossiers : `Train`, `Valid` et `Test` contenant les tuiles séparées

### Informations sur le code : 

Le programme va enregistrer 10% des lames dans le dossier `Test`. Sur les 90% des lames restantes le code selectionne aléatoirement 80% des tuiles qu'il enregistre dans le dossier `Train` et 20% dans le dossier `Valid`. Il est important de noter qu'il y a autant de lame classifiées lit que de lame classifiées background, afin de pouvoir entrainer notre model de classification correctement sans la sureprésentation d'une classe.   

<div id='7'/>

# Création des nouveaux labels

Après de nombreux tests sur nos données, nous nous sommes interrogés sur l'efficacité de nos labels. Nous avons ainsi modifié les labels formés par le PanCK en un label composé uniquement des noyaux tumoraux. Pour ce faire, nous utilisons le code `modification_mask.py`.

<div id='modifmask'/>

### Les données d'entrée :

- `INIT_DIR` : c'est le chemin du dossier dont nous voulons modifier les labels 
- `CSV` : chemin où sont enregistrés les CSV regroupant les coordonnées des noyaux pour chaque lame (CSV obtenu avec le code `compter_noyauxV2`)
- `TEST` : variable binaire informant si nous modifions le masque du dossier `Test`

### Les données de sortie :

Le programme crée un dossier `label_cell` contenant l'ensemble des nouveaux labels.

### Informations sur le code :

Une fois l'ensemble des labels récupérés, nous pouvons re-entraîner notre programme `training_model_segmentation.py` sur nos nouveaux labels.

Extraire l'ensemble des tuiles a demandé 2 semaines d'exécution. Je n'ai ainsi pas eu le temps de lancer le programme sur l'ensemble des données mais uniquement sur un sous-ensemble du dossier `Train`. Voici les résultats que nous avons obtenus :
- **Précision** = 0.9623
- **Sensibilité** = 0.3607
- **Spécificité** = 0.9999
- **VPP** = 0.9946

Il faudra ainsi relancer le modèle sur l'ensemble de nos nouveaux labels ainsi qu'effectuer différents tests.