# Détection des cellules tumorales dans le pancréas

Vous souhaitez créer un modèle de segmentation à partir de lames histologiques ? 
Ici, vous trouverez les étapes à suivre afin de créer votre modèle de Deep Learning pour la segmentation de vos images de lames entières à haute résolution.
Dans notre cas, nous avons appliqué ces outils afin de détecter les cellules tumorales, les nerfs et les îlots sur des coupes histologiques du pancréas.

*******
Table des matières  
 1. [Outils utilisés](#outils)
 2. [Comment installer QuPath ? ](#QuPath1)
 3. [Comment utiliser QuPath ?](#utiliserQuPath)
     * [Créer un projet](#projet) 
     * [Pixel Classifier](#annotations)
     * [Extraction des patchs](#patchs)
 4. [Comment installer DeepMIB ? ](#DeepMIB)
 5. [Comment utiliser DeepMIB ?](#DeepMIB2)
     * [Preprocess](#preprocess)
     * [Train](#train)
     * [Prediction](#prediction)
 6. [Transférer les annotations des prédictions DeepMIB sur QuPath](#QuPath2)

*******
<div id='outils'/> 

## Outils utilisés :
- QuPath
- DeepMIB

<div id='QuPath1'/> 

## Comment installer QuPath ? 

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

<div id='utiliserQuPath'/> 

## Comment utiliser QuPath ?

<div id='projet'/> 

1. Créer un projet

Créer un dossier vide sur votre ordinateur. Ensuite, sur QuPath appuyer sur « Project » puis sur « Create project ».

2. Ajouter les images

Dans un premier temps, sélectionner le dossier vide. Puis, sélectionner les images que vous voulez inclure dans votre projet et glissez-les dans la page 
QuPath. Votre projet est créé !

Vous pouvez également cliquer sur « Add images » et sélectionner les images souhaitées.

Vous pouvez visualiser toutes les images présentes dans votre projet à gauche de la page.

3. Définir le type de l’image 

La première fois que vous affichez une image, il vous ait demandé de définir le type de l’image étudiée : HE, DAB, Fluorescence etc.

<div id='annotations'/> 

4. Faire les annotations 

Si vous n'avez pas de région d'intérêt annotée :

* Pixel Classifier :

Tout d’abord, vous pouvez choisir des classes déjà existantes ou bien les créer, définir leur nom et leur couleur (onglet Annotations). Puis, réaliser quelques 
annotations à l’aide des Tools comme présenté précédemment. Lorsque les annotations sont réalisées, il faut les associer à la classe correspondante. Pour cela, il faut 
appuyer sur l’annotation, puis sur la classe et enfin sur « set class ».

Maintenant vous pouvez démarrer votre Pixel Classifier. 

Afin de créer un Pixel Classifier, il faut se rendre dans « Classify », « Pixel classification » et «Train pixel classifier».

Vous trouverez ci-dessous les modifications à réaliser :
 
Dans « Classifier », soit vous laissez « artificial neural network » ou mettre « Random Tree ». Ensuite, appuyer sur « edit » de la ligne « Features » et choisir « 
Hematoxylin, DAB et residual » pour « Channels », choisir au moins trois valeurs dans « Scales », enfin dans « features » choisir « gaussian, laplacian of gaussian ».
Par ailleurs, nous pouvons visualiser une région de notre image (à droite). Lorsque ces modifications sont réalisées, on peut alors lancer la prédiction en appuyant 
sur « Live prediction ».

Toute l’image est annotée et nous avons une idée de la proportion de région tumorale ou non. En passant la souris sur le camembert le pourcentage 
s’affiche. L'idéal est d'avoir des proportions similaires entre les différentes classes.
Enfin, vous pouvez enregistrer votre Pixel classifier.

Si vous avez une région d'intérêt :

Une autre méthode qui peut potentiellement donner de meilleurs résultats. Tout d’abord, sélectionner une partie de la ROI de plusieurs images (peu 
importe la taille) et les mettre sous une nouvelle classe que vous aurez créé au préalable. Ensuite, vous appuyez sur « Classify », « training images », 
« Create training image » et cliquer sur la classe qui contient vos échantillons (dans l'onglet cliquable) et appuyer sur « Ok ».

Une « image » est créée avec l'ensemble de vos échantillons. Vous pouvez alors effectuer toutes les étapes du pixel classifier dessus, l'enregistrer et l'appliquer aux 
autres images. Pour l'appliquer à toutes les images utiliser le script groovy « PixelClassifier.groovy »

Afin d’appliquer le pixel classifier à d’autres images ou bien le réappliquer à l’image après avoir fermé l’onglet, utiliser «load pixel classifier». 
Sélectionner le modèle que vous souhaitez et il sera automatiquement appliqué à l’entièreté de l’image.

<div id='patchs'/> 

4. Extraire les annotations 

Si vous souhaitez appliquer les annotations à une ROI, sélectionner l'annotation correspondante à celle-ci, rendez vous dans « Automate », « Show script 
editor ». Vous pourrez enfin appliquer cette ligne de code *createAnnotationsFromPixelCLassifier(«nom de votre modèle», 50.0, 50.0, « SELECT_NEW »)*.

Sinon, lancer le Pixel Classifier à l'aide du « load pixel classifier". D’abord, sélectionner le modèle correspondant au Pixel Classifier enregistré. Les 
annotations vont donc être appliqué à toute l’image. Ensuite, sélectionner la ROI en cliquant sur l'annotation. 

Afin de sauvegarder les annotations, sélectionner « Create objects », un onglet s'affiche, il faut garder « Current selection » et appuyer sur « OK ». Un 
nouvel onglet s’affiche, ne rien changer et appuyer sur « OK ».

On remarque que les annotations sont devenues grasses, la sauvegarde a été réalisé ! L’onglet « Annotations » le confirme.
On peut enfin fermer l’onglet « Load pixel classifier ».

Toutes ces étapes sont réalisées sur les IHC, il faut donc transférer ces annotations sur l’image HES.
Pour cela, nous allons les transférer à l’aide d’un script : BATCH_AFFINETRANSF_ANNOTCOPY_1.groovy et BATCH_AFFINETRANSF_ANNOTCOPY_2.groovy, vous devez 
appliquer les deux parties sur le projet contenant les images sur lesquelles vous voulez appliquer les annotations et, après avoir run, sélectionner le 
projet d'où proviennent les images.

Nous pouvons éventuellement exporter les annotations sous forme de fichier GeoJson, soit manuellement soit à l'aide du script GeoJson.groovy (appuyer sur 
run for project).

Une fois les fichiers GeoJson exportés, nous pouvons transférer les annotations en glissant le fichier GeoJSON sur l’image identique en HES.

Enfin, pour pouvoir extraire les masques et les tuiles correspondantes des images HES, nous utilisons le script : exportTiles.groovy
Pour l'appliquer, dirigez vous dans « Automate », « Show script editor » et importer le script que vous pourrez run.

Dans la partie des commentaires associés au script, il vous sera indiqué le nombre de tuiles/masques exportés.

<div id='DeepMIB'/> 

## Comment installer DeepMIB ? 

DeepMIB : version 2.84 (9/12/2022)

Aller sur le site suivant : http://mib.helsinki.fi/

Dans le menu « Downloads », installer la version la plus récente. Ensuite, suivre les étapes dans « Installation instructions ».

* Windows :

Après avoir installer le logiciel, vous n'aurez plus qu'à lancer le logiciel via l'écran de démarrage.
Un terminal s'affichera et quelques minutes plus tard la page d'accueil de DeepMIB.

* Ubuntu :

Sur le terminal, glisser le fichier run_MIB.sh présent dans le dossier application et le fichier v284 : 
'/home/Downloads/MATLAB/MIB_deploy_2701/application/run_MIB.sh' '/home/Downloads/MATLAB/Runtime/v984'

<div id='DeepMIB2'/> 

## Comment utiliser DeepMIB ?

Il existe une vidéo tutoriel : https://www.youtube.com/watch?v=9dTfUwnL6zY&t=519s.
Dirigez-vous dans le menu « Tools » puis « Deep learning segmentation ».
Une autre page s’affichera. Celle-ci possède plusieurs onglets : directories and preprocessing, Train, Predict et Options.

<div id='preprocess'/> 

- Preprocess :

Dans le premier onglet, il faut ajouter les chemins vers les patchs pour le train (les dossiers doivent être nommé Images et Labels pour le train et le 
test), puis ceux pour le test et enfin un dossier vide pour les résultats. ATTENTION ! Les patchs du dossier d'entraînement doivent être différent de ceux du dossier 
test.

Dans notre cas, les images sont des tiff et les masques sont des png. Dans les cases « extension » du train et du test, il faut choisir « TIF » ATTENTION pas « TIFF ».
Si la case « Single MIB model file » est cochée, décocher-la et indiquer « PNG » dans la case « Model extension ». Pour la case « Mask extension », laisser « MASK ».

Ensuite, cocher les cases « compress processed images», « compress processed models » et « use parallel processing ».
Dans « fraction of images for validation », indiquer une valeur cohérente avec la taille de votre dataset. Mettre 0 dans « random generator seed ».
Enfin, laisser « training and prediction » et appuyer sur preprocess.


Le preprocess va permettre de créer les dossiers de trainingset et validationset dans le dossier « résultats ».
Une fois, le preprocess terminé vous pouvez passer à l’onglet « train ».

<div id='train'/> 

- Train :

Dans l’en-tête, vérifier que le workflow indiqué est « 2D semantic » et que l’architecture choisie est « DeepLabV3 Resnet18 ».
Ensuite, dans « input patch size » mettre 512 512 1 3, si vos images ont une autre dimension que 512x512 alors indiquer cette valeur à la place. Dans « Network design 
», garder tous les paramètres par défaut et appuyer sur « check network » afin que le réseau de neurones soit généré.

Pour la partie « Augmentation design », garder également les paramètres par défaut.
Enfin, dans la partie « training process design », « Patches per image » mettre 1, « mini batch size » = {4,8,16,32} et random seed =0.

Appuyer sur « training » afin de configurer quelques hyper-paramètres. Le solverName doit être « adam ». Ensuite, vous pouvez choisir le nombre d’epoch, modifier 
l’initial learn rate en rajoutant ou enlevant des 0 après la virgule jusqu’à obtenir de bons résultats d'accuracy et d'IoU, indiquer « none » pour « learnRateSchedule 
» et mettre le decay rate of gradient moving average à 0.99. Tous les autres hyper-paramètres doivent être laissé par défaut.
Maintenant, vous pouvez lancer l’entrainement de votre modèle.
 
Une fois celui-ci terminé, passer à l’onglet « prediction ». 

<div id='prediction'/> 

- Prediction :

Si la case « overlapping tiles » est cochée, décocher-la. Pour le « batch size » mettre 32. Vous pouvez lancer la prédiction.
 
Quand la prédiction est terminée, vous pouvez évaluer votre modèle en appuyant sur « evaluate segmentation ». Un onglet s’affiche, vous pouvez mettre directement « ok 
».

Si vous êtes satisfait des résultats, créer un dossier contenant des dossiers Images et Labels sur des lames non annotées ou bien sur lesquelles le modèle n’a pas été 
entrainé* (vous pourrez supprimer le dossier Labels car vous obtiendrez uniquement des images blanches). Créer, ensuite, un nouveau dossier résultats, vous pouvez 
l’appeler « results_prediction », par exemple.

Mettez le chemin du dossier contenant les images à la place du dossier test, et le nouveau chemin du dossier résultats à la place de l’ancien. Vous pouvez relancer la 
prédiction sans effectuer aucun autre changement.

Ensuite, appuyer sur « Load images and models ». Les images s’afficheront sur la page d'accueil. Pour les enregistrer, dirigez-vous dans le menu « models », puis dans 
« save model as ». Il faut d’abord modifier l’extension et choisir tif, puis, changer de dossier. Retourner dans « prediction images » ,puis, aller dans « results 
scores ». Vous pouvez, enfin, enregistrer vos annotations de prédiction.
 
*Pour obtenir ces images, il faut passer par QuPath et utiliser le code exportTiles.groovy
Les modifications à réaliser sont :
- .annotatedTilesOnly(true) devient .annotatedTilesOnly(false)
- def classNames = ["Benign", "Malign"], ne pas oublier de mettre les bonnes classes

<div id='QuPath2'/> 

## Transférer les annotations des prédictions DeepMIB sur QuPath 
 
Tout d’abord, le transfert est possible uniquement avec les versions < QuPath 4.0.0

*Le transfert est également compliqué avec les multiclasses car il faut d’abord extraire l’image entière et non des tuiles. Cependant, plus la résolution des images 
est bonne, plus l’image est grosse. Le problème est que DeepMIB ne peut pas réaliser de prédiction sur des images ayant une taille supérieure à 34kx20k environ.*

Ouvrir le projet QuPath où vous voulez transférer vos annotations, dans le menu « Automate », « Show script editor » copier ou importer le code « importTiles.groovy ». 
Modifier les paramètres « className » et « pathOutput » afin qu’il soit adapté à votre cas. Lancer le code pour tout le projet. Vous n’avez plus qu’à patienter et 
faire corriger les résultats par un/e anatomopathologiste. 

FIN.
