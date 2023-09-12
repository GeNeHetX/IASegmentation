# Code utilisé dans Qupath

## Export_geojson

Exporte toutes les annotaitons d'un image sous forme de fichier Geojson

### Entrée

- path (str) : Chemin dans lequel doit être mis les geojson. Attention bien mettre un / à la fin.

### Sortie

Enregistre un fichier Géojson donc le nom est celui de la lame dna sle dossier spécifié dans path

## import_transform_geojson

Permet d'importer les prédictions en géojson des scripts python sur notre image.

### Paramètres

- class_tumor (str) : nom de la class dnas lequel on met les cellules tumorales.
- path_dir (str) : Chemin du dossier dans lequel est situé les géojson

### Remarques

Les geojson doivent avoir le même nom que celui de la lame.

Etant donné que le geojson contient les annotations du non tumoral, il fait une inversion des annotations et suppprime ensuite le stroma.
