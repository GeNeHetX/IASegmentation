import os
import re 
import pandas as pd
import cv2
from PIL import Image
import statistics
from tqdm import tqdm
from openpyxl import load_workbook
import csv
import numpy as np
from PIL import Image, ImageDraw
import json

# Récupere information de la tuile
def selection_tuile(tuile):
    
    resultats = re.search(r'\[d=(\d+),x=(\d+),y=(\d+),w=(\d+),h=(\d+)\]', tuile)
    if resultats is not None:
        d= int(resultats.group(1))
        x = int(resultats.group(2))
        y = int(resultats.group(3))
        w = int(resultats.group(4))
        h = int(resultats.group(5))
    else:
        resultats = re.search(r'\[x=(\d+),y=(\d+),w=(\d+),h=(\d+)\]', tuile)
        if resultats is not None:
            d=1
            x = int(resultats.group(1))
            y = int(resultats.group(2))
            w = int(resultats.group(3))
            h = int(resultats.group(4))

    return d,x, y, w,h

#Recuperer contour du mask 
def annot(LABEL_DIR,tuile):

    chemin=os.path.join(LABEL_DIR,tuile)
    image = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)

    # Appliquer une binarisation (convertir en noir et blanc)
    _, seuil = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Trouver les contours dans l'image binarisée
    contours, _ = cv2.findContours(seuil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Afficher les contours sur une copie de l'image d'origine
    image_contours = image.copy()
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

    image_pil = Image.fromarray(image_contours)

    new_size = (2048, 2048)  # Adjust the new size as needed
    resized_img2 = image_pil.resize(new_size)

    return resized_img2

# Création du nouveau label
def creation_mask(DOSSIER,tuile,df,RESULT):

    # Avec les centroides
    #Cree image blanche
    d,x_min, y_min, width,height=selection_tuile(tuile)

    image_data = np.ones((height, width, 3), dtype=np.uint8) * 255 
    image = Image.fromarray(image_data)
    
    # Récupere coordonnée présente uniquement sur nos tuiles
    filtre = pd.DataFrame(columns=['Y', 'X','Polygone(Y,X)'])
    for index, row in df.iterrows():
        
        if (row['X'] > x_min) and  (row['X'] < x_min + width):
            if (row['Y'] > y_min) and  (row['Y'] < y_min + width):
                nouvelle_ligne = {'Y': row['Y'], 'X': row['X'], 'Polygone(Y,X)': row['Polygone(Y,X)']}
                # Ajout de la nouvelle ligne à la fin du DataFrame
                filtre.loc[len(filtre)] = nouvelle_ligne
    
    #filtre = df[(df['X'] > x_min) & (df['X'] < x_min + width)& (df['Y']<y_min)&(df['Y']>y_min+height)]
    #Récupere tour du mask
    mask=annot(DOSSIER,tuile)

    for index, row in  filtre.iterrows():

        # Récupere coordonnées par rapport à la tuile et non la lame
        x=row['X']-x_min
        y=row['Y']-y_min
        polygone=row['Polygone(Y,X)']

        # Vérifie si le noyaux et dans la partie tumoral
        valeur_pixel=mask.getpixel((x,y))

        if valeur_pixel!=255:
            liste_x=[]
            liste_y = []
            polygone_list = json.loads(polygone)
            pol=np.array(polygone_list)
            # Coordonnée de Stardist en [Y,X]
            x_pol=pol[1]
            y_pol=pol[0]

            # Récupere coordonnées polygone par rapport à la tuile et non la lame
            for w in range(len(x_pol)):
                liste_x.append(x_pol[w]-x_min)

            liste_y=[]
            for z in range(len(y_pol)):
                liste_y.append(y_pol[z]-y_min)
            
            x_modif=np.array(liste_x)
            y_modif=np.array(liste_y)
            # Dessiner un polygone sur l'image
            draw = ImageDraw.Draw(image)
            polygon = list(zip(x_modif,y_modif)) # Former une liste de tuples (x, y)
            draw.polygon(polygon, fill="red")

    #Enregistrement du nouveaux label
    image_resized2 = image.resize((512, 512))
    image_resized2.save(RESULT+tuile, "PNG")


def main():
    #Dossier contenant les tuiles avec les labels 
    INIT_DIR='/'
    # CSV composer des coordonnées des noyaux
    CSV='/'

    TEST=False

    nom_ref=''
    if TEST :
        for lame in tqdm(os.listdir(INIT_DIR)):
            DOSSIER=INIT_DIR+lame+'/label/'
            RESULT=INIT_DIR+lame+'/label_cell/'
            if not os.path.exists(RESULT):
                os.makedirs(RESULT)

            for tuile in os.listdir(DOSSIER):

                chemin_csv=CSV+lame+'.csv'
                creation_mask(DOSSIER, tuile,chemin_csv,RESULT)
        
    else : 
        DOSSIER=INIT_DIR+'label/'
        RESULT=INIT_DIR+'label_cell/'

        if not os.path.exists(RESULT):
            os.makedirs(RESULT)

        for tuile in tqdm(os.listdir(DOSSIER)):
            nom=tuile.split(' ')[0]
            if nom==nom_ref:
                creation_mask(DOSSIER, tuile,df,RESULT)
            else : 
                chemin_csv=CSV+nom+'.csv'
                df=pd.read_csv(chemin_csv)
                creation_mask(DOSSIER, tuile,df,RESULT)
            nom_ref=nom

if __name__ == '__main__':

    main()