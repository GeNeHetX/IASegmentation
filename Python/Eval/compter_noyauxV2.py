## IMPORT 
import os
import csv
from skimage import io
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# Ajoute les coordonnées de la tuile dans un csv
# Ajoute les coordonnées de la tuile dans un csv
def write_to_csv(RESULT_DIR, coordonnate,polygone,nom,x_min,y_min):

    #Vérifie existance du fichier csv 
    file_exists = os.path.isfile(RESULT_DIR+nom+'.csv')

    with open(RESULT_DIR+nom+'.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:  # Si le fichier n'existe pas, écrire l'en-tête
            writer.writerow(['Y', 'X','Polygone(Y,X)'])

        #Ajout des coordonnées
        i=0
        for data in coordonnate:  
            # Coordonnées mis en fonction de la lame et non de la tuile     
            x=data[1]+x_min
            y=data[0]+y_min

            pol=polygone[i]
            liste_x=[]
            for w in range(len(pol[1])):
                liste_x.append(pol[1][w]+x_min)

            liste_y = []
            for w in range(len(pol[0])):
                liste_y.append(pol[0][w]+y_min)

            writer.writerow([y,x,[liste_y,liste_x]])
            i=i+1

if __name__ == "__main__":

    # Dossier contenant les lames 
    LAME_DIR='/'

    #Dossier contenant les tuiles rangées par lame
    LABEL_DIR='r/'

    #Dossier où enregistrer csv
    RESULT_DIR='/'

    # Si l'on veut eviter une lame spécifique 
    lame_etudier=['17AG06944-15_NADJ02_HES']
    ffx=os.listdir(LABEL_DIR)
    # Initialisation stardist 
    StarDist2D.from_pretrained() 
    HE_model = StarDist2D.from_pretrained('2D_versatile_he')

    cellule_tum_tot=0
    cellule_sai_tot=0
    i=0
    #Passage sur toutes les lames étudiéses
    for lame in ffx:
        if lame not in lame_etudier : 
            nom=lame.split('-')
            nom=LAME_DIR+nom[0]+'-'+nom[1]+'/'+lame+'.svs'
            print(nom)

            print('Image en ouverture')
            HE_img = io.imread(nom)
            print('Image ouverte')

            taille=HE_img.shape
            longueur=taille[1]
            largueur=taille[0]
            
            y_min=0
            y_max=1536
            cellule_tum=0
            cellule_sai=0
            i=0
            #while i<10:
            #Découpage de la lame en tuile 
            while y_min < largueur :

                print("Itération ", i)
                x_min=0
                x_max=1536
                
                #while i<1:
                while x_min<longueur:
                    i+=1
                    roi = HE_img[y_min:y_max, x_min:x_max]
                    
                    #Prediction stardist sur la tuile
                    he_labels, he_details = HE_model.predict_instances(normalize(roi))
                    coordinates = he_details["points"]
                    polygone= he_details['coord']
                    if len(coordinates)!=0:
                        #Enregistrer coordonnées
                        write_to_csv(RESULT_DIR, coordinates, polygone,lame,x_min,y_min)
                    
                    x_min=x_max
                    x_max=x_max+1536
                    if x_max>longueur:
                        x_max=longueur

                y_min=y_max
                y_max=y_max+1536
                if y_max>largueur:
                    y_max=largueur
            
