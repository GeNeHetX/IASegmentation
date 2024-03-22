import os, re
from skimage import io
from PIL import Image
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory("/mnt/d/openslide-win64-20231011/bin"):
        import openslide as ops
else:
    import openslide as ops
import cv2
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
def select_tuile(tuile_lame,x_min,y_min,height):
    tuile_de_la_tuile = []
    for tuile in tuile_lame:
        
        d, x, y, w, h = selection_tuile(tuile)
        if x >= (x_min - (h / 2)) and x <= (x_min + (h / 2)):
            if y >= (y_min - (h / 2)) and y <= (y_min + (h / 2)):
                tuile_de_la_tuile.append(tuile)

    liste=list(set(tuile_de_la_tuile))
    return liste

def supperposition_mask(DIR,tuile_de_la_tuile, white_img, x_min,y_min,height):
    z=0
    for image_dir in tuile_de_la_tuile:
        #print(image_dir)
        z+=1
        #création image png 

        d, x, y, w, h = selection_tuile(image_dir)
        img = Image.open(DIR + image_dir)
        new_size = (height, height)  # Adjust the new size as needed
        resized_img = img.resize(new_size)  # Utilisez img.resize(), pas new_size.resize()
        rgba = resized_img.convert("RGBA")
        datas = rgba.getdata()

        
        newData = [] 
        for item in datas: 
            if item[0] == 255 and item[1] == 255 and item[2] == 255:  # finding black colour by its RGB value 
                # storing a transparent value when we find a black colour 
                newData.append((255, 255, 255, 0)) 
            else: 
                newData.append((255,0,0,64))  # other colours remain unchanged 
        
        rgba.putdata(newData) 
        #rgba.save("transparent_image.png", "PNG") 

        x_tuile=x-(x_min-int(height/2))
        y_tuile=y-(y_min-int(height/2))
        white_img.paste(rgba, (x_tuile,y_tuile), mask = rgba)

        #nom = image_nom.split('.')
        #info = nom[0]
    #white_img.save("test/"+str(x_min)+'_'+str(y_min)+".png")

    roi = white_img.crop((int(height/2), int(height/2), int(height/2) + height, int(height/2) + height))
    roi.save("test/"+str(x_min)+'_'+str(y_min)+".png")
    nom="test/"+str(x_min)+'_'+str(y_min)+".png"
    return nom

def proba_tuile(DIR,tuile_lame,x_min,y_min,height,tuile_lame_prob):

    tuile_de_la_tuile=select_tuile(tuile_lame,x_min,y_min,height)
    image_blanche = Image.new('RGBA', ((height*2), (height*2)), color='white')
    prob_tuile=supperposition_mask(DIR,tuile_de_la_tuile, image_blanche, x_min,y_min,height)
    img = Image.open(prob_tuile)
    width, height = img.size
    image_blanche2 = Image.new('RGBA', (width, height), color='white')
    rgba = img.convert("RGBA") 

    datas = rgba.getdata() 
    newData = [] 
    for item in datas: 
        if item[0] == 255 and item[1] == 255 and item[2] == 255:  # finding black colour by its RGB value 
            # storing a transparent value when we find a black colour 
            newData.append((255, 255, 255, 0)) 
        else: 
            tuile_lame_prob.append(item)
            newData.append((255,0,0,255))  # other colours remain unchanged 
    
    rgba.putdata(newData) 
    
    image_blanche2.paste(rgba, (0, 0), mask = rgba)
    new_size = (512, 512)  # Adjust the new size as needed
    resized_img2 = image_blanche2.resize(new_size) 
    resized_img2.save("test2/"+str(x_min)+'_'+str(y_min)+".png")
     
    return tuile_lame_prob

if __name__ == "__main__":

    # Dossier contenant les lames 
    LAME_DIR='/mnt/d/lame/'

    #Dossier contenant les tuiles rangées par lame
    LABEL_DIR='/mnt/d/dataset_neoadj/segmentation/result_adj_ttneo_25NM_cree_epoch10_10/'

    ffx=os.listdir(LABEL_DIR)

    cellule_tum_tot=0
    cellule_sai_tot=0
    i=0
    l=0
    #Passage sur toutes les lames étudiéses
    for lame in ffx:
        if l==0:

            if 'best' in lame : 
                break 
        
            nom=LAME_DIR+lame+'.svs'
            print(nom)

            print('Image en ouverture')
            HE_img = io.imread(nom)
            print('Image ouverte')

            taille=HE_img.shape
            longueur=taille[1]
            largueur=taille[0]

            DIR=LABEL_DIR+lame+'/tile_prediction_overlay/'
            tuile_lame=os.listdir(DIR)
            d,x, y, w,h =selection_tuile(DIR+tuile_lame[0])

            print(h)

            y_min=0
            y_max=h
            cellule_tum=0
            cellule_sai=0
            i=0
            #while i<10:
            #Découpage de la lame en tuile 

            tuile_lame_prob=[]
            while y_min < largueur :

                print("Itération ", i)
                x_min=0
                x_max=h
                
                #while i<1:
                while x_min<longueur:
                    i+=1
                    roi = HE_img[y_min:y_max, x_min:x_max]
                    tuile_lame_prob=proba_tuile(DIR,tuile_lame,x_min,y_min,h,tuile_lame_prob)
                    
                    x_min=x_max
                    x_max=x_max+h
                    if x_max>longueur:
                        x_max=longueur

                y_min=y_max
                y_max=y_max+h
                if y_max>largueur:
                    y_max=largueur

        ensemble_sans_doublons = set(tuile_lame_prob)

        # Reconvertir l'ensemble en liste
        liste_sans_doublons = list(ensemble_sans_doublons)

        # Afficher la liste sans doublons
        print(liste_sans_doublons) 
        l=l+1

