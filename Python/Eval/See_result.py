## Code pour visualiser différence entre la prediction et le mask 

## IMPORT 

import os
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory("/mnt/d/openslide-win64-20231011/bin"):
        import openslide as ops
else:
    import openslide as ops

from PIL import Image
import re
from skimage import io
import torch
from unpatchify_mask import main

## 1) Recupere image
def recup_image(IMAGE_DIR, image_nom):
    #Acces image 
    print("image en ouverture")
    wsi_image=ops.OpenSlide(os.path.join(IMAGE_DIR,image_nom))
    print("image ouverte")

    #transformation numpy array
    img_creer=io.imread(IMAGE_DIR+image_nom)
    #img_creer = np.array(wsi_image.read_region((0, 0), 0, wsi_image.level_dimensions[0]))
    print("image transformée en np")
    # Convert to uint8 
    #img_creer = img_creer.astype(np.uint8)
    img1 = Image.fromarray(img_creer)

    #diminue dimension image
    x_wsi, y_wsi=wsi_image.dimensions
    new_size = (x_wsi // 10, y_wsi // 10)  # Adjust the new size as needed
    resized_img = img1.resize(new_size)  
    print("image redimensionnée")
    #Enregistrement image
    #nom = image_nom.split('.')
    #info = nom[0]
    #resized_img.save(RESULT_DIR + info + "_final.png")

    return resized_img

## 2) Supperposer mask/label sur image
def supperposition_mask(PREDICT_DIR,nouvel_couleur, resized_img, image_nom):
    z=0
    for image_dir in os.listdir(PREDICT_DIR):
        #print(image_dir)
        z+=1
        #création image png 
        img = Image.open(PREDICT_DIR + image_dir) 
        rgba = img.convert("RGBA") 
        datas = rgba.getdata() 
        
        newData = [] 
        for item in datas: 
            if item[0] == 255 and item[1] == 255 and item[2] == 255:  # finding black colour by its RGB value 
                # storing a transparent value when we find a black colour 
                newData.append((255, 255, 255, 0)) 
            else: 
                newData.append(nouvel_couleur)  # other colours remain unchanged 
        
        rgba.putdata(newData) 
        #rgba.save("transparent_image.png", "PNG") 

        # recupere information 
        resultats = re.search(r'\[d=(\d+),x=(\d+),y=(\d+),w=(\d+),h=(\d+)\]', image_dir)
        if resultats is not None:
            x = int(resultats.group(2))
            y = int(resultats.group(3))
            w = int(resultats.group(4))
            h = int(resultats.group(5))
        else:
            resultats = re.search(r'\[x=(\d+),y=(\d+),w=(\d+),h=(\d+)\]', image_dir)
            if resultats is not None:
                x = int(resultats.group(1))
                y = int(resultats.group(2))
                w = int(resultats.group(3))
                h = int(resultats.group(4))

        #assemblage image 
        new_size = (w // 10, h // 10)  # Adjust the new size as needed
        resized_img2 = rgba.resize(new_size)  

        resized_img.paste(resized_img2, (x // 10, y // 10), mask = resized_img2)

        #nom = image_nom.split('.')
        #info = nom[0]
        #resized_img.save(RESULT_DIR + info + "_final.png")
        #print(image_dir + " Numero "+ str(z) + " Enregistré ! ")
    print("TERMINE !! ")
    return resized_img

## MAIN 

if __name__ == "__main__":
    
    ### Chemin d'acces 
    RESULT_DIR='/result_adj_ttneo_25NM_cree_epoch10_10/'

    # Chemin image initiale 
    IMAGE_DIR="/"

    #Chemin avec label
    LABEL_DIR='/'

    image_nom=[]
    label_nom=[]
    for lame in os.listdir(RESULT_DIR):
        if os.path.isdir(LABEL_DIR+'Labels_ranger_FFX/'+lame):
            image_nom.append('image_FFX/'+lame+'.svs')
            label_nom.append('Labels_ranger_FFX/'+lame+'/')
        elif os.path.isdir(LABEL_DIR+'Labels_ranger_GEMOX/'+lame):
            image_nom.append('image_GEMOX/'+lame+'.svs')
            label_nom.append('Labels_ranger_GEMOX/'+lame+'/')
        else : 
            print("Image non trouvé")
    
    print(len(image_nom))
    print(len(label_nom))
    
    # Si on veut faire tourner programme sur lame spécifique 
    #image_nom=["image_FFX/18AG05880-37_NADJ02_HES.svs","image_FFX/20AG05133-11_NADJ02_HES.svs"]
    #label_nom=["Labels_ranger_FFX/18AG05880-37_NADJ02_HES/","Labels_ranger_FFX/20AG05133-11_NADJ02_HES/"]

    DIR="/"

    SAVE_GEOJSON=False

    for i in range (0,len(image_nom)):
        if i!=4:
            image=image_nom[i]
            print(image)
            label=label_nom[i]

            nom=image.split('/')
            nom2=nom[1].split('.')

            lame=nom2[0]
            # Pour chaque fichier result
            test=['result_adj_ttneo_25NM_cree_epoch10_10']

            for j in range(len(test)):

                print("Preparation " +test[j])

                # Chemin où enregistrer result
                RESULT_DIR=DIR+test[j]+'/'+lame+'/'
            
                #Chemin avec prediction 
                PREDICT_DIR=RESULT_DIR+'tile_prediction_overlay/'
        
                DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print("Debut \n")
                image_init=recup_image(IMAGE_DIR, image)
                print("Image créé \n")

                image_avec_label=supperposition_mask(LABEL_DIR+label,(0,255,0,31), image_init, nom[1])
                print("Enregstrement image avec label")
                #nom = image_nom.split('.')
                #info = nom[0]
                #image_avec_label.save(RESULT_DIR + info + "_label.png")
                print("Image avec label créé \n")

                image_avec_predict_mask=supperposition_mask(PREDICT_DIR,(255,0,0,31), image_avec_label, nom[1])
                #nom = image_nom.split('.')
                #info = nom[0]
                image_avec_predict_mask.save(RESULT_DIR + lame + "_model_cree_overlay2.png")
                print("Image avec label supperposé créé \n")

                # Vert = FN
                #Jaune = TP
                #Rouge = FP

                if SAVE_GEOJSON:
                    main(RESULT_DIR,[PREDICT_DIR],lame)
                    print('Geojson save !!')
                
                print("\nOPERATION TERMINE")