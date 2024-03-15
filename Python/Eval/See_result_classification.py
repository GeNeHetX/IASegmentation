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
def supperposition_mask(PREDICT_LIT,LABEL_LIT, resized_img, image_nom):
    z=0
    for image_dir in os.listdir(PREDICT_LIT):
        #print(image_dir)
        z+=1
        #création image png 

        img = Image.open(PREDICT_LIT + image_dir) 
        rgba = img.convert("RGBA") 
        datas = rgba.getdata() 

        if os.path.isfile(LABEL_LIT+image_dir):
            #Vrai positif
            newData = [] 
            newData = [(0,255,0,125) for _ in datas]
            
            rgba.putdata(newData) 
        else:
            #Faux positif
            newData = [] 
            newData = [(255,0,0,125) for _ in datas]
            
            rgba.putdata(newData) 
        #rgba.save("transparent_image.png", "PNG") 

        # recupere information 
        resultats = re.search(r'\[d=(\d+),x=(\d+),y=(\d+),w=(\d+),h=(\d+)\]', image_dir)
        x = int(resultats.group(2))
        y = int(resultats.group(3))
        w = int(resultats.group(4))
        h = int(resultats.group(5))

        #assemblage image 
        new_size = (w // 10, h // 10)  # Adjust the new size as needed
        resized_img2 = rgba.resize(new_size)  

        resized_img.paste(resized_img2, (x // 10, y // 10), mask = resized_img2)

        #nom = image_nom.split('.')
        #info = nom[0]
        #resized_img.save(RESULT_DIR + info + "_final.png")
        #print(image_dir + " Numero "+ str(z) + " Enregistré ! ")

    for image_dir in os.listdir(LABEL_LIT):
        if not os.path.isfile(PREDICT_LIT+image_dir):
            #Faux-negatif
            img = Image.open(LABEL_LIT + image_dir) 
            rgba = img.convert("RGBA") 
            datas = rgba.getdata() 

            newData = [] 
            newData = [(0,0,255,125) for _ in datas]
            
            rgba.putdata(newData) 

            # recupere information 
            resultats = re.search(r'\[d=(\d+),x=(\d+),y=(\d+),w=(\d+),h=(\d+)\]', image_dir)
            x = int(resultats.group(2))
            y = int(resultats.group(3))
            w = int(resultats.group(4))
            h = int(resultats.group(5))

            #assemblage image 
            new_size = (w // 10, h // 10)  # Adjust the new size as needed
            resized_img2 = rgba.resize(new_size)  

            resized_img.paste(resized_img2, (x // 10, y // 10), mask = resized_img2)

    print("TERMINE !! ")
    return resized_img

## MAIN 

if __name__ == "__main__":
    
    ### Chemin d'acces 

    # Chemin image initiale 
    IMAGE_DIR="/"
    image_nom=".svs"

    PREDICT_LIT='/result3/12AG01551-14_NADJ02_HES/predict_lit/'

    LABEL_LIT='/Test/12AG01551-14_NADJ02_HES/lit/'

    RESULT_DIR="/result3/12AG01551-14_NADJ02_HES/"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Debut \n")
    image_init=recup_image(IMAGE_DIR, image_nom)
    print("Image créé \n")

    image_avec_label=supperposition_mask(PREDICT_LIT,LABEL_LIT, image_init, image_nom)
    print("Enregstrement image")
    nom = image_nom.split('.')
    info = nom[0]
    image_avec_label.save(RESULT_DIR + info + "_final_modif1.png")
    print("Image créé \n")

    print("\nOPERATION TERMINE")