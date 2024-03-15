import os
import re 
import pandas as pd
import cv2
from PIL import Image
import statistics
from tqdm import tqdm
from openpyxl import load_workbook

# Recupere information tuile 
def selection_tuile(tuile):
    
    resultats = re.search(r'\[d=(\d+),x=(\d+),y=(\d+),w=(\d+),h=(\d+)\]', tuile)
    if resultats is not None:
        x = int(resultats.group(2))
        y = int(resultats.group(3))
        w = int(resultats.group(4))
        h = int(resultats.group(5))
    else:
        resultats = re.search(r'\[x=(\d+),y=(\d+),w=(\d+),h=(\d+)\]', tuile)
        if resultats is not None:
            x = int(resultats.group(1))
            y = int(resultats.group(2))
            w = int(resultats.group(3))
            h = int(resultats.group(4))

    return x, y, w,h

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

    new_size = (512, 512)  # Adjust the new size as needed
    resized_img2 = image_pil.resize(new_size)

    return resized_img2


def identifier_cc(DONNE_DIR,df):

    #création Dataframe
    p = pd.DataFrame(columns=['X','Y'])

    #parcourir fichier avec les tuiles 
    for tuile in tqdm(os.listdir(DONNE_DIR)):
        
        #récupere information tuile
        x,y,w,h=selection_tuile(tuile)

        x_min=x
        x_max=x+h
        y_min=y
        y_max=y+w

        #récupere coordonnée noyaux dans la tuile
        filtre = df[(df['X'] > x_min) & (df['X'] < x_max) & (df['Y']>y_min)&(df['Y']<y_max)]
        
        #recupere coutour du mask
        mask=annot(DONNE_DIR,tuile)

        #pour tous les noyuax filtrés 
        for index, row in filtre.iterrows():

            point_x=row['X']
            point_y=row['Y']

            #récupère coordonnées en fonction de la tuile 
            x_p=point_x%512
            y_p=point_y%512
            
            #identifie si noyaux dans le mask 
            valeur_pixel = mask.getpixel((x_p, y_p))
            
            if valeur_pixel!=255:
                #print("ok")
                #Si dedans enregistre résultat
                nouvelle_ligne={'X':point_x,'Y':point_y}
                indice_derniere_ligne = len(p)
                p.loc[indice_derniere_ligne] = nouvelle_ligne

    p_sans_duplicates = p.drop_duplicates() # enlève noyaux compté deux fois 
    return p_sans_duplicates

def identifier_cs(df,p):

    #création Dataframe 
    n=pd.DataFrame(columns=['X','Y'])

    # Sélectionner les lignes qui sont dans df mais pas dans p
    merged_df = pd.merge(df, p, on=['X', 'Y'], how='outer', indicator=True)
    n = merged_df[merged_df['_merge'] == 'left_only'][['X', 'Y']]         

    return n 

def each_lame(LABEL_DIR,PREDICT_DIR,chemin_csv):
    
    #enregistre cooronnée noyaux de la lame dans un DataFrame 
    print(chemin_csv)
    df = pd.read_csv(chemin_csv)
    print(len(df))

    #Identifie noyaux dans zone tumorale
    p_label=identifier_cc(LABEL_DIR,df)
    
    #Selectionne noyaux sain
    n_label = identifier_cs(df,p_label)
    
    print("Cellule cancéreuse : ",len(p_label))
    print("Cellule saine : ", len(n_label))

   
    #Récupere identification pour tuile predite 
    p_predict=identifier_cc(PREDICT_DIR,df)
    n_predict = identifier_cs(df,p_predict)

    TP=0
    TN=0
    FP=0
    FN=0

    #Compte nombre de noyaux correctement identifié comme tumoraux 
    merged_df = pd.merge(p_label, p_predict, on=['X', 'Y'], how='inner')
    TP=len(merged_df)

    #Compte nombre noyaux mal identifié qui sont normalement tumoraux 
    left_df = pd.merge(p_label, p_predict, on=['X', 'Y'], how='outer', indicator=True)
    left = left_df[left_df['_merge'] == 'left_only'][['X', 'Y']]  
    FN=len(left) 

    #Compte nombre de noyaux correctement identifié comme sain 
    merged_df = pd.merge(n_label, n_predict, on=['X', 'Y'], how='inner')
    TN=len(merged_df)

    #Compte nombre de noyaux mal identifié qui sont normalement sain 
    left_df = pd.merge(n_label, n_predict, on=['X', 'Y'], how='outer', indicator=True)
    left = left_df[left_df['_merge'] == 'left_only'][['X', 'Y']]  
    FP=len(left) 

    if (TP+TN+FN+FP)==(len(p_label)+len(n_label)):
        print("Identification noyaux réussi !!")
    else : 
        print ("Problème d'équilibre !!")

    #Calcul des métriques 
    if TP+TN+FP+FN!=0:
        accuracy=(TP+TN)/(TP+TN+FP+FN)    
    if TP+FN !=0:
        sensitivity=TP/(TP+FN)        
    if TN+FP!=0:
        specificity=TN/(TN+FP)
    if TP+FP!=0:
        ppv=TP/(TP+FP)
        

    print("Accuracy Lame", accuracy)
    print("Sensitivity Lame",sensitivity)
    print("Specificity Lame", specificity)
    print("PPV Lame",ppv)

    print("\nTable de contingence stradist lame ")
    print("TP = ", TP)
    print('TN = ', TN)
    print('FN = ', FN)
    print('FP = ',FP)

    return accuracy,sensitivity,specificity,ppv
    
    

def main() : 

    #Dossier contenant les tuiles avec les labels 
    LABEL_DIR="/"
    #Nom des lames specifiques que nous voulons calculer 
    label_nom=[]
    #label_nom=["Labels_ranger_GEMOX/12AG01323-22_NADJ03_HES/","Labels_ranger_FFX/18AG05880-37_NADJ02_HES","Labels_ranger_FFX/20AG05133-11_NADJ02_HES"]

    RESULT_DIR='/result_adj_neo_cree_1020/'
    for lame in os.listdir(RESULT_DIR):
        if os.path.isdir(LABEL_DIR+'Labels_ranger_FFX/'+lame):
            label_nom.append('Labels_ranger_FFX/'+lame+'/')
        elif os.path.isdir(LABEL_DIR+'Labels_ranger_GEMOX/'+lame):
            label_nom.append('Labels_ranger_GEMOX/'+lame+'/')
        else : 
            print("Image non trouvé")
    
    print(label_nom)
    #Dossier contenant tuiles predites 
    PREDICT_DIR="/"
    #Nom des dossier où se trouve les prédictions 
    predict_nom=['result_adj_ttneo_25NM_cree_epoch10_10']

    #Dossier où sont situé les csv 
    CSV='/CSV/'

    #Pour chaque dossier de prediction  
    for j in range(len(predict_nom)) :
        
            print("# Preparation " +predict_nom[j])

            accuracy_mo=[]
            sensitivity_mo=[]
            specificity_mo=[]
            ppv_mo=[]

            #regarde lame specifique 
            for i in range (0,len(label_nom)):
                
                if i!=4:
                    label=label_nom[i]
                    nom=label.split('/')
                    lame=nom[1]
                    
                    #calcul des différentes métriques 
                    accuracy,sensitivity,specificity,ppv=each_lame(LABEL_DIR+label,PREDICT_DIR+predict_nom[j]+'/'+lame+'/'+'tile_prediction/',CSV+lame+'.csv')

                    accuracy_mo.append(accuracy)
                    sensitivity_mo.append(sensitivity)
                    specificity_mo.append(specificity)
                    ppv_mo.append(ppv)

                


            
            #Calule des métriques par dossier de résultat --> permet comparer model  
            print("Accuracy moyenne", statistics.mean(accuracy_mo))
            print("Sensitivity moyenne",statistics.mean(sensitivity_mo))
            print("Specificity moyenne", statistics.mean(specificity_mo))
            print("PPV moyen ", statistics.mean(ppv_mo))   
       
if __name__ == '__main__':

    main()