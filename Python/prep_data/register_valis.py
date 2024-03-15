#docker run -it --mount type=bind,source=/home/audrey/Documents/Valis/data/input,target=/mnt/chemin/acces/au/dossier genehetx/slideregist:0.1 bash
#sudo docker run --memory=5g  -v $(pwd):/usr/local/src cdgatenbee/valis-wsi:latest python3 /mnt/chemin/acces/au/dossier


#sudo docker run --memory=20g  -v $(pwd):$(pwd) cdgatenbee/valis-wsi:latest python3 /mnt/chemin/acces/au/fichier/register_valis.py


from alignement_annot_valis import registration
import alignement_annot_valis
import os
import sys 

# Dossier avec les images à aligner 
slide_src_dir = "./"
# dossier enregistre résult
results_dst_dir = "./"
# image de référence 
target_img_f = ".svs"



# 	Create a Valis object and use it to register the slides in slide_src_dir
registrar = registration.Valis(slide_src_dir, results_dst_dir, reference_img_f=target_img_f, align_to_reference = True)
rigid_registrar, non_rigid_registrar, error_df = registrar.register()
registrar.warp_and_save_slides(results_dst_dir, crop="reference")

registration.kill_jvm()







