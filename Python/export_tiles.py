import os, cv2
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(r'C:/openslide-win64/bin'):
        import openslide as ops
else:
    import openslide as ops
import numpy as np
from tqdm import tqdm

# IMG_DIR=["12AG00001-14-HES.ome.tiff", "12AG01599-15-HES.ome.tiff", "13AG00449-18-HES.ome.tiff", "13AG03751-11-HES.ome.tiff", "13AG07634-47-HES.ome.tiff", "14AG06301-8-HES.ome.tiff", "14AG06588-17-HES.ome.tiff", "14AG05301-10-HES.ome.tiff", "14AG03434-18-HES.ome.tiff", "B0549407-25-HES.ome.tiff"]
RESULT_DIR="./tiles_test_tcga_ds2/"
# DATA_DIR="../../Raw_data/EIT-HES post hematox pre PANCK registered/"
# DATA_DIR="C:/Users/Hajar/Documents/multicentrique_select/"
DATA_DIR="D:/Raw_data/multicentrique/test_multi/"
#RESULT_DIR="./tiles_select_ds2_256/"


if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


size=(512, 512)
down_sample=2
overlap=256

threshold=220
part_tissue_min=0.2


def test_tissue(image, threshold=0, size=(512, 512), part_tissue_min=0):
    image=np.asarray(image)
    image_transform=np.sum(image, axis=2)
    image_binarize=np.where((image_transform/3)<=threshold, 1,0)
    sum_tissue=np.sum(image_binarize)
    ammount_tissue=sum_tissue/(size[0]*size[1])
    # print(sum_tissue)
    if ammount_tissue>=part_tissue_min:
        return True
    
    # print(np.asarray(image))
    return False

def down_sampling(image, size=(512,512)):
    # Get the new dimensions for downsamplingpy 

    
    # Resize the image using cubic interpolation
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    
    return resized_image

for image_dir in os.listdir(DATA_DIR)[:-2]:
# for image_dir in (IMG_DIR):

    print(image_dir)
    lame_name=os.path.splitext(image_dir)[0]


    wsi_image=ops.OpenSlide(os.path.join(DATA_DIR ,image_dir))



    x_wsi, y_wsi=wsi_image.dimensions

    size_x_tiles_export= (size[0]-overlap)*down_sample
    size_y_tiles_export= (size[1]-overlap)*down_sample

    size_region=(size[0]*down_sample, size[1]*down_sample)


    for x in tqdm(range(0, x_wsi, size_x_tiles_export)):

        for y in (range(0, y_wsi, size_y_tiles_export)):
            image=wsi_image.read_region((x, y), 0, size_region)
            image=image.resize(size)
            if test_tissue(image, threshold, size, part_tissue_min):

                name= lame_name + " [d=" + str(down_sample) + ",x=" + str(x) + ",y=" + str(y) + ",w=" + str(size[0]*down_sample) + ",h=" + str(size[1]*down_sample) + "].tif"

                path_name=os.path.join(RESULT_DIR, name)

                image.save(path_name)





