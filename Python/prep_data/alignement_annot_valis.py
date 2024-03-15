import json
from valis import registration

# Annotation and target images directory
slide_src_dir = "/"
# Alignment results directory
results_dst_dir = "/"
# The annotation image at which the original features are aligned
annotation_img_f = ".geojson"
# The target image to which the features will be aligned
target_img_f = ".svs"
# The original features as geojson file
annotation_geojson_f = ".geojson"
# The warped features aligned to the target image as geojson file
warped_geojson_annotation_f = ".geojson"

# Create a Valis object and use the target_img_file as reference
registrar = registration.Valis(src_dir=slide_src_dir,
                               dst_dir=results_dst_dir,
                               reference_img_f=target_img_f,
                               align_to_reference=True)

# Apply the registration
rigid_registrar, non_rigid_registrar, error_df = registrar.register()

# Register the annotation source slide from the MALDI.tif image
annotation_source_slide = registrar.get_slide(src_f=annotation_img_f)
# Register the annotation target slide from the HES.svs image
target_slide = registrar.get_slide(src_f=target_img_f)

# Transfer the annotation from MALDI.tif to HES.svs using the pixels.geojson file
warped_geojson = annotation_source_slide.warp_geojson_from_to(geojson_f=annotation_geojson_f,
                                                              to_slide_obj=target_slide)

# Save annotation as warped_pixels in the form geojson file, that can be dragged and dropped into QuPath
with open(warped_geojson_annotation_f, 'w') as f:
    json.dump(warped_geojson, f)
