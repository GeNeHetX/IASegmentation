def annotations = getAnnotationObjects()
def path = "D:/Raphael DT/geojson_segmentation/result_3/"
def imageData= getCurrentImageData()
def name=GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName()).minus("_HES")

// 'FEATURE_COLLECTION' is standard GeoJSON format for multiple objects
exportObjectsToGeoJson(annotations, path + name + ".geojson", "FEATURE_COLLECTION")

// The same method without the 'FEATURE_COLLECTION' parameter outputs a simple JSON object/array
// exportObjectsToGeoJson(annotations, path)

//Fonctionne pour une image à la fois