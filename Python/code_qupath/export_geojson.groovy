def path = "./"


def annotations = getAnnotationObjects()
def imageData= getCurrentImageData()
def name=GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName()).minus("-HES")

exportObjectsToGeoJson(annotations, path + name + ".geojson", "FEATURE_COLLECTION")
