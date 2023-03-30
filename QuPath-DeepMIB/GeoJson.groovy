def annotations = getAnnotationObjects()
def path = "path/to/file.geojson"

// 'FEATURE_COLLECTION' is standard GeoJSON format for multiple objects
exportObjectsToGeoJson(annotations, path, "FEATURE_COLLECTION")

// The same method without the 'FEATURE_COLLECTION' parameter outputs a simple JSON object/array
// exportObjectsToGeoJson(annotations, path)

//Fonctionne pour une image à la fois