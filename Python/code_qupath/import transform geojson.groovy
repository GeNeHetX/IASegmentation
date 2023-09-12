import qupath.lib.scripting.QP

def class_tumor = "Tumor"

def read_geojson()
{
    def path_dir = "./"

    def server = QP.getCurrentImageData().getServer()

    // need to add annotations to hierarchy so qupath sees them
    def hierarchy = QP.getCurrentHierarchy()
    
    //to have the name of the image
    
    name= GeneralTools.getNameWithoutExtension(server.getMetadata().getName()).minus(" CKPAN-AML")
        
    //*********Get GeoJSON automatically based on naming scheme 
    def path = path_dir + name + ".geojson" ;

    def JSONfile = new File(path)
    if (!JSONfile.exists()) {
        println "No GeoJSON file for this image..."
        return
    }

    var objs = PathIO.readObjects(JSONfile)
    for (annotation in objs) {
        println "Object: "+annotation.toString()
            
        annotation.setLocked(true)
        hierarchy.addPathObject(annotation) 
    }
}


read_geojson()


selectObjectsByClassification("Stroma");
makeInverseAnnotation()
selectObjectsByClassification("Stroma")
clearSelectedObjects(true)
selectObjectsByClassification(null)
classifySelected(class_tumor)
//runPlugin('qupath.lib.plugins.objects.SplitAnnotationsPlugin', '{}')
