import qupath.lib.scripting.QP

def read_geojson()
{
    def server = QP.getCurrentImageData().getServer()

    // need to add annotations to hierarchy so qupath sees them
    def hierarchy = QP.getCurrentHierarchy()
    
    //to have the name of the image
    
    name= GeneralTools.getNameWithoutExtension(server.getMetadata().getName()).minus(" CKPAN-AML")
        
    //*********Get GeoJSON automatically based on naming scheme 
    def path = "D:/raphael_dt/python/result2/geojson/" + name + ".geojson" ;

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