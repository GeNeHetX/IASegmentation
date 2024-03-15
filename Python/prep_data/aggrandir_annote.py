import geojson
import os

def multiply_coordinates_by_factor(geometry, factor):
    # Changer grossisement en multipliant donner par le factor
    if geometry["type"] == "Point":
        geometry["coordinates"] = [coord * factor for coord in geometry["coordinates"]]
    elif geometry["type"] in ["LineString", "MultiPoint"]:
        geometry["coordinates"] = [[coord * factor for coord in point] for point in geometry["coordinates"]]
    elif geometry["type"] in ["Polygon", "MultiLineString"]:
        geometry["coordinates"] = [[[coord * factor for coord in ring] for ring in polygon] for polygon in geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        geometry["coordinates"] = [[[[coord * factor for coord in ring] for ring in polygon] for polygon in multi_polygon] for multi_polygon in geometry["coordinates"]]
    return geometry

def multiply_coordinates_in_geojson(geojson_file, factor, output_folder):
    #ouverture du fichier 
    with open(geojson_file, 'r') as file:
        data = geojson.load(file)

    for feature in data["features"]:
        feature["geometry"] = multiply_coordinates_by_factor(feature["geometry"], factor)

    # Créer le chemin pour le nouveau fichier GeoJSON dans le dossier de destination
    output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(geojson_file))[0]}_scaled.geojson")

    # écriture du nouveaux geojson
    with open(output_file, 'w') as file:
        geojson.dump(data, file, indent=2)

    print(f"Le fichier GeoJSON a été modifié et enregistré dans : {output_file}")


if __name__ == '__main__':

    # dossier contenant geojson
    DATA_DIR="/"
    # dossier résultat
    RESULT_DIR="/"
    #nom du fichier
    fichier='.geojson'

    scaling_factor=1.883 # Remplacez par le facteur de multiplication souhaité
    #1.883
    #0.531
    multiply_coordinates_in_geojson(DATA_DIR+fichier, scaling_factor, RESULT_DIR)
