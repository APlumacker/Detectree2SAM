# Enchaine detectree puis SAM. Les centroides des polygones tracés par detectree servent d'input à SAM
# Modification de SAM pour qu'il travaille sur des tuiles de l'image principale
# Génère 3 masques et donne la possibilité de tous les garder, ou le meilleur, ou celui généré en single mask
# Après detectree, fais une passe SAM en mode all, puis n'utilise que les input en dehors des polygones qui viennt d'être générés
# génère un id unique pour chaque polygone detectree et le suis jusqu'à la phase de post-traitement
# On peut désactiver la sortie de tous les polygones pour les masques qui en contiennent plus que 3 (car très très long)


import detectree2SAM_lib as d2s
from datetime import timedelta
import shutil 
from segment_anything import sam_model_registry
import warnings
import time
import torch
import rasterio
from detectron2.engine import DefaultPredictor
import os
from pathlib import Path
print("Import du script detectree2SAM_lib")
print("Contenu de /app/to_segment :")
print(os.listdir("/app/to_segment"))
# Dossier contenant les images à traiter
# Attention, à la fin du traitement de chaque image elle est déplacée dans le dossier portant son nom
# Ainsi, en cas d'arrêt avant d'avoir traité toutes les images, il y a simplement à relancer le programme
img_dir = "/app/to_segment/"
#Chemin pour les tuiles temporaires de detectree. Ce dossier peut être supprimé à la fin
tiles_path = "/app/tilespred/" 

# Si True, on traite les masques contenant plus de 3 polygones (polygones aberrants, très très long)
all_poly = False
# Si True, les alogos continue jusqu'en bordure de l'image. Peut être utile sur une petite image, mais beaucoup d'artefacts sur les bords
bord = False
# Si True, il y a une première passe SAM en mode all avec SAM input
SAMall = False

# Surface minimum et maximum des couronnes d'arbres, en m²
tree_min_size = 2
tree_max_size = 2500

# Config detectree
trained_model = "/app/model_detectree/220723_withParacouUAV.pth"
cfg = d2s.setup_cfg(update_model=trained_model) # update_model arg can be used to load in trained  model
#Si pas de GPU ou erreur de pilote CUDA pour Detectree:
# cfg.MODEL.DEVICE='cpu'

# Config du modèle Segment Anything
checkpoints_path = "/app/model_SAM/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Torch CUDA avalaible :',torch.cuda.is_available())
if (DEVICE == torch.device('cpu')):
  print('Activer le GPU') 
# Pour forcer l'utilisation du CPU avec segment Anything (problème de pilote CUDA par exemple)
# DEVICE = torch.device('cpu')

print("Configuration de Segment Anything")
#Configuration de SAM
sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoints_path)
sam.to(device=DEVICE)

#%%

def main(tiles_path,img_path,im_name,shape_dir,sam,all_poly,bord,SAMall,tree_min_size,tree_max_size):
    '''
    Module principal, qui appel la succession des différents traitements


    '''
    tps1 = time.time()
    print("\n #################################################################")
    print("traitement de l'image :",img_path)
    
    if (torch.cuda.is_available()):
        torch.cuda.empty_cache()
        print('Vidage mémoire GPU')
    shutil.rmtree(tiles_path,ignore_errors=True)
    # Read in the tiff file
    im_rasterio = rasterio.open(img_path,mode="r+")
    im_rasterio.nodata = None
    

    
    # Pour detectree, semble la meilleure configuration
    buffer_detectree = 20
    tile_width_detectree = 45
    tile_height_detectree = 45
    
    # Pour segment anything : augmenter le buffer si certaines couronnes sont coupées verticalement ou horizontalement
    tile_width_SAM = 40
    tile_height_SAM = 40
    buffer_SAM = 60


    ############################################## Première passe avec Detectree ###############################
    print("Tuilage pour detectree")
    # Crée 2 versions de la tuile: une geotif et une png
    d2s.tile_data(im_rasterio, tiles_path, buffer_detectree, tile_width_detectree, tile_height_detectree, dtype_bool = True)
    # trained_model = "./copie_git/model_garden/230103_randresize_full.pth"
    
    print("Predictions detectree")
    # Réalise des prédiction sur chaque tuile et enregistre le résultat dans un json au format coco au même nonm, dans le dossier "predictions"
    d2s.predict_on_data(tiles_path, DefaultPredictor(cfg))
    
    # Transforme les json des polygones précédents en geojson aux coordonnées geographiques. enregistre ces geojson dans le dossier predictions_geo
    d2s.project_to_geojson(tiles_path, tiles_path + "predictions/", tiles_path + "predictions_geo/")
    warnings.filterwarnings('ignore')
    folder = tiles_path+ "/predictions_geo"
    crowns = d2s.stitch_crowns(folder, 1)
    crowns = crowns[crowns.is_valid]
    # Simplify crowns to help with editing 
    crowns = crowns.set_geometry(crowns.simplify(0.3))

    crowns_net = d2s.clean_crowns(crowns, iou_threshold=0.4, confidence=0.2,min_area=tree_min_size)
    crowns_net["id_point"] = crowns_net.index + 1

    crowns_net.to_file(shape_dir+im_name + "_detectree.shp", driver='ESRI Shapefile',crs = crowns.crs)
    
 
    ###################################################### SAM ####################################

    print("shapefile to points")
    # transformation des couronnes de detectree en points pour input SAM
    crowns_net.geometry = crowns_net.centroid
    crowns_net.crs = im_rasterio.crs


    print('génération des masques SAM')
    
    # 3 possibilités pour multi_mask:
        # "1" : mode un mask
        # "best" : génère 3 mask et prend celui avec le meilleur score
        # "3" : garde les 3 masques
    SAM_vignette_values = d2s.SAM_vignette(
        im_rasterio = im_rasterio,
        tile_width_SAM = tile_width_SAM,
        tile_height_SAM = tile_height_SAM,
        buffer_SAM = buffer_SAM,
        crowns_net  = crowns_net,
        sam = sam,
        multi_mask="3",
        all_poly= all_poly, #prend tous les polygones (très long) ou seulement s'il y en a au plus 3 dans un masque
        bord = bord,
        SAMall = SAMall

        )
    
    
    detectree_sam_samInput_max3Polygons = SAM_vignette_values[0:3]
    detectree_sam_samInput_allPolygons = SAM_vignette_values[3:6]
    sam_only_max3Polygons = SAM_vignette_values[6] 
    sam_only_allPolygons = SAM_vignette_values[7]
    detectree_samInput_max3Polygons = SAM_vignette_values[8:11]
    detectree_samInput_allPolygons = SAM_vignette_values[11:14]
    
    if SAMall:
        print("detectree / SAM mode all / SAM avec les polygones detectree (en dehors des polygones SAM all) en input")
    
        # detectree / SAM mode all / SAM avec les polygones detectree (en dehors des polygones SAM all) en input
        if all_poly:
                # Enregistrement de tous les polygones bruts sans traitement:
            d2s.tree2shapefile_UTM(im_rasterio, im_name, shape_dir, *detectree_sam_samInput_allPolygons,reglages="_detectree_SAM_SAMinput_brut")
            # Filtrage et enregistrement des masques avec au maximum 3 polygones:
        polygons_filtered = d2s.sizeFilteringUTM(*detectree_sam_samInput_max3Polygons,size_min=tree_min_size,size_max=tree_max_size)
        d2s.tree2shapefile_UTM(im_rasterio, im_name, shape_dir, *polygons_filtered,reglages="_detectree_SAM_SAMinput_filtre")
            # Post traitement et enregistrement
        d2s.postTraitment_recording(im_rasterio,im_name,shape_dir,polygons_filtered[0],workflow = "detectree_SAM_SAMinput",tree_min_size=tree_min_size,tree_max_size=tree_max_size)
        
    
        print("SAM mode all only")
        
        # SAM mode all only
            # Enregistrement de tous les polygones bruts sans traitement:
        if all_poly:
            d2s.tree2shapefile_UTM(im_rasterio, im_name, shape_dir, sam_only_allPolygons,reglages="_SAM_brut")
                # Filtrage et enregistrement des masques avec au maximum 3 polygones:
        polygons_filtered = d2s.sizeFilteringUTM(sam_only_max3Polygons,size_min=tree_min_size,size_max=tree_max_size)
        d2s.tree2shapefile_UTM(im_rasterio, im_name, shape_dir, polygons_filtered,reglages="_SAM_filtre")
            # Post traitement et enregistrement
        d2s.postTraitment_recording(im_rasterio,im_name,shape_dir,polygons_filtered,workflow = "SAM",tree_min_size=tree_min_size,tree_max_size=tree_max_size)

    print("detectree / SAM avec tous les polygones detectree en input")
    # detectree / SAM avec tous les polygones detectree en input
    if all_poly:
        # Enregistrement de tous les polygones bruts sans traitement:
        d2s.tree2shapefile_UTM(im_rasterio, im_name, shape_dir, *detectree_samInput_allPolygons,reglages="_detectree_SAMinput_brut")
        # Filtrage et enregistrement des masques avec au maximum 3 polygones:
    polygons_filtered = d2s.sizeFilteringUTM(*detectree_samInput_max3Polygons,size_min=tree_min_size,size_max=tree_max_size)
    d2s.tree2shapefile_UTM(im_rasterio, im_name, shape_dir, *polygons_filtered,reglages="_detectree_SAMinput_filtre")
        # Post traitement et enregistrement
    d2s.postTraitment_recording(im_rasterio,im_name,shape_dir,polygons_filtered[0],workflow = "detectree_SAMinput",tree_min_size=tree_min_size,tree_max_size=tree_max_size)

    # Supprime le dossier de tuilage de detectree
    shutil.rmtree(tiles_path,ignore_errors=True)
    im_rasterio.close()
    print('Durée d\'exécution (hh:mm:ss):', timedelta(seconds=time.time()-tps1))



# Crée une liste de dictionnaires contenants les chemin vers toutes les images du dossier
filepaths = d2s.get_filepaths(img_dir)

for d in filepaths:
    file_path = d['file_path']
    out_dir = file_path[:-4]
    os.makedirs(out_dir, exist_ok=True)
    img_path = out_dir+"/"+Path(file_path).name
    shutil.copy(file_path,img_path)
    shape_dir = out_dir+"/"
    im_name = (Path(file_path).name)[:-4]
    
    main(tiles_path,img_path,im_name,shape_dir,sam,all_poly,bord,SAMall,tree_min_size,tree_max_size)

    # A commenter si on ne veut pas effacer les images du dossier à traiter
    if os.path.isfile(file_path):
        os.remove(file_path)