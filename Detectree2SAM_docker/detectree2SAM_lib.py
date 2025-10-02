# Contient les fonctions necessaires au fonctionnement de detectree2SAM

import rasterio
from rasterio.io import DatasetReader
import os
from pathlib import Path
import numpy as np
from shapely.geometry import box
import pandas as pd
import geopandas as gpd
import json
from rasterio.mask import mask
import cv2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import random
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
import glob
from typing import Any, Dict, List
from detectron2.structures import BoxMode
import pycocotools.mask as mask_util
from shapely.geometry import Polygon, shape
from segment_anything import  SamPredictor,SamAutomaticMaskGenerator
from tqdm import tqdm
from shapely import LineString
from shapely.ops import unary_union, polygonize
import shapely
from shapely.validation import make_valid

def get_filenames(directory: str):
    """Get the file names if no geojson is present.
    Récupère les noms de fichiers avec l'extension ".png" dans le répertoire spécifié et les stocke dans une 
    liste de dictionnaires, où chaque dictionnaire contient le nom de fichier sous la clé "file_name"

    Allows for predictions where no delinations have been manually produced.

    Args:
        directory (str): directory of images to be predicted on
    """
    dataset_dicts = []
    files = glob.glob(directory + "*.png")
    # print("files : ",files)
    for filename in [file for file in files]:
        file = {}
        # filename = os.path.join(directory, filename)
        file["file_name"] = filename

        dataset_dicts.append(file)
    return dataset_dicts

def get_features(gdf: gpd.GeoDataFrame):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them.

    Args:
      gdf: Input geopandas dataframe

    Returns:
      json style data
    """
    return [json.loads(gdf.to_json())["features"][0]["geometry"]]

def tile_data(
    data: DatasetReader,
    out_dir: str,
    buffer: int = 30,
    tile_width: int = 200,
    tile_height: int = 200,
    dtype_bool: bool = False,
) -> None:
    """Tiles up orthomosaic for making predictions on.
    
    Enregistre les tuiles en geotif et en png.
    En geotif : bien georef
    En png : utilisation de cv2. Image strandardisée (8bits)
    
    Tiles up full othomosaic into managable chunks to make predictions on. Use tile_data_train to generate tiled
    training data. A bug exists on some input raster types whereby outputed tiles are completely black - the dtype_bool
    argument should be switched if this is the case.

    Args:
        data: Orthomosaic as a rasterio object in a UTM type projection
        buffer: Overlapping buffer of tiles in meters (UTM)
        tile_width: Tile width in meters
        tile_height: Tile height in meters
        dtype_bool: Flag to edit dtype to prevent black tiles

    Returns:
        None
    """
    os.makedirs(out_dir, exist_ok=True)
    crs = str(data.crs.to_epsg())
    tilename = Path(data.name).stem #Extrait uniquement le nom de l'image, sans extension

    for minx in np.arange(data.bounds[0], data.bounds[2] - tile_width,
                          tile_width, int):
        for miny in np.arange(data.bounds[1], data.bounds[3] - tile_height,
                              tile_height, int):# supprimer - tile width pour le vignetage
            # Naming conventions
            out_path_root = out_dir + tilename + "_" + str(minx) + "_" + str(
                miny) + "_" + str(tile_width) + "_" + str(buffer) + "_" + crs
            # new tiling bbox including the buffer. Peu importe si les coordonnées sortent de l'image
            bbox = box(
                minx - buffer,
                miny - buffer,
                minx + tile_width + buffer,
                miny + tile_height + buffer,
            )

            # turn the bounding boxes into geopandas DataFrames
            geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=data.crs)

            # here we are cropping the tiff to the bounding box of the tile we want
            coords = get_features(geo)
            # print("Coords:", coords)

            # define the tile as a mask of the whole tiff with just the bounding box
            out_img, out_transform = mask(data, shapes=coords, crop=True)
            # out_img = out_img[:3,:,:]

            # Discard scenes with many out-of-range pixels
            out_sumbands = np.sum(out_img, 0)
            zero_mask = np.where(out_sumbands == 0, 1, 0)
            nan_mask = np.where(out_sumbands == 765, 1, 0)
            sumzero = zero_mask.sum()
            sumnan = nan_mask.sum()
            totalpix = out_img.shape[1] * out_img.shape[2]
            # Ne traite pas l'image si plus d'un quart est noir (0) ou blanc (255) sur les 3 canaux
            if sumzero > 0.25 * totalpix:
                continue
            elif sumnan > 0.25 * totalpix:
                continue

            # Attribue les meta données de l'image entière rasterio (crs, transform....) à la tuile et remplace les valeurs différentes
            out_meta = data.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "nodata": None,
            })
            # dtype needs to be unchanged for some data and set to uint8 for others
            if dtype_bool:
                out_meta.update({"dtype": "uint8"})
            # print("Out Meta:",out_meta)

            # Saving the tile as a new tiff, named by the origin of the tile.
            # If tile appears blank in folder can show the image here and may
            # need to fix RGB data or the dtype
            # show(out_img)
            out_tif = out_path_root + ".tif"
            with rasterio.open(out_tif, "w", **out_meta) as dest:
                dest.write(out_img)

            # read in the tile we have just saved
            # Ainsi on ne charge jamais l'image entière en mémoire, mais seulement une tuile à la fois
            clipped = rasterio.open(out_tif)
            # read it as an array
            # show(clipped)
            arr = clipped.read()

            # each band of the tiled tiff is a colour!
            r = arr[0]
            g = arr[1]
            b = arr[2]

            # stack up the bands in an order appropriate for saving with cv2,
            # then rescale to the correct 0-255 range for cv2

            rgb = np.dstack((b, g, r))  # BGR for cv2

            if np.max(g) > 255:
                rgb_rescaled = 255 * rgb / 65535
            else:
                rgb_rescaled = rgb  # scale to image

            # save this as jpg or png...we are going for png...again, named with the origin of the specific tile
            # here as a naughty method
            cv2.imwrite(
                out_path_root + ".png",
                rgb_rescaled,
            )

# Configuration du model
def setup_cfg(
    base_model: str = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    trains=("trees_train", ),
    tests=("trees_val", ),
    update_model=None,
    workers=2,
    ims_per_batch=2,
    gamma=0.1,
    backbone_freeze=3,
    warm_iter=120,
    momentum=0.9,
    batch_size_per_im=1024,
    base_lr=0.0003389,
    weight_decay=0.001,
    max_iter=1000,
    num_classes=1,
    eval_period=100,
    out_dir="./train_outputs",
    resize=True,
):
    """Set up config object # noqa: D417.

    Args:
        base_model: base pre-trained model from detectron2 model_zoo
        trains: names of registered data to use for training
        tests: names of registered data to use for evaluating models
        update_model: updated pre-trained model from detectree2 model_garden
        workers: number of workers for dataloader
        ims_per_batch: number of images per batch
        gamma: gamma for learning rate scheduler
        backbone_freeze: backbone layer to freeze
        warm_iter: number of iterations for warmup
        momentum: momentum for optimizer
        batch_size_per_im: batch size per image
        base_lr: base learning rate
        weight_decay: weight decay for optimizer
        max_iter: maximum number of iterations
        num_classes: number of classes
        eval_period: number of iterations between evaluations
        out_dir: directory to save outputs
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_model))
    cfg.DATASETS.TRAIN = trains
    cfg.DATASETS.TEST = tests
    cfg.DATALOADER.NUM_WORKERS = workers
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.GAMMA = gamma
    cfg.MODEL.BACKBONE.FREEZE_AT = backbone_freeze
    cfg.SOLVER.WARMUP_ITERS = warm_iter
    cfg.SOLVER.MOMENTUM = momentum
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = batch_size_per_im
    cfg.SOLVER.WEIGHT_DECAY = weight_decay
    cfg.SOLVER.BASE_LR = base_lr
    cfg.OUTPUT_DIR = out_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    if update_model is not None:
        cfg.MODEL.WEIGHTS = update_model
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_model)

    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.RESIZE = resize
    cfg.INPUT.MIN_SIZE_TRAIN = 1000
    return cfg

def predict_on_data(
    directory: str = "./",
    predictor=DefaultPredictor,
    eval=False,
    save: bool = True,
    num_predictions=0,
    ):
    """Make predictions on tiled data.

    Predicts crowns for all png images present in a directory and outputs masks as jsons.
    """
    # print("directory",directory)
    pred_dir = os.path.join(directory, "predictions")

    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    if eval:
        dataset_dicts = get_tree_dicts(directory)
    else:
        dataset_dicts = get_filenames(directory)
    
    # print("dataset_dicts :",dataset_dicts)

    # Works out if all items in folder should be predicted on
    if num_predictions == 0:
        num_to_pred = len(dataset_dicts)
    else:
        num_to_pred = num_predictions

    for d in tqdm(random.sample(dataset_dicts, num_to_pred)):
        # print("chemin: ",d["file_name"])
        # sys.exit(1)
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)

        # Creating the file name of the output file
        file_name_path = d["file_name"]
        # Strips off all slashes so just final file name left
        file_name = os.path.basename(os.path.normpath(file_name_path))
        file_name = file_name.replace("png", "json")
        output_file = os.path.join(pred_dir, f"Prediction_{file_name}")
        # print(output_file)

        if save:
            # Converting the predictions to json files and saving them in the
            # specfied output file.
            evaluations = instances_to_coco_json(outputs["instances"].to("cpu"), d["file_name"])
            with open(output_file, "w") as dest:
                json.dump(evaluations, dest)

def get_tree_dicts(directory: str, classes: List[str] = None, classes_at: str = None) -> List[Dict]:
    """Get the tree dictionaries.

    Args:
        directory: Path to directory
        classes: List of classes to include
        classes_at: Signifies which column (if any) corresponds to the class labels

    Returns:
        List of dictionaries corresponding to segmentations of trees. Each dictionary includes
        bounding box around tree and points tracing a polygon around a tree.
    """

    if classes is not None:
        # list_of_classes = crowns[variable].unique().tolist()
        classes = classes
    else:
        classes = ["tree"]
    # classes = Genus_Species_UniqueList #['tree'] # genus_species list
    dataset_dicts = []

    for filename in [file for file in os.listdir(directory) if file.endswith(".geojson")]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)
        # Turn off type checking for annotations until we have a better solution
        record: Dict[str, Any] = {}

        # filename = os.path.join(directory, img_anns["imagePath"])
        filename = img_anns["imagePath"]

        # Make sure we have the correct height and width
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
        record["image_id"] = filename[0:400]
        record["annotations"] = {}
        # print(filename[0:400])

        objs = []
        for features in img_anns["features"]:
            anno = features["geometry"]
            # pdb.set_trace()
            # GenusSpecies = features['properties']['Genus_Species']
            px = [a[0] for a in anno["coordinates"][0]]
            py = [np.array(height) - a[1] for a in anno["coordinates"][0]]
            # print("### HERE IS PY ###", py)
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            # print("#### HERE ARE SOME POLYS #####", poly)
            if classes != ["tree"]:
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": classes.index(features["properties"][classes_at]),  # id
                    # "category_id": 0,  #id
                    "iscrowd": 0,
                }
            else:
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,  # id
                    "iscrowd": 0,
                }
            # pdb.set_trace()
            objs.append(obj)
            # print("#### HERE IS OBJS #####", objs)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def project_to_geojson(tiles_path, pred_fold=None, output_fold=None):  # noqa:N803
    """Projects json predictions back in geographic space.

    Takes a json and changes it to a geojson so it can overlay with orthomosaic. Another copy is produced to overlay
    with PNGs.

    Args:
        tiles_path (str): Path to the tiles folder.
        pred_fold (str): Path to the predictions folder.
        output_fold (str): Path to the output folder.

    Returns:
        None
    """

    Path(output_fold).mkdir(parents=True, exist_ok=True)
    entries = os.listdir(pred_fold)

    for filename in tqdm(entries):
        if ".json" in filename:
            # print(filename)
            tifpath = Path(tiles_path, (filename.replace("Prediction_", "")))
            tifpath = tifpath.with_suffix(".tif")
            # print(tifpath)

            data = rasterio.open(tifpath)
            epsg = str(data.crs).split(":")[1]
            raster_transform = data.transform
            # print(raster_transform)
            # create a dictionary for each file to store data used multiple times

            # create a geofile for each tile --> the EPSG value should be done
            # automatically
            geofile = {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {
                        "name": "urn:ogc:def:crs:EPSG::" + epsg
                    },
                },
                "features": [],
            }

            # load the json file we need to convert into a geojson
            with open(pred_fold + "/" + filename) as prediction_file:
                datajson = json.load(prediction_file)
            # print("data_json:",datajson)

            # json file is formated as a list of segmentation polygons so cycle through each one
            for crown_data in datajson:
                crown = crown_data["segmentation"]
                confidence_score = crown_data["score"]

                # changing the coords from RLE format so can be read as numbers, here the numbers are
                # integers so a bit of info on position is lost
                mask_of_coords = mask_util.decode(crown)
                crown_coords = polygon_from_mask(mask_of_coords)
                if crown_coords == 0:
                    continue
                moved_coords = []

                # coords from json are in a list of [x1, y1, x2, y2,... ] so convert them to [[x1, y1], ...]
                # format and at the same time rescale them so they are in the correct position for QGIS
                for c in range(0, len(crown_coords), 2):
                    x_coord = crown_coords[c]
                    y_coord = crown_coords[c + 1]

                    # Using rasterio transform here is slower but more reliable
                    x_coord, y_coord = rasterio.transform.xy(transform=raster_transform,
                                                              rows=y_coord,
                                                              cols=x_coord)

                    moved_coords.append([x_coord, y_coord])

                geofile["features"].append({
                    "type": "Feature",
                    "properties": {
                        "Confidence_score": confidence_score
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [moved_coords],
                    },
                })

            # Check final form is correct - compare to a known geojson file if error appears.
            # print("geofile",geofile)

            output_geo_file = os.path.join(output_fold, filename.replace(".json", ".geojson"))
            # print("output location:", output_geo_file)
            with open(output_geo_file, "w") as dest:
                json.dump(geofile, dest)

def polygon_from_mask(masked_arr):
    """Convert RLE data from the output instances into Polygons.

    Leads to a small about of data loss but does not affect performance?
    https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py <-- found here
    """

    contours, _ = cv2.findContours(masked_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points) -  for security
        if contour.size >= 10:
            segmentation.append(contour.flatten().tolist())
    # rles = mask_util.frPyObjects(segmentation, masked_arr.shape[0], masked_arr.shape[1])
    # RLE = mask_util.merge(RLEs) # not used
    # RLE = mask.encode(np.asfortranarray(masked_arr))
    # area = mask_util.area(RLE) # not used
    [x, y, w, h] = cv2.boundingRect(masked_arr)

    if len(segmentation) > 0:
        return segmentation[0]  # , [x, y, w, h], area
    else:
        return 0

def stitch_crowns(folder: str, shift: int = 1):
    """Stitch together predicted crowns.

    Args:
        folder: Path to folder containing geojson files.
        shift: Number of meters to shift the size of the bounding box in by. This is to avoid edge crowns.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing all the crowns.
    """
    crowns_path = Path(folder)
    files = crowns_path.glob("*geojson")
    _, _, _, _, crs = filename_geoinfo(list(files)[0])
    print("crs : ",crs)
    files = crowns_path.glob("*geojson")
    crowns = gpd.GeoDataFrame(
        columns=["Confidence_score", "geometry"],
        geometry="geometry",
        # crs=from_epsg(crs),
        crs = 'epsg:'+str(crs)
    )  # initiate an empty gpd.GDF
    for file in files:
        crowns_tile = gpd.read_file(file)

        geo = box_filter(file, shift)
        # geo.plot()
        crowns_tile = gpd.sjoin(crowns_tile, geo, "inner", "within")
        crowns_tile = crowns_tile.set_crs(crowns.crs, allow_override=True)
        # print(crowns_tile)
        crowns = pd.concat([crowns, crowns_tile], ignore_index=True)
        # crowns = pd.concat([crowns,crowns_tile])
        # print(crowns)
    crowns = crowns.drop("index_right", axis=1).reset_index().drop("index", axis=1)
    # crowns = crowns.drop("index", axis=1)
    return crowns

def filename_geoinfo(filename):
    """Return geographic info of a tile from its filename."""
    parts = os.path.basename(filename).replace(".geojson", "").split("_")

    parts = [int(part) for part in parts[-5:]]  # type: ignore
    minx = parts[0]
    miny = parts[1]
    width = parts[2]
    buffer = parts[3]
    crs = parts[4]
    return (minx, miny, width, buffer, crs)

def box_filter(filename, shift: int = 0):
    """Create a bounding box from a file name to filter edge crowns.

    Args:
        filename: Name of the file.
        shift: Number of meters to shift the size of the bounding box in by. This is to avoid edge crowns.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the bounding box."""
    minx, miny, width, buffer, crs = filename_geoinfo(filename)
    bounding_box = box_make(minx, miny, width, buffer, crs, shift)
    return bounding_box

def box_make(minx: int, miny: int, width: int, buffer: int, crs, shift: int = 0):
    """Generate bounding box from geographic specifications.

    Args:
        minx: Minimum x coordinate.
        miny: Minimum y coordinate.
        width: Width of the tile.
        buffer: Buffer around the tile.
        crs: Coordinate reference system.
        shift: Number of meters to shift the size of the bounding box in by. This is to avoid edge crowns.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the bounding box.
    """
    bbox = box(
        minx - buffer + shift,
        miny - buffer + shift,
        minx + width + buffer - shift,
        miny + width + buffer - shift,
    )
    # geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=from_epsg(crs))
    geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs='epsg:'+str(crs))
    return geo


def clean_crowns(crowns: gpd.GeoDataFrame, iou_threshold=0.7, confidence=0.2,min_area=2):
    """Clean overlapping crowns.

    Outputs can contain highly overlapping crowns including in the buffer region.
    This function removes crowns with a high degree of overlap with others but a
    lower Confidence Score.

    Args:
        crowns (gpd.GeoDataFrame): Crowns to be cleaned.
        iou_threshold (float, optional): IoU threshold that determines whether crowns are overlapping.
        confidence (float, optional): Minimum confidence score for crowns to be retained. Defaults to 0.2.

    Returns:
        gpd.GeoDataFrame: Cleaned crowns.
    """
    # Filter any rows with empty geometry
    crowns = crowns[crowns.is_empty == False]
    # Filter any rows with invalid geometry
    crowns = crowns[crowns.is_valid]
    crowns = crowns[crowns.area>=min_area]
    # Reset the index
    crowns = crowns.reset_index(drop=True)
    # Create an object to store the cleaned crowns
    crowns_out = gpd.GeoDataFrame()
    # crowns_out.set_crs(crowns.crs)
    for index, row in crowns.iterrows():  # iterate over each crown
        if index % 1000 == 0:
            print(str(index) + " / " + str(len(crowns)) + " cleaned")
        # if there is not a crown interesects with the row (other than itself)
        if crowns.intersects(shape(row.geometry)).sum() == 1:
            crowns_out = pd.concat([crowns_out, row.to_frame().T], ignore_index=True)

        else:
            # Find those crowns that intersect with it
            intersecting = crowns.loc[crowns.intersects(shape(row.geometry))]
            intersecting = intersecting.reset_index(drop=True)
            iou = []
            for (
                    index1,
                    row1,
            ) in intersecting.iterrows():  # iterate over those intersecting crowns
                # print(row1.geometry)
                iou.append(calc_iou(row.geometry, row1.geometry))  # Calculate the IoU with each of those crowns
            # print(iou)
            intersecting["iou"] = iou
            matches = intersecting[intersecting["iou"] > iou_threshold]  # Remove those crowns with a poor match
            matches = matches.sort_values("Confidence_score", ascending=False).reset_index(drop=True)
            match = matches.loc[[0]]  # Of the remaining crowns select the crown with the highest confidence
            if match["iou"][0] < 1:  # If the most confident is not the initial crown
                continue
            else:
                match = match.drop("iou", axis=1)
                # print(index)
                crowns_out = pd.concat([crowns_out, match], ignore_index=True)
                # crowns_out = pd.concat([crowns_out,match])
            

    # Convert pandas into back geopandas if it is not already
    if not isinstance(crowns_out, gpd.GeoDataFrame):
        crowns_out = gpd.GeoDataFrame(crowns_out)
    # Filter remaining crowns based on confidence score
    if confidence != 0:
        crowns_out = crowns_out[crowns_out["Confidence_score"] > confidence]
    return crowns_out.reset_index(drop=True)

def calc_iou(shape1, shape2):
    """Calculate the IoU of two shapes."""
    iou = shape1.intersection(shape2).area / shape1.union(shape2).area
    return iou



def shape2points(shape_points,transform):
    '''
    Convertit un dataframe avec des points en coordonnées projetées en liste de point en cordonnées image

    Parameters
    ----------
    shape_points : TYPE : geopandas dataframe
        DESCRIPTION. : dataframe des centroïdes des couronnes segmentées par detectree
    transform : TYPE : rasterio transform
        DESCRIPTION. : paramètres de la transformation projection/image

    Returns
    -------
    Listes des coordonnées des points, des label associés (1 pour SAM), des scores de confiance et des id

    '''
    input_point = []
    input_label = []
    Confidence_score = []
    id_point = []
    for _,row in shape_points.iterrows():
        x = int((row.geometry.x-transform[2])/transform[0])
        y = int((row.geometry.y-transform[5])/transform[4])
        input_point.append([x,y])
        input_label.append(1)
        Confidence_score.append(row.Confidence_score)
        id_point.append(row.id_point)
        
    input_point = np.array(input_point)
    input_label = np.array(input_label)
    
    return(input_point,input_label,Confidence_score,id_point)


    
def point_input(input_point,input_label,Confidence_score,id_point,predictor,multi='3'):
    '''
    Génère un input à SAM pour chaque point de la liste passée en argument

    Parameters
    ----------
    input_point : TYPE : list of points ,[(x1,y1),(x2,y2)...]
        DESCRIPTION. : liste des points à passer en input à SAM
    input_label : TYPE : list of label
        DESCRIPTION. : label passé en input à SAM. 1 pour indiquer que c'est à l'intérieur du polygone à segmenter
    predictor : TYPE : SAM predictor
        DESCRIPTION. : modèle de prediction
    multi : TYPE, optional : string
        DESCRIPTION. The default is 1. Peut valoir : "1" - mode multimask_outpout = False, "best" - on prend le masque avec
        le meilleurs score, 3 - on prend les 3 masques

    Returns
    -------
    l_masks : TYPE : list of masks 
        DESCRIPTION. : liste des masques générés par SAMinput

    '''
    l_masks = []
    l_id = []
    l_Confidence = []
    if multi == '1':
        multi_output = False
    else:
        multi_output = True
    # print("Segmentation, création des masques")
    for point_coord,point_label,c,i in zip(input_point,input_label,Confidence_score,id_point):
        masks, scores, _ = predictor.predict(
        point_coords=np.array([point_coord]),
        point_labels=np.array([point_label]),
        multimask_output=multi_output,
        )
        if multi == "3":
            l_masks = l_masks+[m for m in masks]
            l_id = l_id + [i]*3
            l_Confidence = l_Confidence + [c]*3
        else:
            best_mask = sorted(zip(scores,masks))[-1][-1]
            l_masks.append(best_mask)
            l_id.append(i)
            l_Confidence.append(c)
    return l_masks,l_id,l_Confidence


    



def l_masks2allTrees(l_masks,mask_id,mask_Confidence,all_poly=False):
    '''
    Transforme la liste des masques en liste  de polygones de format shapely

    Parameters
    ----------
    l_masks : TYPE : list of numpy array
        DESCRIPTION. : liste des masques générées par SAM
        
    mask_id : TYPE : list of int
        DESCRIPTION. : liste des id des masques générés par Detectree, qui ont servi d'input

    mask_confidence : TYPE : list of float
        DESCRIPTION. : liste des score de confiance des masqies générés par Detectree, qui ont servi d'input

    all_poly : TYPE : boolean flag
        DESCRIPTION. : flag qui indique si on doit utiliser tous les polygones de tous les masques ou seulement ceux issus 
            des masques qui n'en contiennent pas plus que 3 (conseillé, sinon très long et génère des masques absurdes

    Returns
    -------
    all_trees : TYPE : list of shapely polygons
        DESCRIPTION. : list des masques sous forme de polygones, en coordonnées image 

    '''

    all_trees = []
    all_trees_all = []
    all_id = []
    all_id_all =[]
    all_Confidence = []
    all_Confidence_all = []

                
    for m,i,c in zip(l_masks,mask_id,mask_Confidence):
        # tot_area = m.shape[0]*m.shape[1]
        maskInt = m.astype('uint8')
        # area = np.sum(maskInt)
        # if area <tot_area*0.75: # Un masque ne doit pas couvrir plus de 75% de l'image
        shapes = rasterio.features.shapes(maskInt)
        trees_t = []
        id_t = []
        Confidence_t = []

        for s in shapes:

            if s[1]==1:
                polygon = shapely.geometry.Polygon(s[0]["coordinates"][0])
                trees_t.append(polygon)
                id_t.append(i)
                Confidence_t.append(c)
        
        if all_poly :
            all_trees_all = all_trees_all+trees_t
            all_id_all = all_id_all+id_t
            all_Confidence_all = all_Confidence_all+Confidence_t
    
        if len(trees_t)<=3:
            all_trees = all_trees+trees_t
            all_id = all_id+id_t
            all_Confidence = all_Confidence+Confidence_t

  
    return all_trees,all_id,all_Confidence,all_trees_all,all_id_all,all_Confidence_all


def sizeFilteringUTM(l_poly,all_id=None,all_Confidence=None,size_min=2,size_max=5000):
    '''
    Filtre les polygones dans une liste de polygones shapely en fonction d'une superficie minimum et maximum'

    Parameters
    ----------
    l_poly : TYPE : list of shapely polygons
        DESCRIPTION. : liste des polygones à filtrer
    size_min : TYPE : float
        DESCRIPTION. : aire minimum, en m²
    size_max : TYPE : float
        DESCRIPTION. : aire maximum, en m²


    Returns
    -------
    poly_union_filtered : TYPE : list of shapely polygons
        DESCRIPTION. : liste des polygones filtrés

    '''
    if all_id==None:
        l_poly_filtered = [p for p in l_poly if (p.area >= size_min and p.area<=size_max)]
        return l_poly_filtered
    else:
        filtered = [[p,i,c] for p,i,c in zip(l_poly,all_id,all_Confidence) if (p.area >= size_min and p.area<=size_max)]
        l_poly_filtered = [el[0] for el in filtered ]
        l_id_filtered = [el[1] for el in filtered]
        l_Confidence_filtered = [el[2] for el in filtered]

        return l_poly_filtered,l_id_filtered,l_Confidence_filtered
    

def union(all_trees):
    '''
    Equivalent au traitement QGIS :
    Union - supprimer les géom dupliquées - morceaux multiples à morceaux uniques

    Parameters
    ----------
    all_trees : TYPE : list of shapely polygons
        DESCRIPTION. : liste des polygones dont on veut réaliser le traitement

    Returns
    -------
    result : TYPE : list of shapely polygons
        DESCRIPTION. : liste des polygones traités

    '''
    rings = [LineString(list(pol.exterior.coords)) for pol in all_trees]
    union = unary_union(rings)
    result = [geom for geom in polygonize(union)]
    print(len(result))
    return result

def opening_UTM(poly_union_filtered,buffer=0.5):
    '''
    Réalise une érosion puis une dilatation des polygones (ouverture)

    Parameters
    ----------
    poly_union_filtered : TYPE : list of shapely polygons
        DESCRIPTION. : liste des polygones à traiter
    image_path : TYPE : str
        DESCRIPTION. : chemin vers l'image Ã  segmenter'
    buffer : TYPE : float
        DESCRIPTION. : dimension du buffer pour l'ouverture, en mètre'

    Returns
    -------
    poly_buffered_simple : TYPE : list of shapely polygons
        DESCRIPTION. : liste des polygones traités

    '''
    poly_buffered = [p.buffer(-buffer).buffer(buffer) for p in poly_union_filtered]
    poly_buffered_simple = []
    for p in poly_buffered:
        if p.geom_type == 'Polygon':
            poly_buffered_simple.append(p)
        else:
            for geom in list(p.geoms):
                poly_buffered_simple.append(geom)
    

    return poly_buffered_simple

def tree2shapefile_UTM(im_rasterio,im_name,shape_dir,all_trees_UTM,all_id=None,all_Confidence=None,reglages=""):

    '''
    Crée un shapefile avec géométrie construite par SAM à partir des masques filtrés issus de la chaine de traitement

    Parameters
    ----------
    im_rasterio : TYPE : reader rasterio
    DESCRIPTION. : reader de l'image
    
    im_name : str
    DESCRIPTION. : nom de l'image traitée
    
    shape_dir : TYPE : str
    DESCRIPTION. : chemin vers le shapeFile de  vectorisation

    all_trees : TYPE : list of shapely geometries
        DESCRIPTION. : list of masks in shapes of shapely polygons


    Returns
    -------
    None
    '''
    crs = 'epsg:'+str(im_rasterio.crs.to_epsg())
    if all_id ==None:
        gdf = gpd.GeoDataFrame(crs=crs,geometry=all_trees_UTM)
        gdf.to_file(shape_dir+im_name+reglages+".shp", driver='ESRI Shapefile')
    else:
        gdf = gpd.GeoDataFrame({"id_point":all_id, "Confidence":all_Confidence},crs=crs,geometry=all_trees_UTM)
        gdf.to_file(shape_dir+im_name+reglages+".shp", driver='ESRI Shapefile')




def SAM_vignette(
        im_rasterio,
        tile_width_SAM, #largeur des tuiles pour le vignettage (m)
        tile_height_SAM, # hauteur des tuiles pour le vignettage (m)
        buffer_SAM, #  taille du buffer (m). Cela évite de couper des couronnes sur les bords. A agrandir si ce problème est observé
        crowns_net, # centroïde des couronnes segmentées par Detectree
        sam,
        multi_mask="3", #Nombre de masques générés par input. Peut valoir : "1" - mode multimask_outpout = False, "best" - on prend le masque avec le meilleurs score, 3 - on prend les 3 masques
        all_poly=False, #permet de ne pas filtrer les masque. A laisser sur False, sauf pour tester
        bord = False, #génère aussi des masques sur les bord. Faible qualité dans cette zone, mais necessaire sur une petite image
        SAMall = True # indique si on réalise une passe en SAM mode all (détection sans input)
        ):
    '''
    Module qui réalise le vignettage pour segment anything puis les différentes segmentations

    '''
    
    print ("all_poly :",all_poly)
    print ("bord :",bord)
    print ("SAMall :",SAMall)
    
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        # pred_iou_thresh=0.95,
        # stability_score_thresh=0.92,
        # crop_n_layers=1,
        # crop_n_points_downscale_factor=1,
        # min_mask_region_area=10,  # Requires open-cv to run post-processing
    )

    all_trees_UTM = [] #detectree - SAMall - SAMinput <=3 polygones / masque
    all_trees_UTM_all = [] #detectree - SAMall - SAMinput tous les polygones / masque
    all_trees_UTM_SAM = [] #SAMall - <=3 polygones / masque
    all_trees_UTM_SAM_all = [] #SAMall tous les polygones / masque
    all_trees_UTM_SAMinput = [] #detectree - SAMinput <=3 polygones / masque
    all_trees_UTM_SAMinput_all = [] #detectree - SAMinput tous les polygones / masque
    all_id = []
    all_id_all = []
    all_id_input = []
    all_id_input_all = []
    all_Confidence = []
    all_Confidence_all = []
    all_Confidence_input = []
    all_Confidence_input_all = []

    if bord :
        l_minx = np.arange(im_rasterio.bounds[0], im_rasterio.bounds[2], tile_width_SAM, int)
        l_miny = np.arange(im_rasterio.bounds[1], im_rasterio.bounds[3], tile_height_SAM, int)
    else : 
        l_minx = np.arange(im_rasterio.bounds[0], im_rasterio.bounds[2] - tile_width_SAM, tile_width_SAM, int)
        l_miny = np.arange(im_rasterio.bounds[1], im_rasterio.bounds[3] - tile_height_SAM, tile_height_SAM, int)

    for minx in tqdm(l_minx):
        for miny in l_miny:
            # new tiling bbox including the buffer. Peu importe si les coordonnées sortent de l'image
            bbox = box(
                minx - buffer_SAM,
                miny - buffer_SAM,
                minx + tile_width_SAM + buffer_SAM,
                miny + tile_height_SAM + buffer_SAM,
            )

            # turn the bounding boxes into geopandas DataFrames
            geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=im_rasterio.crs)

            # here we are cropping the tiff to the bounding box of the tile we want
            coords = get_features(geo)
            # print("Coords:", coords)

            # define the tile as a mask of the whole tiff with just the bounding box
            im_arr, transform = mask(im_rasterio, shapes=coords, crop=True)
            im_arr = im_arr[:3,:,:]

            # Discard scenes with many out-of-range pixels
            out_sumbands = np.sum(im_arr, 0)
            zero_mask = np.where(out_sumbands == 0, 1, 0)
            nan_mask = np.where(out_sumbands == 765, 1, 0)
            sumzero = zero_mask.sum()
            sumnan = nan_mask.sum()
            totalpix = im_arr.shape[1] * im_arr.shape[2]
            # Ne traite pas l'image si plus d'un quart est noir (0) ou blanc (255) sur les 3 canaux
            if sumzero > 0.25 * totalpix:
                continue
            elif sumnan > 0.25 * totalpix:
                continue

            im_arr = np.transpose(im_arr,(1,2,0))
            
            bbox_ker = box(
                minx,
                miny,
                minx + tile_width_SAM,
                miny + tile_height_SAM,
            )
            
            crowns_ker = crowns_net[crowns_net.geometry.within(bbox_ker)]
            crowns_ker_input = crowns_ker.copy(deep=True)
            
            ########################## SAM mode all ################################
            if SAMall :
                l_dict_masks_SAM = mask_generator.generate(im_arr)
                l_masks_SAM = [m['segmentation'] for m in l_dict_masks_SAM]
                # Les masques et les polygones générés par SAM all auront un id et une confidence de 0 pour les distinguer des autres
                mask_id = [0]*len(l_masks_SAM)
                mask_Confidence = [0]*len(l_masks_SAM)
                trees_SAM,_,_,trees_SAM_all,_,_ = l_masks2allTrees(l_masks_SAM,mask_id,mask_Confidence,all_poly=all_poly)
                trees_SAM_UTM = coordIm2coordUTM(trees_SAM, transform)
                trees_SAM_inBboxKer = [tree for tree in trees_SAM_UTM if tree.centroid.within(bbox_ker)]
                multi_trees_SAM = shapely.geometry.MultiPolygon(trees_SAM_inBboxKer)
                if not (multi_trees_SAM.is_valid):
                    multi_trees_SAM = make_valid(multi_trees_SAM)
                crowns_ker = crowns_ker[~crowns_ker.within(multi_trees_SAM)]
                all_trees_UTM = all_trees_UTM + trees_SAM_inBboxKer
                all_id = all_id + [0]*len(trees_SAM_inBboxKer)
                all_Confidence = all_Confidence + [0]*len(trees_SAM_inBboxKer)
                all_trees_UTM_SAM = all_trees_UTM_SAM + trees_SAM_inBboxKer
    
    
                if all_poly:
                    trees_SAM_all_UTM = coordIm2coordUTM(trees_SAM_all, transform)
                    trees_SAM_all_inBboxKer = [tree for tree in trees_SAM_all_UTM if tree.centroid.within(bbox_ker)]
                    all_trees_UTM_all = all_trees_UTM_all + trees_SAM_all_inBboxKer
                    all_id_all = all_id_all + [0]*len(trees_SAM_all_inBboxKer)
                    all_Confidence_all = all_Confidence_all + [0]*len(trees_SAM_all_inBboxKer)
                    all_trees_UTM_SAM_all = all_trees_UTM_SAM_all + trees_SAM_all_inBboxKer


                
            ######################### SAM mode input point #############################
            predictor.set_image(im_arr)
            
            if (len(crowns_ker)>0 and SAMall): 
                input_point,input_label,Confidence_score,id_point = shape2points(crowns_ker,transform)
                # 3 possibilités pour le dernier argument:
                    # "1" : mode un mask
                    # "best" : génère 3 mask et prend celui avec le meilleur score
                    # "3" : garde les 3 masques
                l_masks,l_id,l_Confidence = point_input(input_point,input_label,Confidence_score,id_point,predictor,multi=multi_mask)
                # polygones des masques en coordonnées image
                trees,l_id,l_Confidence,trees_all,l_id_all,l_Confidence_all = l_masks2allTrees(l_masks,l_id,l_Confidence,all_poly=all_poly)
                trees_UTM = coordIm2coordUTM(trees, transform)
                all_trees_UTM = all_trees_UTM + trees_UTM
                all_id = all_id + l_id
                all_Confidence = all_Confidence + l_Confidence
                
                if all_poly:
                    trees_UTM_all = coordIm2coordUTM(trees_all, transform)
                    all_trees_UTM_all = all_trees_UTM_all + trees_UTM_all
                    all_id_all = all_id_all + l_id_all
                    all_Confidence_all = all_Confidence_all + l_Confidence_all


            if (len(crowns_ker_input)>0): 
                input_point,input_label,Confidence_score,id_point = shape2points(crowns_ker_input,transform)
                # 3 possibilités pour le dernier argument:
                    # "1" : mode un mask
                    # "best" : génère 3 mask et prend celui avec le meilleur score
                    # "3" : garde les 3 masques
                l_masks,l_id,l_Confidence = point_input(input_point,input_label,Confidence_score,id_point,predictor,multi=multi_mask)
                # polygones des masques en coordonnées image
                trees,l_id,l_Confidence,trees_all,l_id_all,l_Confidence_all = l_masks2allTrees(l_masks,l_id,l_Confidence,all_poly=all_poly)
                trees_UTM = coordIm2coordUTM(trees, transform)
                all_trees_UTM_SAMinput = all_trees_UTM_SAMinput + trees_UTM
                all_id_input = all_id_input + l_id
                all_Confidence_input = all_Confidence_input + l_Confidence
                
                
                if all_poly:
                    trees_UTM_all = coordIm2coordUTM(trees_all, transform)
                    all_trees_UTM_SAMinput_all = all_trees_UTM_SAMinput_all + trees_UTM_all
                    all_id_input_all = all_id_input_all + l_id_all
                    all_Confidence_input_all = all_Confidence_input_all + l_Confidence_all
                
                
                
    retour = (all_trees_UTM,
              all_id,
              all_Confidence,
              all_trees_UTM_all,
              all_id_all,
              all_Confidence_all,
              all_trees_UTM_SAM,
              all_trees_UTM_SAM_all,
              all_trees_UTM_SAMinput,
              all_id_input,
              all_Confidence_input,
              all_trees_UTM_SAMinput_all,
              all_id_input_all,
              all_Confidence_input_all
              )
            
    return retour


def coordIm2coordUTM(trees,transform):
    '''
    Converti les coordonnées images d'un polygone en coordonnées UTM à l'aide de la transformation affine correspondante

    Parameters
    ----------
    trees : TYPE : list of shapely polygons
        DESCRIPTION. : liste des polygones shapely dont on veut convertir les coordonnées
    transform : TYPE : rasterio transform
        DESCRIPTION. : paramètres de la transformation image / projection

    Returns
    -------
    trees_geo : TYPE : list of shapely polygons
        DESCRIPTION. : liste des polygones avec les coordonnées converties

    '''
    trees_geo=[]
    for poly in trees:
        poly_coords = shapely.get_coordinates(poly)
        for t in range (len(poly_coords)):
            poly_coords[t] = (poly_coords[t][0]*transform[0]+transform[2],poly_coords[t][1]*transform[4]+transform[5])
        polygon = Polygon(poly_coords.tolist())
        trees_geo.append(polygon)
    return trees_geo

def postTraitment_recording(im_rasterio,im_name,shape_dir,polygons_filtered,workflow,tree_min_size=2,tree_max_size=5000):
    '''
    Post traitement et enregistrement avec un algorithme de nettoyage



    '''

            #Suppress superposition
    all_trees_UTM_union = union(polygons_filtered)
            #filtering from area 
    all_trees_UTM_union = sizeFilteringUTM(all_trees_UTM_union,size_min=tree_min_size,size_max=tree_max_size)
            # erosion, dilatation of the polygons 
    poly_buffered = opening_UTM(all_trees_UTM_union,buffer=0.5)
            # filtering from area 
    poly_buffered_filtered = sizeFilteringUTM(poly_buffered,size_min=tree_min_size,size_max=tree_max_size)
            # Recording
    tree2shapefile_UTM(im_rasterio, im_name, shape_dir, poly_buffered_filtered,reglages = "_"+workflow+"_posttraite")


def get_filepaths(directory: str):
    """
    Retourne un dictionnaire avec les chemins de toutes les images tif présentes dans le dossier passé en argument

        
    """
    dataset_dicts = []
    files = glob.glob(directory + "*.tif")
    # print("files : ",files)
    for filename in [file for file in files]:
        file = {}
        file["file_path"] = filename

        dataset_dicts.append(file)
    return dataset_dicts
