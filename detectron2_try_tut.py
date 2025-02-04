# -*- coding: utf-8 -*-
"""
Abstracted from the official Detectron tutorial 
https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

# Detectron2 Beginner's Tutorial

<img src="https://dl.fbaipublicfiles.com/detectron2/Detectron2-Logo-Horz.png" width="500">

Has basics usage of detectron2, including the following:
"""
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import traceback

try:
    import torch
except Exception as e:
    print("exception...", str(e))
    traceback.print_exc()
    print('')
    

import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

import numpy as np
import json, cv2, random
from cv2 import imshow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode

from detectron2.engine import DefaultTrainer

from detectron2.utils.visualizer import ColorMode

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from tools.utils import get_project_dir
from tqdm import tqdm
import traceback

# from google.colab.patches import imshow
# cv2.waitKey(0)

# # closing all open windows
# cv2.destroyAllWindows()

proj_dir = get_project_dir()

if torch.cuda.is_available():
    device = torch.device("cuda")
    device_type = "gpu"
else:
    device = torch.device("cpu")
    device_type = "cpu"


# """Then, we create a detectron2 config and a detectron2 `DefaultPredictor` to run inference on this image."""

# # cfg = get_cfg()
# # # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# # # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")


def visualize_output_in_d2(image, cfg, outputs, scale = 1.2):
    """_summary_

    Args:
        image (_type_): _description_
        cfg (_type_): _description_
        outputs (_type_): _description_
        scale (float, optional): _description_. Defaults to 1.2.
    """  
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=scale)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    imshow('', out.get_image()[:, :, ::-1])
    cv2.waitKey(1)

################################################

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")
    

def get_balloon_dicts(img_dir):
    """Register the balloon dataset to detectron2, following the [detectron2 custom dataset tutorial](https://detectron2.readthedocs.io/tutorials/datasets.html).

    """
    
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_custom_dataset_in_d2(data_splits = ["train", "val"], dataset_prefix = "balloon_", dataset_path = r"TRY_D2\datasets\balloon_dataset\balloon", data_classes = ["balloon"]):
    #Register the dataset
    for d in data_splits:
        DatasetCatalog.register(dataset_prefix + d, lambda d=d: get_balloon_dicts(os.path.join(dataset_path, d)))
        MetadataCatalog.get(dataset_prefix + d).set(thing_classes=data_classes)
    

####################################################


def d2_config_for_basemodel(model_yaml="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
    """Get config for selected model from model zoo

    Args:
        model_yaml (_type_): _description_

    Returns:
        _type_: _description_
    """    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_yaml))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml)  # Let training initialize from model zoo
    cfg.MODEL.DEVICE = device_type
    return cfg

def update_d2_config_for_training(model_config, train_ds_name, num_classes = 1, num_workers = 2, imgs_per_batch = 2, base_lr = 0.00025, max_iter = 300, roi_batch_per_img = 128, output_dir="TRY_D2\outputs"):
    """update the model config with training params

    Args:
        model_config (_type_): _description_
        train_ds_name (_type_): _description_
        num_classes (int, optional): _description_. Defaults to 1.
        num_workers (int, optional): _description_. Defaults to 2.
        imgs_per_batch (int, optional): _description_. Defaults to 2.
        base_lr (float, optional): _description_. Defaults to 0.00025.
        max_iter (int, optional): _description_. Defaults to 300.
        roi_batch_per_img (int, optional): _description_. Defaults to 128.

    Returns:
        _type_: _description_
    """    
    model_config.DATASETS.TRAIN = (train_ds_name,)
    model_config.DATASETS.TEST = ()
    model_config.DATALOADER.NUM_WORKERS = num_workers
    model_config.SOLVER.IMS_PER_BATCH = imgs_per_batch  # This is the real "batch size" commonly known to deep learning people
    model_config.SOLVER.BASE_LR = base_lr  # pick a good LR
    model_config.SOLVER.MAX_ITER = max_iter    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    model_config.SOLVER.STEPS = []        # do not decay learning rate
    model_config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roi_batch_per_img   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512), only these will be used in loss qualculation for regr, cls
    model_config.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    model_config.OUTPUT_DIR = output_dir
    os.makedirs(model_config.OUTPUT_DIR, exist_ok=True)
    
    return model_config

def train_configured_model(model_config, resume=False):
    """This method trains the model using the d2 config file setup

    Args:
        model_config (_type_): _description_
        resume (bool, optional): _description_. Defaults to False.
    """    
    trainer = DefaultTrainer(model_config)
    trainer.resume_or_load(resume=resume)
    trainer.train()
    

# Commented out IPython magic to ensure Python compatibility.
# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output

#################################################################


def update_model_config_for_inference(model_config, trained_mode_path = r"balloon_data\model_final.pth", score_threshold_inf = 0.7):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    model_config.MODEL.WEIGHTS = os.path.join(model_config.OUTPUT_DIR, trained_mode_path)  # path to the model we just trained
    model_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold_inf   # set a custom testing threshold
    
    return model_config


#metadata=balloon_metadata
def validate_and_visualize_detections(predictor, metadata, val_data_path = r"TRY_D2\datasets\balloon_dataset\balloon\val", sample_size = 3, score_threshold_inf = 0.5):
    """This method loads the model with config, and predicts the output

    Args:
        model_config (_type_): _description_
    """    

    """Then, we randomly select several samples to visualize the prediction results."""

    dataset_dicts = get_balloon_dicts(val_data_path)
    for d in tqdm(random.sample(dataset_dicts, sample_size)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=metadata,
                    scale=score_threshold_inf,
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        imshow('', out.get_image()[:, :, ::-1])
        cv2.waitKey(1)


########################################################################

def get_coco_validation_results(predictor, dataset_name="balloon_val", output_dir="TRY_D2\outputs"):

    evaluator = COCOEvaluator(dataset_name, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`
    

########VIDEO##############

# """# Run panoptic segmentation on a video"""
# # This is the video we're going to process
# from IPython.display import YouTubeVideo, display
# video = YouTubeVideo("ll8TgCZ0plk", width=500)
# display(video)

# Install dependencies, download the video, and crop 5 seconds for processing
# pip install youtube-dl
# youtube-dl https://www.youtube.com/watch?v=ll8TgCZ0plk -f 22 -o video.mp4
# ffmpeg -i video.mp4 -t 00:00:06 -c:v copy video-clip.mp4

# Commented out IPython magic to ensure Python compatibility.
# Run frame-by-frame inference demo on this video (takes 3-4 minutes) with the "demo.py" tool we provided in the repo.

# Note: this is currently BROKEN due to missing codec. See https://github.com/facebookresearch/detectron2/issues/2901 for workaround.
# %run detectron2/demo/demo.py --config-file detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml --video-input video-clip.mp4 --confidence-threshold 0.6 --output video-output.mkv \
#   --opts MODEL.WEIGHTS detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl

# # Download the results
# from google.colab import files
# files.download('video-output.mkv')

def infer_detection_model_in_d2(image_path = r"TRY_D2\input.jpg", model_yaml="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
    #model Configure for mask_rcnn_R_50_FPN_3x
    model_yaml="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"    
    model_config = d2_config_for_basemodel(model_yaml=model_yaml)    
    try_im = cv2.imread(image_path)
    imshow("", try_im)  
    cv2.waitKey(1)  
    predictor = DefaultPredictor(model_config)
    outputs = predictor(try_im)    
    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    print("\n", f"MODEL: {str(os.path.basename(model_yaml)).upper()}", "\n")
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)        
    visualize_output_in_d2(try_im, model_config, outputs)
    

def run_detectron_tut():
    """This method runs the tutorial, step-by-step
    """    
    #############################################################
    
    """# Run a pre-trained detectron2 model on a image from the COCO dataset:
    """
    # wget http://images.cocodataset.org/val2017/000000439715.jpg -O C:\Users\abilash.ananthula\HCLERS\detectron2-master-ext\TRY_D2\input.jpg
    
    # USE Instance Segmentation mask_rcnn_R_50_FPN_3x
    model_yaml="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" 
    # image_path = r"TRY_D2\input.jpg"   
    image_path = os.path.join(proj_dir, "TRY_D2", "input.jpg")
    infer_detection_model_in_d2(image_path = image_path, model_yaml=model_yaml) 
    
    # USE keypoint detection model for Inference 
    model_yaml="COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"    
    infer_detection_model_in_d2(image_path = image_path, model_yaml=model_yaml)
    
    # USE Panoptic segmentation model
    model_yaml="COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"    
    infer_detection_model_in_d2(image_path = image_path, model_yaml=model_yaml)
            
    
    #############################################################
    
    """# Train on a custom dataset
    In this section, we show how to train an existing detectron2 model on a custom dataset in a new format.
    We use [the balloon segmentation dataset](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon)
    which only has one class: balloon.
    We'll train a balloon segmentation model from an existing model pre-trained on COCO dataset, available in detectron2's model zoo.
    Note that COCO dataset does not have the "balloon" category. We'll be able to recognize this new class in a few minutes.    
    """

    # download, decompress the data
    # wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip -O C:\Users\abilash.ananthula\HCLERS\detectron2-master-ext\TRY_D2\datasets\balloon_dataset.zip
    # unzip C:\Users\abilash.ananthula\HCLERS\detectron2-master-ext\TRY_D2\datasets\balloon_dataset.zip 

    """
    ## Prepare the dataset
    Here, the dataset is in its custom format, therefore we write a function to parse it and prepare it into detectron2's standard format. User should write such a function when using a dataset in custom format. See the tutorial for more details.
    """
    # dataset_path = r"TRY_D2\datasets\balloon_dataset\balloon"
    dataset_path = os.path.join("TRY_D2", "datasets", "balloon_dataset", "balloon")
    
    #Register the dataset
    register_custom_dataset_in_d2(data_splits = ["train", "val"], dataset_prefix = "balloon_", dataset_path = dataset_path, data_classes = ["balloon"])
        
    train_ds_name = "balloon_train"
    balloon_metadata = MetadataCatalog.get(train_ds_name)
    
    
    """To verify the dataset is in correct format, let's visualize the annotations of randomly selected samples in the training set:
    """
    train_dataset_dicts = get_balloon_dicts(os.path.join(dataset_path, "train"))
    for d in random.sample(train_dataset_dicts, 1):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        imshow("", out.get_image()[:, :, ::-1])   
        cv2.waitKey(1) 
    
    #############################################################
    
    """## Train!
    Now, let's fine-tune a COCO-pretrained R50-FPN Mask R-CNN model on the balloon dataset. It takes ~2 minutes to train 300 iterations on a P100 GPU.
    """
    model_yaml="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    output_dir=os.path.join(proj_dir, "TRY_D2", "outputs")
    
    #Configue the model for training
    model_config = d2_config_for_basemodel(model_yaml=model_yaml)
    model_config = update_d2_config_for_training(model_config, train_ds_name, num_classes = 1, num_workers = 2, imgs_per_batch = 2, base_lr = 0.00025, max_iter = 300, roi_batch_per_img = 128)
    
    train_configured_model(model_config, resume=False)
    
    #############################################################
    
    """## Inference & evaluation using the trained model
    Now, let's run inference with the trained model on the balloon validation dataset. First, let's create a predictor using the model we just trained:
    """
    model_config = update_model_config_for_inference(model_config, trained_mode_path = r"balloon_data\r50_fpn_mask_rcnn_balloon_final.pth", score_threshold_inf = 0.7)
    
    predictor = DefaultPredictor(model_config)
    validate_and_visualize_detections(predictor, balloon_metadata, val_data_path = r"TRY_D2\datasets\balloon_dataset\balloon\val", sample_size = 1, score_threshold_inf = 0.5)    
    
    #############################################################
    
    """We can also evaluate its performance using AP metric implemented in COCO API.
    """
    get_coco_validation_results(predictor, dataset_name="balloon_val", output_dir=output_dir)
    
    print("COMPLETED...!")
    

if __name__ == "__main__":
    
    try:
        run_detectron_tut()
        print("Training introduction completed.....")
    except Exception as e:        
        try:
            exc_info = sys.exc_info()

            # do you usefull stuff here
            # (potentially raising an exception)
            # try:
            #     raise TypeError("Again !?!")
            # except:
            #     pass
            # end of useful stuff
        finally:
            print("Exception in the tutorial...:\n", str(e))
            print("\n")
            traceback.print_exc()
            # print(traceback.format_exc())
            
            # Display the *original* exception
            traceback.print_exception(*exc_info)
            del exc_info
            print("....")
        
        

    
