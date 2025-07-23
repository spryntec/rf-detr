#inférence
"""inference without NMS and oriented bounging boxes"""

import io
import os
import requests
import supervision as sv
#from supervision.detection.utils import non_max_suppression
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv
print("Supervision version:", sv.__version__)

#COCO_CLASSES = ["person"]

# Chemin vers le dossier contenant les images de test
test_images_dir = "/home/pryntec/train_loaf_model/data/loaf/images/resolution_1k/doc"

# Charge toutes les images du dossier test (jpg et png)
image_paths = [os.path.join(test_images_dir, fname)
               for fname in os.listdir(test_images_dir)
               if fname.lower().endswith((".jpg", ".jpeg", ".png"))]#[:10]


# Charge les images avec PIL
images = [Image.open(path).convert("RGB") for path in image_paths]

# Charge ton modele avec les poids pre-entrainés
model = RFDETRBase(pretrain_weights="/home/pryntec/fisheye_detection_without_prynel_and_habbof_pretrain/rf-detr/rfdetr_outputs3/checkpoint_best_ema.pth")

# Effectue les predictions
#detections_list = model.predict(images, threshold=0.5)
detections_list = [model.predict(image, threshold=0.5) for image in images]
print(detections_list)
# Affiche les resultats pour chaque image
for path, image, detections in zip(image_paths, images, detections_list):
    #nms
    #detections = sv.box_non_max_suppression(predictions=detections, iou_threshold=0.5)
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

    print(f"Affichage : {os.path.basename(path)}")
    sv.plot_image(annotated_image)
