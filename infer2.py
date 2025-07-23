
"""inférence with non maximum suppression and rotated boxes (Radious aligned boxes)"""

import io
import os
import requests
import supervision as sv
import numpy as np

from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

#COCO_CLASSES = ["person"]

# Chemin vers le dossier contenant les images de test
test_images_dir = "/home/pryntec/train_loaf_model/data/loaf/images/resolution_1k/doc"

# Charge toutes les images du dossier test (jpg et png)
image_paths = [os.path.join(test_images_dir, fname)
               for fname in os.listdir(test_images_dir)
               if fname.lower().endswith((".jpg", ".jpeg", ".png"))]#[:10]


# Charge les images avec PIL
images = [Image.open(path).convert("RGB") for path in image_paths]

# Charge ton modole avec les poids pre-entrainés
model = RFDETRBase(pretrain_weights="/home/pryntec/fisheye_detection_without_prynel_and_habbof_pretrain/rf-detr/rfdetr_outputs3/checkpoint_best_ema.pth")

# Effectue les predictions
#detections_list = model.predict(images, threshold=0.5)
detections_list = [model.predict(image, threshold=0.5) for image in images]

for path, image, detections in zip(image_paths, images, detections_list):
    # Convertit l'objet Detections en tableau pour NMS
    boxes = detections.xyxy  # format (x1, y1, x2, y2)
    scores = detections.confidence
    class_ids = detections.class_id

    # Crée le tableau attendu par ta fonction bbox_non_max_suppression
    pred_array = np.concatenate([
        boxes,
        scores[:, None],
        class_ids[:, None]
    ], axis=1)

    # Applique le NMS
    keep_mask = sv.box_non_max_suppression(pred_array, iou_threshold=0.3)
    filtered = detections[keep_mask]

    # Labels + visualisation
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(filtered.class_id, filtered.confidence)
    ]
    annotated_image = image.copy()
    annotated_image = sv.RRBoxAnnotator().annotate(annotated_image, filtered)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, filtered, labels)

    print(f"Affichage : {os.path.basename(path)}")
    sv.plot_image(annotated_image)
