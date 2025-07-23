from ultralytics import YOLO

# Charger le mod�le
model = YOLO("fisheye_detection_without_prynel_and_habbof_pretrain.pt")

# Effectuer la validation
metrics = model.val(
    data="loaf_validation.yaml",
    imgsz=1024,
    batch=4,
    conf=0.3,
    iou=0.5,
    device="0"
)
#print("maps:", metrics.box.maps)
print("map75",metrics.box.map75) 


# Affiche les attributs et m�thodes disponibles dans `metrics`

#ap75 = metrics.box.maps[5]  # AP@0.75 est la 6�me valeur
#print(f"AP@0.75: {ap75:.4f}")
"""
# Recuperer les noms des classes
names = model.names  # Dictionnaire id -> nom de classe

# Afficher les AP50 et AP75 pour chaque classe
for i, class_name in names.items():
    # Obtenir les metriques pour la classe i
    ap75 = metrics.box.ap[5]
    print(f"{class_name:<20} AP75: {ap75:.3f}")

"""
