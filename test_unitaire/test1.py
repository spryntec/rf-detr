from rfdetr.models.lwdetr import SetCriterion
import torch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

"""
    test de la fonction loss_boxes pour voir ce que nous avons comme 
    résultat quand on passe des tensors de manière implicite.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Créer les tenseurs src_boxes et target_boxes
src_boxes = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)  
target_boxes = torch.tensor([[0.3, 0.3,0.3, 0.3]], dtype=torch.float32, device=device)  

outputs = {
    'pred_boxes': torch.tensor([[[0.3, 0.3, 0.3, 0.3]]], dtype=torch.float32, device=device)  # [batch_size=1, num_queries=1, 4]
}
targets = [
    {'boxes': target_boxes}  # Une seule image avec une boîte cible
]
indices = [(torch.tensor([0], dtype=torch.long, device=device), torch.tensor([0], dtype=torch.long, device=device))]  # Correspondance 1:1
num_boxes = 1.0  # Nombre de boîtes cibles

# Initialiser le critère
criterion = SetCriterion(
    num_classes=1,  
    matcher=None,  
    weight_dict={'loss_bbox': 0.5, 'loss_giou': 0.25},  
    focal_alpha=0.25,
    losses=['boxes']
)

# Appeler la fonction loss_boxes
losses = criterion.loss_boxes(outputs, targets, indices, num_boxes)

# Afficher les résultats
print("Pertes calculées :")
print(f"loss_bbox: {losses['loss_bbox'].item()}")
print(f"loss_giou: {losses['loss_giou'].item()}")