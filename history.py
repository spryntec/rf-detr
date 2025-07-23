import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Charger l'historique sauvegard\E9
with open("training_history.pkl", "rb") as f:
    history = pickle.load(f)

# Cr\E9er le DataFrame
df = pd.DataFrame(history)

# Tracer les courbes
plt.figure(figsize=(12, 8))

plt.plot(
    df['epoch'],
    df['train_loss'],
    label='Training Loss',
    marker='o',
    linestyle='-'
)

plt.plot(
    df['epoch'],
    df['test_loss'],
    label='Validation Loss',
    marker='o',
    linestyle='--'
)

plt.title('Train/Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()

