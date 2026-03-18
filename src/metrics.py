import os
import matplotlib.pyplot as plt

epochs = list(range(1, 11))

train_loss = [0.7691, 0.6745, 0.6548, 0.6417, 0.6243, 0.6124, 0.6048, 0.5936, 0.5900, 0.5847]
val_loss   = [0.7035, 0.6723, 0.6597, 0.6406, 0.6270, 0.6108, 0.6082, 0.6017, 0.5960, 0.5929]
val_miou   = [0.3358, 0.3523, 0.3619, 0.3724, 0.3856, 0.3945, 0.3987, 0.4043, 0.4081, 0.4117]

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2)
plt.plot(epochs, val_loss, 'r-x', label='Validation Loss', linewidth=2)
plt.title('Training & Validation Loss Converging', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.xticks(epochs)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.plot(epochs, val_miou, 'g-s', label='Val mIoU', linewidth=2)
plt.title('mIoU Performance Improvement', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('mIoU Score', fontsize=12)
plt.xticks(epochs)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_output_path = os.path.join(_repo_root, "outputs", "drywall_training_metrics.png")
plt.savefig(_output_path, dpi=300)
plt.show()

print(f"Graph saved as '{_output_path}'")