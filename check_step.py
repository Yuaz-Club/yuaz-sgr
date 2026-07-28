import torch
c = torch.load('checkpoints/best_val_model.pt', map_location='cpu', weights_only=True)
print('step:', c.get('step', '?'))
print('keys:', list(c.keys()))
