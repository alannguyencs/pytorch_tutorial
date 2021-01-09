import torch
import torchvision


resnet = torchvision.models.resnet18(pretrained=True)


# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
# state_dict objects are Python dictionaries, they can be easily saved,
# updated, altered, and restored, adding a great deal of modularity
# to PyTorch models and optimizers
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))