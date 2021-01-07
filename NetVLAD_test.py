import torch
from AggNet import NetVLAD, SetNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define model
model = SetNet(emb_dim=256, base_model_architecture="resnet50", pretrained=True, pooling=False,
               num_clusters=8).to(device)

# Define loss
# criterion = Loss().cuda()

labels = torch.randint(0, 10, (40, )).long()
x = torch.rand(40, 3, 224, 224)
output = model(x, m=10)  # Set
output = model(x, m=1)  # Single vector
# triplet_loss = criterion(output, labels)
