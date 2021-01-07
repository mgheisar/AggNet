import torch
import torch.nn as nn
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeatureExtractor(nn.Module):
    """Constructs a model
    Args:
        emb_dim (int): Required dimension of the resulting embedding layer that is outputted by the model.
                        Defaults to 1024.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to true.
    """

    def __init__(self, emb_dim=256, model_architecture="resnet50", pretrained=True, pooling=False):
        super(FeatureExtractor, self).__init__()

        if model_architecture == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
            if pooling:
                self.model.avgpool = nn.AdaptiveAvgPool2d(1)  # when input size is not 224*224
            input_features_fc_layer = self.model.fc.in_features
            # Output embedding
            self.model.fc = nn.Linear(input_features_fc_layer, emb_dim)

        elif model_architecture == "vgg16":
            self.model = models.vgg16(pretrained=pretrained)
            input_features_fc_layer = self.model.classifier[-1].in_features
            mod = list(self.model.classifier.children())
            mod.pop()
            mod.append(nn.Linear(input_features_fc_layer, emb_dim))
            new_classifier = nn.Sequential(*mod)
            self.model.classifier = new_classifier

        elif model_architecture == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
            input_features_fc_layer = self.model.fc.in_features
            self.model.fc = nn.Linear(input_features_fc_layer, emb_dim)

        elif model_architecture == "resnet101":
            self.model = models.resnet101(pretrained=pretrained)
            input_features_fc_layer = self.model.fc.in_features
            self.model.fc = nn.Linear(input_features_fc_layer, emb_dim)

        elif model_architecture == "resnet34":
            self.model = models.resnet34(pretrained=pretrained)
            input_features_fc_layer = self.model.fc.in_features
            self.model.fc = nn.Linear(input_features_fc_layer, emb_dim)

        elif model_architecture == "inceptionresnetv2":
            self.model = models.inceptionresnetv2(pretrained=pretrained)
            self.model.last_linear = nn.Linear(1536, emb_dim)

    def l2_norm(self, input):
        """Perform l2 normalization operation on an input vector.
        code copied from liorshk's repository: https://github.com/liorshk/facenet_pytorch/blob/master/model.py
        """
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization and multiplication
        by scalar (alpha)."""
        embedding = self.model(images)
        embedding = self.l2_norm(embedding)

        return embedding
