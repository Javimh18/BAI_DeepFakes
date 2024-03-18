import torch.nn as nn
import numpy as np


class modelEmbedding(nn.Module):
    def __init__(self, original_model):
        super(modelEmbedding, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.avgpool = original_model.avgpool
        self.embedding = nn.Sequential(
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedding(x)

        return embedding

class modelReconstructed(nn.Module):
    def __init__(self, modelEmbedding):
        super(modelReconstructed, self).__init__()
        self.embedding = modelEmbedding 
        self.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.embedding(x)
        outputs = self.fc(x)

        return outputs

class modelEmbeddingT3(nn.Module):
    def __init__(self, original_model):
        super(modelEmbeddingT3, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.avgpool = original_model.avgpool
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedding(x)
        return embedding

class modelReconstructedT3(nn.Module):
    def __init__(self, modelEmbedding):
        super(modelReconstructedT3, self).__init__()
        self.embedding = modelEmbedding 
        self.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.embedding(x)
        outputs = self.fc(x)

        return outputs
