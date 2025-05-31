from transformers import DistilBertModel
import torch.nn as nn

class DistilBertShard1(nn.Module):
    """First shard: Embeddings + first half of transformer layers."""
    def __init__(self, model):
        super().__init__()
        self.embeddings = model.embeddings
        self.transformer = nn.ModuleList(model.transformer.layer[:3])
        self.config = model.config

    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        for layer in self.transformer:
            x = layer(x, attention_mask=attention_mask)[0]
        return x

class DistilBertShard2(nn.Module):
    """Second shard: second half of transformer layers + output head."""
    def __init__(self, model):
        super().__init__()
        self.transformer = nn.ModuleList(model.transformer.layer[3:])
        self.pre_classifier = model.pre_classifier
        self.activation = model.activation
        self.dropout = model.dropout
        self.classifier = model.classifier
        self.config = model.config

    def forward(self, hidden_states, attention_mask=None):
        x = hidden_states
        for layer in self.transformer:
            x = layer(x, attention_mask=attention_mask)[0]
        x = x[:, 0]  # Take [CLS] token
        x = self.pre_classifier(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
