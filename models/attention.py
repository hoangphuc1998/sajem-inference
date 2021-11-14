import torch.nn as nn
import torch.nn.functional as F
class CustomSelfAttention(nn.Module):
    '''
    Custom self attention module (inspired from Multi-Head Self Attention)
    '''
    def __init__(self, embed_dim, bias = True, dropout = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.query_dropout = nn.Dropout(dropout)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.key_dropout = nn.Dropout(dropout)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.value_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([embed_dim])
    def forward(self, image_features):
        query = self.query_proj(image_features) # (S, E)
        query = self.query_dropout(query)
        key = self.key_proj(image_features)
        key = self.key_dropout(key)
        value = self.value_proj(image_features)
        value = self.value_dropout(value)
        attn_weights = F.softmax(query.mm(key.t()), dim=1)
        attn_output = attn_weights.mm(value) 
        residual = self.layer_norm(image_features + attn_output)
        output = residual.mean(dim=0, keepdim=True)
        return output