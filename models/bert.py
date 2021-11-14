import torch
import torch.nn as nn
class BertFinetune(nn.Module):
    def __init__(self, bert_model, output_type='cls'):
        super().__init__()
        self.bert_model = bert_model
        self.output_type = output_type
        #self.dropout = nn.Dropout(0.2)
    def forward(self, input_ids, attention_mask):
        output = self.bert_model(input_ids, attention_mask = attention_mask)
        if self.output_type == 'mean':
            feature = (output[0] * attention_mask.unsqueeze(2)).sum(dim=1).div(attention_mask.sum(dim=1, keepdim=True))
        elif self.output_type == 'cls2':
            feature = torch.cat((output[2][-1][:,0,...], output[2][-2][:,0,...]), -1)
        elif self.output_type == 'cls4':
            feature = torch.cat((output[2][-1][:,0,...], output[2][-2][:,0,...], output[2][-3][:,0,...], output[2][-4][:,0,...]), -1)
        else:
            feature = output[2][-1][:,0,...]
        return feature