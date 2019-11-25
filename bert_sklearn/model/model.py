import torch
import torch.nn as nn
import torch.nn.functional as F

from .pytorch_pretrained import BertModel
from .pytorch_pretrained import BertPreTrainedModel

def CNN(D, H):
    """
    CNN.

    Parameters
    ----------
    D : int, size of input layer
    H : int, size of hidden layer
    K : int, size of output layer
    """
    
    print("Using CNN with D=%d"%(D))
    layers = [nn.BatchNorm1d(D),
              nn.Linear(D, H)]
    #conv1 = nn.Conv2d(1, 100, (3, D))
    #nn.MaxPool1d(1),
    #nn.Linear(D, 2)]
    return torch.nn.Sequential(*layers)


class BertPlusCNN(BertPreTrainedModel):
    """
    Bert model with CNN classifier head.

    Based on pytorch_pretrained_bert.modeling.BertForSequenceClassification

    Parameters
    ----------
    config : BertConfig
        stores configuration of BertModel
    """

    def __init__(self, config):

        super(BertPlusCNN, self).__init__(config)
        
        self.bert = BertModel(config)
        self.input_dim = config.hidden_size

        self.cnn = CNN(D=self.input_dim,
                       H=self.num_mlp_hiddens)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids=None, input_mask=None, labels=None):

        hidden, pooled_output = self.bert(input_ids,
                                          segment_ids,
                                          input_mask,
                                          output_all_encoded_layers=False)
        output = pooled_output
        print('SIZE:', output.shape)
        output = self.cnn(output)

        if labels is not None:
            loss_criterion = nn.CrossEntropyLoss(reduction='none')
            loss = loss_criterion(output.view(-1, output.shape[-1]), labels.view(-1))
            return loss, output
        else:
            return output
