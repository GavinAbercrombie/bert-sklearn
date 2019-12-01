import torch
import torch.nn as nn
import torch.nn.functional as F

from .pytorch_pretrained import BertModel
from .pytorch_pretrained import BertPreTrainedModel

def LinearBlock(H1, H2, p):
    return nn.Sequential()
        #nn.Linear(H1, H2))#,
        #nn.BatchNorm1d(H2))#,
        #nn.ReLU())#,
        #nn.Dropout(p))

def CNN(D, n, H, K, p):
    """
    CNN.

    Parameters
    ----------
    D : int, size of input layer
    n : int, number of hidden layers
    H : int, size of hidden layer
    K : int, size of output layer (no of labels)
    p : float, dropout probability
    """
    
    print("Using mlp with D=%d,H=%d,K=%d,n=%d"%(D, H, K, n))
    
    return torch.nn.Sequential(nn.Conv1d(K, D, 1))
    #layers = [nn.BatchNorm1d(D),
    #          LinearBlock(D, H, p)]
    #for _ in range(n-1):
    #    layers.append(LinearBlock(H, H, p))
    #layers.append(nn.Linear(H, K))
    #print('CNN', type(torch.nn.Sequential(*layers)))
    #return torch.nn.Sequential(*layers)


class BertPlusCNN(BertPreTrainedModel):
    """
    Bert model with CNN classifier head.

    Based on pytorch_pretrained_bert.modeling.BertForSequenceClassification

    Parameters
    ----------
    config : BertConfig
        stores configuration of BertModel
    """

    def __init__(self, config,
                 num_labels=2,
                 num_mlp_layers=2,
                 num_mlp_hiddens=100):

        super(BertPlusCNN, self).__init__(config)
        self.num_labels = num_labels
        self.num_mlp_layers = num_mlp_layers
        self.num_mlp_hiddens = num_mlp_hiddens
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel(config)
        self.input_dim = config.hidden_size

        self.cnn = CNN(D=self.input_dim,
                       n=self.num_mlp_layers,
                       H=self.num_mlp_hiddens,
                       K=self.num_labels,
                       p=config.hidden_dropout_prob)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids=None, input_mask=None, labels=None):

        hidden, pooled_output = self.bert(input_ids,
                                          segment_ids,
                                          input_mask,
                                          output_all_encoded_layers=False)
        output = pooled_output
        #print(type(output), output.shape) # torch.Size([8, 768]) -- batch size, length
        output = self.dropout(output)
        #print('Dropout output', type(output), output.shape) # torch.Size([8, 768])
        output = self.cnn(output)
        print('Output', type(output), output.shape) # torch.Size([8, 2])

        if labels is not None:
            loss_criterion = nn.CrossEntropyLoss(reduction='none')
            loss = loss_criterion(output.view(-1, output.shape[-1]), labels.view(-1))
            return loss, output
        else:
            return output
