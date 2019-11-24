import torch
import torch.nn as nn

from .pytorch_pretrained import BertModel
from .pytorch_pretrained import BertPreTrainedModel

def LinearBlock(H1, H2, p):
    return nn.Sequential(
        nn.Linear(H1, H2),
        nn.BatchNorm1d(H2),
        #nn.Conv2d(H1, H2, kernel_size=1, stride=1, padding=1),
        #nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.Dropout(p))
        #nn.Linear(H1, H2))

def CNN(D, n, H, K, p):
    """
    CNN w batchnorm and dropout.

    Parameters
    ----------
    D : int, size of input layer
    n : int, number of hidden layers
    H : int, size of hidden layer
    K : int, size of output layer
    p : float, dropout probability
    """

    if n == 0:
        print("Defaulting to linear classifier/regressor")
        return nn.Linear(D, K)
    else:
        print("Using CNN with D=%d,H=%d,K=%d,n=%d"%(D, H, K, n))
        layers = [nn.BatchNorm1d(D),
                  LinearBlock(D, H, p)]
        for _ in range(n-1):
            layers.append(LinearBlock(H, H, p))
        layers.append(nn.Linear(H, K))
        return torch.nn.Sequential(*layers)


class BertPlusCNN(BertPreTrainedModel):
    """
    Bert model with CNN classifier head.

    Based on pytorch_pretrained_bert.modeling.BertForSequenceClassification

    Parameters
    ----------
    config : BertConfig
        stores configuration of BertModel

    model_type : string
         'text_classifier'

    num_labels : int
        For a classifier, this is the number of distinct classes.
        For a regressor his will be 1.

    num_mlp_layers : int
        the number of mlp layers. If set to 0, then defualts
        to the linear classifier/regresor in the original Google paper and code.

    num_mlp_hiddens : int
        the number of hidden neurons in each layer of the mlp.
    """

    def __init__(self, config,
                 model_type="text_classifier",
                 num_labels=2,
                 num_mlp_layers=2,
                 num_mlp_hiddens=500):

        super(BertPlusCNN, self).__init__(config)
        self.model_type = model_type
        self.num_labels = num_labels
        self.num_mlp_layers = num_mlp_layers
        self.num_mlp_hiddens = num_mlp_hiddens

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel(config)
        self.input_dim = config.hidden_size

        self.mlp = CNN(D=self.input_dim,
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

        if self.model_type == "token_classifier":
            output = hidden
        else:
            output = pooled_output
            output = self.dropout(output)

        output = self.mlp(output)

        if labels is not None:
            if self.model_type == "text_classifier":
                loss_criterion = nn.CrossEntropyLoss(reduction='none')
                loss = loss_criterion(output.view(-1, output.shape[-1]), labels.view(-1))
            elif self.model_type == "text_regressor":
                loss_criterion = nn.MSELoss(reduction='none')
                output = torch.squeeze(output)
                loss = loss_criterion(output, labels)
            elif self.model_type == "token_classifier":
                loss_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
                loss = loss_criterion(output.view(-1, output.shape[-1]), labels.view(-1))
            return loss, output
        else:
            return output
