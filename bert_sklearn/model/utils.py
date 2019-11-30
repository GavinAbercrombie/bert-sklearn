import torch

from .pytorch_pretrained import BertTokenizer, BasicTokenizer
from .pytorch_pretrained import PYTORCH_PRETRAINED_BERT_CACHE

from .model import BertPlusCNN

def get_basic_tokenizer(do_lower_case):
    """
    Get a  basic tokenizer(punctuation splitting, lower casing, etc.).
    """
    return BasicTokenizer(do_lower_case=do_lower_case)


def get_tokenizer(bert_model='bert-base-uncased',
                  bert_vocab_file=None,
                  do_lower_case=False):
    """
    Get a BERT wordpiece tokenizer.

    Parameters
    ----------
    bert_model : string
        one of SUPPORTED_MODELS i.e 'bert-base-uncased','bert-large-uncased'
    bert_vocab_file: string
        Optional pathname to vocab file to initialize BERT tokenizer
    do_lower_case : bool
        use lower case with tokenizer

    Returns
    -------
    tokenizer : BertTokenizer
        Wordpiece tokenizer to use with BERT
    """
    if bert_vocab_file is not None:
        return BertTokenizer(bert_vocab_file, do_lower_case=do_lower_case)
    else:
        return BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)


def get_model(bert_model='bert-base-uncased',
              bert_config_json=None,
              from_tf=False,
              num_mlp_layers=0,
              num_mlp_hiddens=500
              state_dict=None,
              local_rank=-1):
    """
    Get a BertPlusCNN model.

    Parameters
    ----------
    bert_model : string
        one of SUPPORTED_MODELS i.e 'bert-base-uncased','bert-large-uncased'
    state_dict : collections.OrderedDict object
         an optional state dictionnary
    local_rank : (int)
        local_rank for distributed training on gpus

    Returns
    -------
    model : BertPlusCNN
        BERT model plus CNN head
    """

    cache_dir = PYTORCH_PRETRAINED_BERT_CACHE/'distributed_{}'.format(local_rank)

    if bert_config_json is not None:
        # load from a tf checkpoint file, pytorch checkpoint file,
        # or a pytorch state dict
        model = BertPlusCNN.from_model_ckpt(config_file_or_dict=bert_config_json,
                                            weights_path=bert_model,
                                            state_dict=state_dict,
                                            from_tf=from_tf,
                                            num_mlp_hiddens=num_mlp_hiddens,
                                            num_mlp_layers=num_mlp_layers)
    else:
        # Load from pre-trained model archive
        print("Loading %s model..."%(bert_model))
        model = BertPlusCNN.from_pretrained(bert_model,
                                            cache_dir=cache_dir,
                                            state_dict=state_dict,
                                            num_mlp_hiddens=num_mlp_hiddens,
                                            num_mlp_layers=num_mlp_layers)

    return model
