"""sklearn interface to finetuning BERT.

Overall flow:
-------------

    # define model
    model = BertClassifier()       # text/text pair classification

    # fit model to training data
    model.fit(X_train, y_train)

    # score model on holdout data
    model.score(X_dev, y_dev)

    # predict model on new inputs
    model.predict(X_test)


Model inputs X, y:
------------------

    For text pair tasks:
        X = [X1, X2]
            Model inputs are triples : (text_a, text_b, label/target)
            X1 is 1D list-like of text_a strings
            X2 is 1D list-like of text_b strings

    For single text tasks:
        X = 1D list-like of text strings

    For text classification tasks:
        y = 1D list-like of string labels

"""

import sys
import logging
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.base import is_classifier
from sklearn.metrics import f1_score
from sklearn.utils.validation import  check_is_fitted
from sklearn.exceptions import NotFittedError

from .config import model2config
from .data import get_test_dl
from .data.utils import flatten
from .model import get_model
from .model import get_tokenizer
from .model import get_basic_tokenizer
from .utils import prepare_model_and_device
from .utils import get_logger
from .utils import set_random_seed
from .utils import to_numpy
from .utils import unpack_data
from .finetune import finetune
from .finetune import eval_model
from .model.pytorch_pretrained.modeling import PRETRAINED_MODEL_ARCHIVE_MAP

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

SUPPORTED_MODELS = list(PRETRAINED_MODEL_ARCHIVE_MAP.keys())


try:
    import google.colab
    IN_COLAB = True
    pbar = tqdm_notebook
except ModuleNotFoundError:
    IN_COLAB = False
    pbar = tqdm



class BaseBertEstimator(BaseEstimator):
    """
    Base Class for Bert Classifiers and Regressors.

    Parameters
    ----------
    bert_model : string, either
        - one of SUPPORTED_MODELS, i.e 'bert-base-uncased', 'bert-large-uncased'...
        - path to an optional file containing pytorch or tensorflow model weights to load
    bert_config_json : string
        path to an optional file containg BERT model configuration in json
    bert_vocab_file : string
        path to an optional file containg bert vocabulary to initialize the tokenizer
    from_tf : bool
        if the  bert_model is a tensorflow checkpoint that needs to be converted
    do_lower_case : bool
        inform the BERT tokenizer to lowercase all strings before tokenizing
    restore_file : string
        file to restore model state from previous savepoint
    epochs : int
        number of finetune training epochs
    max_seq_length : int
        maximum length of input text sequence (text_a + text_b)
    train_batch_size : int
        batch size for training
    eval_batch_size : int
        batch_size for validation
    label_list :list of strings
        list of classifier labels. For regressors this is None.
    learning_rate :float
        inital learning rate of Bert Optimizer
    warmup_proportion : float
        proportion of training to perform learning rate warmup
    gradient_accumulation_steps : int
        number of update steps to accumulate before performing a backward/update pass
    fp16 : bool
        whether to use 16-bit float precision instead of 32-bit
    loss_scale : float
        loss scaling to improve fp16 numeric stability. Only used when
        fp16 set to True
    local_rank : int
        local_rank for distributed training on gpus
    use_cuda : bool
        use GPU(s) if available
    random_state : intt
        seed to initialize numpy and torch random number generators
    validation_fraction : float
        fraction of training set to use for validation
    logname : string
        path name for logfile
    ignore_label : list of strings
        Labels to be ignored when calculating f1 for token classifiers
    """
    def __init__(self, bert_model='bert-base-uncased',
                 bert_config_json=None, bert_vocab=None,
                 from_tf=False, do_lower_case=None, label_list=None, restore_file=None,
                 epochs=3, max_seq_length=128, train_batch_size=32,
                 eval_batch_size=8, learning_rate=2e-5, warmup_proportion=0.1,
                 gradient_accumulation_steps=1, fp16=False, loss_scale=0,
                 local_rank=-1, use_cuda=True, random_state=42,
                 validation_fraction=0.1, logfile='bert_sklearn.log',
                 ignore_label=None):

        self.id2label, self.label2id = {}, {}
        self.input_text_pairs = None

        self.bert_model = bert_model
        self.bert_config_json = bert_config_json
        self.bert_vocab = bert_vocab
        self.from_tf = from_tf
        self.do_lower_case = do_lower_case
        self.label_list = label_list
        self.restore_file = restore_file
        self.epochs = epochs
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.loss_scale = loss_scale
        self.local_rank = local_rank
        self.use_cuda = use_cuda
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.logfile = logfile
        self.ignore_label = ignore_label
        self.majority_label = None
        self.majority_id = None

        # if given a restore_file, then finish loading a previously finetuned
        # model. Normally a user wouldn't do this directly. This is called from
        # load_model() to finish constructing the object
        if restore_file is not None:
            self.restore_finetuned_model(restore_file)

        self._validate_hyperparameters()

        print("Building sklearn text classifier...")
        self.model_type = "text_classifier"

        self.logger = get_logger(logfile)
        self.logger.info("Loading model:\n" + str(self))

    def load_tokenizer(self):
        """
        Load Basic and BERT Wordpiece Tokenizers
        """
        if self.do_lower_case is None:
            self.do_lower_case = True if 'uncased' in self.bert_model else False

        # get basic tokenizer
        self.basic_tokenizer = get_basic_tokenizer(self.do_lower_case)

        # get bert wordpiece tokenizer
        self.tokenizer = get_tokenizer(self.bert_model, self.bert_vocab, self.do_lower_case)

        return self.tokenizer

    def load_bert(self, state_dict=None):
        """
        Load a BertPlusCNN model from a pretrained checkpoint.

        This will be a pretrained BERT ready to be finetuned.
        """

        # load a vanilla bert model ready to finetune:
        # pretrained bert LM + untrained classifier/regressor
        self.model = get_model(bert_model=self.bert_model,
                               bert_config_json=self.bert_config_json,
                               from_tf=self.from_tf,
                               num_labels=self.num_labels,
                               model_type=self.model_type,
                               state_dict=state_dict,
                               local_rank=self.local_rank)

    def _validate_hyperparameters(self):
        """
        Check hyperpameters are within allowed values.
        """
        if (self.bert_config_json is None) or (self.bert_vocab is None):
            # if bert_config_json and bert_vocab is not specified, then
            # bert_model must be one of the pretrained downloadable models
            if self.bert_model not in SUPPORTED_MODELS:
                raise ValueError("The bert model '%s' is not supported. The list of supported "
                                 "models is %s." % (self.bert_model, SUPPORTED_MODELS))

        if (not isinstance(self.epochs, int) or self.epochs < 1):
            raise ValueError("epochs must be an integer >= 1, got %s" %self.epochs)

        if (not isinstance(self.max_seq_length, int) or self.max_seq_length < 2 or \
                           self.max_seq_length > 512):
            raise ValueError("max_seq_length must be an integer >=2 and <= 512, "
                             "got %s" %self.max_seq_length)

        if (not isinstance(self.train_batch_size, int) or self.train_batch_size < 1):
            raise ValueError("train_batch_size must be an integer >= 1, got %s" %
                             self.train_batch_size)

        if (not isinstance(self.eval_batch_size, int) or self.eval_batch_size < 1):
            raise ValueError("eval_batch_size must be an integer >= 1, got %s" %
                             self.eval_batch_size)

        if self.learning_rate < 0 or self.learning_rate >= 1:
            raise ValueError("learning_rate must be >= 0 and < 1, "
                             "got %s" % self.learning_rate)

        if self.warmup_proportion < 0 or self.warmup_proportion >= 1:
            raise ValueError("warmup_proportion must be >= 0 and < 1, "
                             "got %s" % self.warmup_proportion)

        if (not isinstance(self.gradient_accumulation_steps, int) or \
                self.gradient_accumulation_steps > self.train_batch_size or \
                self.gradient_accumulation_steps < 1):
            raise ValueError("gradient_accumulation_steps must be an integer"
                             " >= 1 and <= train_batch_size, got %s" %
                             self.gradient_accumulation_steps)

        if not isinstance(self.fp16, bool):
            raise ValueError("fp16 must be either True or False, got %s." %
                             self.fp16)

        if not isinstance(self.use_cuda, bool):
            raise ValueError("use_cuda must be either True or False, got %s." %
                             self.fp16)

        if self.validation_fraction < 0 or self.validation_fraction >= 1:
            raise ValueError("validation_fraction must be >= 0 and < 1, "
                             "got %s" % self.validation_fraction)

    def fit(self, X, y, load_at_start=True):
        """
        Finetune pretrained Bert model.

        Parameters
        ----------
        X : 1D or 2D list-like of strings
            Input text, text pair, or token list of data features

        y : 1D or 2D list-like of strings or floats
            Labels/targets for text or token data

        load_at_start : bool
            load model from saved checkpoint file at the start of the fit

        """
        # validate params
        self._validate_hyperparameters()

        # set random seed for reproducability
        set_random_seed(self.random_state, self.use_cuda)

        # unpack data
        texts_a, texts_b, labels = unpack_data(X, y)
        self.input_text_pairs = not texts_b is None

        if is_classifier(self):
            # if the label_list not specified, then infer it from training data
            if self.label_list is None:
                if self.model_type == "text_classifier":
                    self.label_list = np.unique(labels)
                elif self.model_type == "token_classifier":
                    self.label_list = np.unique(flatten(labels))

            # build label2id and id2label maps
            self.num_labels = len(self.label_list)
            for (i, label) in enumerate(self.label_list):
                self.label2id[label] = i
                self.id2label[i] = label

            # calculate majority label for token_classifier
            if self.model_type == "token_classifier":
                c = Counter(flatten(y))
                self.majority_label = c.most_common()[0][0]
                self.majority_id = self.label2id[self.majority_label]

        if load_at_start:
            self.load_tokenizer()
            self.load_bert()

        # to fix BatchLayer1D prob in rare case last batch is a singlton w MLP
        drop_last_batch = False if self.num_mlp_layers == 0 else True

        # create a finetune config object
        config = model2config(self)
        config.drop_last_batch = drop_last_batch
        config.train_sampler = 'random'

        # check lengths match
        assert len(texts_a) == len(labels)
        if texts_b is not None:
            assert len(texts_a) == len(texts_b)

        # finetune model!
        self.model = finetune(self.model, texts_a, texts_b, labels, config)

        return self

    def setup_eval(self, texts_a, texts_b, labels):
        """
        Get dataloader and device for eval.
        """
        config = model2config(self)
        _, device = prepare_model_and_device(self.model, config)
        config.device = device

        dataloader = get_test_dl(texts_a, texts_b, labels, config)
        self.model.eval()
        return dataloader, config

    def score(self, X, y, verbose=True):
        """
        Score model on test/eval data.

        Parameters
        ----------
        X : 1D or 2D list-like of strings
            Input text, text pair, or token list of data features

        y : 1D or 2D list-like of strings or floats
            Labels/targets for text or token data

        Returns
        ----------
        accy: float
            classification accuracy, or pearson for regression

        Raises
        ----------
        NotFittedError - if model has not been fitted yet
        """
        check_is_fitted(self, ["model", "tokenizer"])
        texts_a, texts_b, labels = unpack_data(X, y)

        dataloader, config = self.setup_eval(texts_a, texts_b, labels)

        res = eval_model(self.model, dataloader, config, "Testing")
        loss, accy = res['loss'], res['accy']

        if verbose:
            msg = "\nLoss: %0.04f, Accuracy: %0.02f%%"%(loss, accy)
            if 'f1' in res:
                msg += ", f1: %0.02f"%(100 * res['f1'])
            print(msg)
        return accy

    def save(self, filename):
        """
        Save model state to disk.
        """
        # Only save the model it-self
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        bert_config_json = model_to_save.config.to_dict()
        bert_vocab = self.tokenizer.vocab

        state = {
            'params': self.get_params(),
            'class_name' : type(self).__name__,
            'model_type' : self.model_type,
            'num_labels' : self.num_labels,
            'id2label'   : self.id2label,
            'label2id'   : self.label2id,
            'do_lower_case': self.do_lower_case,
            'bert_config_json' : bert_config_json,
            'bert_vocab'  : bert_vocab,
            'state_dict' : model_to_save.state_dict(),
            'input_text_pairs' : self.input_text_pairs
        }
        torch.save(state, filename)

    def restore_finetuned_model(self, restore_file):
        """
        Restore a previously finetuned model from a restore_file

        This is called from the BertClassifier or BertRegressor. The saved model
        is a finetuned BertPlusMLP
        """
        print("Loading model from %s..."%(restore_file))
        state = torch.load(restore_file)

        params = state['params']
        self.set_params(**params)

        self.model_type = state['model_type']
        self.num_labels = state['num_labels']
        self.do_lower_case = state['do_lower_case']
        self.bert_config_json = state['bert_config_json']
        self.bert_vocab = state['bert_vocab']
        self.from_tf = False
        self.num_labels = state['num_labels']
        self.input_text_pairs = state['input_text_pairs']
        self.id2label = state['id2label']
        self.label2id = state['label2id']

        # get tokenizer and model
        self.load_tokenizer()
        self.load_bert(state_dict=state['state_dict'])


class BertClassifier(BaseBertEstimator, ClassifierMixin):
    """
    A text classifier built on top of a pretrained Bert model.
    """

    def predict_proba(self, X):
        """
        Make class probability predictions.

        Parameters
        ----------
        X : 1D or 2D list-like of strings
            Input text or text pairs

        Returns
        ----------
        probs: numpy 2D array of floats
            probability estimates for each class

        Raises
        ----------
        NotFittedError - if model has not been fitted yet
        """
        check_is_fitted(self, ["model", "tokenizer"])
        texts_a, texts_b = unpack_data(X)

        dataloader, config = self.setup_eval(texts_a, texts_b, labels=None)
        device = config.device

        probs = []
        sys.stdout.flush()
        batch_iter = pbar(dataloader, desc="Predicting", leave=True)

        for batch in batch_iter:
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = self.model(*batch)
                prob = F.softmax(logits, dim=-1)
            prob = prob.detach().cpu().numpy()
            probs.append(prob)
        sys.stdout.flush()
        return np.vstack(tuple(probs))

    def predict(self, X):
        """
        Predict most probable class.

        Parameters
        ----------
        X : 1D or 2D list-like of strings
            Input text, or text pairs

        Returns
        ----------
        y_pred: numpy array of strings
            predicted class estimates

        Raises
        ----------
        NotFittedError - if model has not been fitted yet
        """
        y_pred = np.argmax(self.predict_proba(X), axis=1)
        y_pred = np.array([self.id2label[y] for y in y_pred])
        return y_pred




def load_model(filename):
    """
    Load BertClassifier, BertRegressor, or BertTokenClassifier from a disk file.

        Parameters
        ----------
        filename : string
            filename of saved model file

        Returns
        ----------
        model : BertClassifier, BertRegressor, or BertTokenClassifier model
    """
    state = torch.load(filename)
    class_name = state['class_name']

    classes = {'BertClassifier': BertClassifier}

    # call the constructor to load the model
    model_ctor = classes[class_name]
    model = model_ctor(restore_file=filename)
    return model
