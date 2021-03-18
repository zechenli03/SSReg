import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

import pytorch_pretrained_bert.utils as utils
from shared.model_setup import get_tunable_state_dict
import pytorch_pretrained_bert.modeling as modeling
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModelWithLMHead


def save_bert(classification_lm_model, optimizer, args, save_path, save_mode="model_all", verbose=True):
    classification_model = classification_lm_model.classification_model
    lm_model = classification_lm_model.lm_model
    assert save_mode in [
        "all", "tunable", "model_all", "model_tunable",
    ]

    save_dict = dict()

    # Save args
    save_dict["args"] = vars(args)

    # Save model
    classification_model_to_save = classification_model.module \
        if hasattr(classification_model, 'module') \
        else classification_model  # Only save the model itself
    if save_mode in ["all", "model_all"]:
        classification_model_state_dict = classification_model_to_save.state_dict()
        lm_state_dict = get_lm_cls_state_dict(lm_model)
    elif save_mode in ["tunable", "model_tunable"]:
        classification_model_state_dict = get_tunable_state_dict(classification_model_to_save)
        lm_state_dict = get_tunable_state_dict(get_lm_cls_state_dict(lm_model))
    else:
        raise KeyError(save_mode)
    if verbose:
        print("Saving {} classification model elems:".format(len(classification_model_state_dict)))
        print("Saving {} lm model elems:".format(len(lm_state_dict)))
    save_dict["model"] = utils.to_cpu(classification_model_state_dict)
    save_dict["lm_model"] = utils.to_cpu(lm_state_dict)

    # Save optimizer
    if save_mode in ["all", "tunable"]:
        optimizer_state_dict = utils.to_cpu(optimizer.state_dict()) if optimizer is not None else None
        if verbose:
            print("Saving {} optimizer elems:".format(len(optimizer_state_dict)))

    torch.save(save_dict, save_path)


def get_lm_cls_state_dict(lm_model):
    lm_state_dict = lm_model.state_dict()
    for k in list(lm_state_dict):
        if k.startswith("bert."):
            del lm_state_dict[k]
    return lm_state_dict



class MyBertClassificationLM(nn.Module):
    def __init__(self, bert_load_path='', num_labels=2):
        super().__init__()
        self._classification_loss_fct = CrossEntropyLoss()
        config = AutoConfig.from_pretrained(
        bert_load_path,
        num_labels=num_labels,
    )
        self.classification_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=bert_load_path,
        config=config
    )
        self.lm_model = AutoModelWithLMHead.from_pretrained(
        pretrained_model_name_or_path=bert_load_path
    )
        
        self.lm_model.roberta = self.classification_model.roberta
        self.classification_model.to(torch.device("cuda"))
        self.lm_model.to(torch.device("cuda"))
    
    def forward(self, input_ids_classification, input_ids_lm, token_type_ids=None, attention_mask=None,
                classification_labels=None, masked_lm_labels=None, use_lm=False):
        out_cls = self.classification_model(input_ids=input_ids_classification, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=classification_labels)

        if not use_lm:
            if classification_labels is not None:
                classification_loss = out_cls[0]
                del out_cls
                return classification_loss
            else:
                classification_logits = out_cls[0]
                del out_cls
                return classification_logits
        else:
            assert classification_labels is not None
            classification_loss = out_cls[0]
            del out_cls    
            if masked_lm_labels is not None:
                outputs_lm = self.lm_model(input_ids=input_ids_lm, masked_lm_labels=masked_lm_labels)
                masked_lm_loss = outputs_lm[0]
                del outputs_lm
                return classification_loss, masked_lm_loss
            else:
                raise RuntimeError