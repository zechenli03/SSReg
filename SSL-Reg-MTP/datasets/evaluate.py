from sklearn.metrics import matthews_corrcoef, f1_score

import argparse
import json
import os
import pandas as pd

from scipy.stats import pearsonr, spearmanr

import datasets.tasks as tasks


PROCESSORS = {
    "partisan": tasks.PartisanProcessor,
    "citation_intent": tasks.CitationProcessor,
    "sciie": tasks.SciieProcessor,
    "chemprot": tasks.ChemprotProcessor,
    "imdb": tasks.IMDBProcessor,
    "agenws:": tasks.AGProcessor,
    "rct": tasks.RCTProcessor,
    "amazon": tasks.AmazonProcessor
}

OUTPUT_MODES = {
    "partisan": "classification",
    "citation_intent": "classification",
    "sciie": "classification",
    "chemprot": "classification",
    "imdb": "classification",
    "amazon": "classification",
    "agnews": "classification",
    "rct": "classification"
}

DEFAULT_FOL_NAMES = {
    "partisan": "Partisan",
    "citation_intent": "citation_intent",
    "sciie": "sciie",
    "chemprot": "chemprot",
    "imdb": "imdb",
    "agnews": "agnews",
    "rct": "rct",
    "amazon": "amazon"
}


def simple_accuracy(pred_srs, label_srs):
    return (pred_srs == label_srs).mean()

def acc_and_f1(pred_srs, label_srs, average='macro'):
    acc = simple_accuracy(pred_srs, label_srs)
    f1 = f1_score(y_true=label_srs, y_pred=pred_srs, average=average)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(pred_srs, label_srs):
    pearson_corr = float(pearsonr(pred_srs, label_srs)[0])
    spearman_corr = float(spearmanr(pred_srs, label_srs)[0])
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def compute_metrics(task_name, pred_srs, label_srs):
    assert len(pred_srs) == len(label_srs)
    if task_name == "partisan":
        return acc_and_f1(pred_srs, label_srs)
    elif task_name == "citation_intent":
        return acc_and_f1(pred_srs, label_srs)
    elif task_name == "sciie":
        return acc_and_f1(pred_srs, label_srs)
    elif task_name == "imdb":
        return acc_and_f1(pred_srs, label_srs)
    elif task_name == "agnews":
        return acc_and_f1(pred_srs, label_srs)
    elif task_name == "amazon":
        return acc_and_f1(pred_srs, label_srs)
    elif task_name == "chemprot":
        return acc_and_f1(pred_srs, label_srs, average='micro')
    elif task_name == "rct":
        return acc_and_f1(pred_srs, label_srs, average='micro')
    else:
        raise KeyError(task_name)

def load_labels(task_name, data_dir):
    processor = PROCESSORS[task_name]()
    examples = processor.get_dev_examples(data_dir)
    output_mode = OUTPUT_MODES[task_name]
    if output_mode == "classification":
        label2idx = {label: num for (num, label) in enumerate(processor.get_labels())}
        label_srs = pd.Series([label2idx[example.label] for example in examples])
    elif output_mode == "regression":
        label_srs = pd.Series([example.label for example in examples])
    else:
        raise KeyError(output_mode)
    return label_srs

def load_preds(task_name, pred_file_path):
    pred_df = pd.read_csv(pred_file_path, header=None, sep="\t")
    output_mode = OUTPUT_MODES[task_name]
    if output_mode == "classification":
        pred_srs = pred_df.idxmax(axis=1)
    elif output_mode == "regression":
        pred_srs = pred_df[0]
    else:
        raise KeyError(output_mode)
    return pred_srs

def compute_metrics_from_paths(task_name, pred_file_path, task_data_dir):
    pred_srs = load_preds(task_name, pred_file_path)
    label_srs = load_labels(task_name, task_data_dir)
    return compute_metrics(task_name, pred_srs, label_srs)

def get_default_task_data_dir(task_name):
    data_path = os.environ["DATA_DIR"]
    return os.path.join(data_path, DEFAULT_FOL_NAMES[task_name])


