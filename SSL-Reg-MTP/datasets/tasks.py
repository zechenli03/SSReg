import csv
import os
import logging

from .core import InputExample

logger = logging.getLogger(__name__)


class TaskType:
    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class DomainProcessor(DataProcessor):
    """Processor for the dataset in dont stop pre-training paper."""
    TASK_TYPE = TaskType.CLASSIFICATION

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class AmazonProcessor(DomainProcessor):
    def get_labels(self):
        """See base class."""
        return ["helpful", "unhelpful"]


class AGProcessor(DomainProcessor):
    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4"]

class RCTProcessor(DomainProcessor):
    def get_labels(self):
        """See base class."""
        return ["OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS", "BACKGROUND"]

class IMDBProcessor(DomainProcessor):
    def get_labels(self):
        """See base class."""
        return ["0", "1"]

class ChemprotProcessor(DomainProcessor):
    def get_labels(self):
        """See base class."""
        return ["INHIBITOR", "ANTAGONIST", "AGONIST", "AGONIST-INHIBITOR", "PRODUCT-OF", 
            "SUBSTRATE_PRODUCT-OF", "SUBSTRATE", "INDIRECT-UPREGULATOR", "UPREGULATOR", 
            "INDIRECT-DOWNREGULATOR", "DOWNREGULATOR", "ACTIVATOR", "AGONIST-ACTIVATOR"]

class SciieProcessor(DomainProcessor):
    def get_labels(self):
        """See base class."""
        return ["CONJUNCTION", "FEATURE-OF", "HYPONYM-OF", "USED-FOR", "PART-OF", "COMPARE", "EVALUATE-FOR"]

class CitationProcessor(DomainProcessor):
    def get_labels(self):
        """See base class."""
        return ["Background", "Uses", "CompareOrContrast", "Extends", "Motivation", "Future"]

class PartisanProcessor(DomainProcessor):
    def get_labels(self):
        """See base class."""
        return ["0", "1"]

PROCESSORS = {
    "partisan": PartisanProcessor,
    "citation_intent": CitationProcessor,
    "sciie": SciieProcessor,
    "chemprot": ChemprotProcessor,
    "imdb": IMDBProcessor,
    "agnews": AGProcessor,
    "rct": RCTProcessor,
    "amazon": AmazonProcessor
}


DEFAULT_FOLDER_NAMES = {
    "partisan": "PARTISAN",
    "citation_intent": "citation_intent",
    "sciie": "sciie",
    "chemprot": "chemprot",
    "imdb": "imdb",
    "agnews": "agnews",
    "rct": "rct",
    "amazon": "amazon"
}


class Task:
    def __init__(self, name, processor, data_dir):
        self.name = name
        self.processor = processor
        self.data_dir = data_dir
        self.task_type = processor.TASK_TYPE

    def get_train_examples(self):
        return self.processor.get_train_examples(self.data_dir)

    def get_dev_examples(self):
        return self.processor.get_dev_examples(self.data_dir)

    def get_test_examples(self):
        return self.processor.get_test_examples(self.data_dir)

    def get_labels(self):
        return self.processor.get_labels()


def get_task(task_name, data_dir):
    task_name = task_name.lower()
    task_processor = PROCESSORS[task_name]()
    if data_dir is None:
        data_dir = os.path.join(os.environ["DATA_DIR"], DEFAULT_FOLDER_NAMES[task_name])
    return Task(task_name, task_processor, data_dir)
