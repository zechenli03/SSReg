import csv
import os
import logging
import pandas as pd

from utils.utils import InputExample

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

    def get_train_df(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_df(self, data_dir, multi_testset_times):
        """Gets a collection of `InputExample`s for the dev set."""
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


class ColaProcessor(DataProcessor):
    """Processor for the Cola data set (GLUE version)."""
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

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_train_df(self, data_dir):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_df(self, data_dir, multi_testset_times=1):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", multi_testset_times)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = line[1]
                label = "0"
            else:
                text_a = line[3]
                label = line[1]
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_df(self, lines, set_type, multi_testset_times=1):
        """Creates examples for the training and dev sets."""
        df = pd.DataFrame(columns=["sentence"])
        for (i, line) in enumerate(lines):
            text = line[3]
            for a in range(multi_testset_times):
                df = df.append({'sentence': text}, ignore_index=True)
        # df = df.sample(frac=1).reset_index(drop=True)
        return df


class SstProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""
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

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_train_df(self, data_dir):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_df(self, data_dir, multi_testset_times=1):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", multi_testset_times)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = line[1]
                label = "0"
            else:
                text_a = line[0]
                label = line[1]
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def _create_df(self, lines, set_type, multi_testset_times=1):
        """Creates examples for the training and dev sets."""
        df = pd.DataFrame(columns=["sentence"])
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text = line[0]
            for a in range(multi_testset_times):
                df = df.append({'sentence': text}, ignore_index=True)
        # df = df.sample(frac=1).reset_index(drop=True)
        return df


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
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

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_train_df(self, data_dir):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_df(self, data_dir, multi_testset_times=1):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", multi_testset_times)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            if set_type == "test":
                label = "0"
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_df(self, lines, set_type, multi_testset_times=1):
        """Creates examples for the training and dev sets."""
        df = pd.DataFrame(columns=["sentence"])
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[3]
            text_b = line[4]
            text = text_a + ' ' + text_b
            for a in range(multi_testset_times):
                df = df.append({'sentence': text}, ignore_index=True)
        # df = df.sample(frac=1).reset_index(drop=True)
        return df


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""
    TASK_TYPE = TaskType.REGRESSION

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def get_train_df(self, data_dir):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_df(self, data_dir, multi_testset_times=1):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", multi_testset_times)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[7]
            text_b = line[8]
            if set_type == "test":
                label = 0.
            else:
                label = float(line[-1])
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_df(self, lines, set_type, multi_testset_times=1):
        """Creates examples for the training and dev sets."""
        df = pd.DataFrame(columns=["sentence"])
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[7]
            text_b = line[8]
            text = text_a + ' ' + text_b
            for a in range(multi_testset_times):
                df = df.append({'sentence': text}, ignore_index=True)
        # df = df.sample(frac=1).reset_index(drop=True)
        return df


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""
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

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_train_df(self, data_dir):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_df(self, data_dir, multi_testset_times=1):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", multi_testset_times)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            try:
                if set_type == "test":
                    text_a = line[1]
                    text_b = line[2]
                    label = "0"
                else:
                    text_a = line[3]
                    text_b = line[4]
                    label = line[5]
            except IndexError:
                continue
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_df(self, lines, set_type, multi_testset_times=1):
        """Creates examples for the training and dev sets."""
        df = pd.DataFrame(columns=["sentence"])
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            try:
                if set_type == "test":
                    text_a = line[1]
                    text_b = line[2]
                else:
                    text_a = line[3]
                    text_b = line[4]
            except IndexError:
                continue
            for a in range(multi_testset_times):
                df = df.append({'sentence': text_a + ' ' + text_b}, ignore_index=True)
        # df = df.sample(frac=1).reset_index(drop=True)
        return df


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""
    TASK_TYPE = TaskType.CLASSIFICATION

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def get_train_df(self, data_dir):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_df(self, data_dir, multi_testset_times=1):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev", multi_testset_times)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            if set_type == "test":
                label = "contradiction"
            else:
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_df(self, lines, set_type, multi_testset_times=1):
        """Creates examples for the training and dev sets."""
        df = pd.DataFrame(columns=["sentence"])
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[8]
            text_b = line[9]
            text = text_a + ' ' + text_b
            for a in range(multi_testset_times):
                df = df.append({'sentence': text}, ignore_index=True)
        # df = df.sample(frac=1).reset_index(drop=True)
        return df


class MnliMismatchedProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""
    TASK_TYPE = TaskType.CLASSIFICATION

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test")

    def get_AX_examples(self, data_dir):
        """See base class."""
        return self._create_examples2(
            self._read_tsv(os.path.join(data_dir, "diagnostic.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def get_train_df(self, data_dir):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_df(self, data_dir, multi_testset_times=1):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev", multi_testset_times)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            if set_type == "test":
                label = "contradiction"
            else:
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
    def _create_examples2(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            if set_type == "test":
                label = "contradiction"
            else:
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_df(self, lines, set_type, multi_testset_times=1):
        """Creates examples for the training and dev sets."""
        df = pd.DataFrame(columns=["sentence"])
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[8]
            text_b = line[9]
            text = text_a + ' ' + text_b
            for a in range(multi_testset_times):
                df = df.append({'sentence': text}, ignore_index=True)
        # df = df.sample(frac=1).reset_index(drop=True)
        return df


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""
    TASK_TYPE = TaskType.CLASSIFICATION

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")),
                "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def get_train_df(self, data_dir):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_df(self, data_dir, multi_testset_times=1):
        """See base class."""
        return self._create_df(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", multi_testset_times)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, 1)
            text_a = line[1]
            text_b = line[2]
            if set_type == "test":
                label = "entailment"
            else:
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_df(self, lines, set_type, multi_testset_times=1):
        """Creates examples for the training and dev sets."""
        df = pd.DataFrame(columns=["sentence"])
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[1]
            text_b = line[2]
            text = text_a + ' ' + text_b
            for a in range(multi_testset_times):
                df = df.append({'sentence': text}, ignore_index=True)
        # df = df.sample(frac=1).reset_index(drop=True)
        return df


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""
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

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def get_train_df(self, data_dir):
        """See base class."""
        return self._create_df(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_df(self, data_dir, multi_testset_times=1):
        """See base class."""
        return self._create_df(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", multi_testset_times)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            if set_type == "test":
                label = "entailment"
            else:
                label = line[-1]
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_df(self, lines, set_type, multi_testset_times=1):
        """Creates examples for the training and dev sets."""
        df = pd.DataFrame(columns=["sentence"])
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[1]
            text_b = line[2]
            text = text_a + ' ' + text_b
            for a in range(multi_testset_times):
                df = df.append({'sentence': text}, ignore_index=True)
        # # df = df.sample(frac=1).reset_index(drop=True)
        return df


class WnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""
    TASK_TYPE = TaskType.CLASSIFICATION

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_train_df(self, data_dir):
        """See base class."""
        return self._create_df(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_df(self, data_dir, multi_testset_times=1):
        """See base class."""
        return self._create_df(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", multi_testset_times)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            if set_type == "test":
                label = "0"
            else:
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_df(self, lines, set_type, multi_testset_times=1):
        """Creates examples for the training and dev sets."""
        df = pd.DataFrame(columns=["sentence"])
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[1]
            text_b = line[2]
            text = text_a + ' ' + text_b
            for a in range(multi_testset_times):
                df = df.append({'sentence': text}, ignore_index=True)
        # # df = df.sample(frac=1).reset_index(drop=True)
        return df


class SnliProcessor(DataProcessor):
    """Processor for the SNLI data set (GLUE version)."""
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

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def get_train_df(self, data_dir):
        """See base class."""
        return self._create_df(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_df(self, data_dir, multi_testset_times=1):
        """See base class."""
        return self._create_df(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", multi_testset_times)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_df(self, lines, set_type, multi_testset_times=1):
        """Creates examples for the training and dev sets."""
        df = pd.DataFrame(columns=["sentence"])
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[7]
            text_b = line[8]
            text = text_a + ' ' + text_b
            for a in range(multi_testset_times):
                df = df.append({'sentence': text}, ignore_index=True)
        # # df = df.sample(frac=1).reset_index(drop=True)
        return df


class Aug4Processor(DataProcessor):
    """Processor for the augment data set."""
    TASK_TYPE = TaskType.CLASSIFICATION

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["sr", "rd", "ri", "rs"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Aug3Processor(DataProcessor):
    """Processor for the augment data set."""
    TASK_TYPE = TaskType.CLASSIFICATION

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["sr", "rd", "ri"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Aug2Processor(DataProcessor):
    """Processor for the augment data set."""
    TASK_TYPE = TaskType.CLASSIFICATION

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["sr", "rd"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


PROCESSORS = {
    "cola": ColaProcessor,
    "sst": SstProcessor,
    "sst-2": SstProcessor,
    "mrpc": MrpcProcessor,
    "stsb": StsbProcessor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "mnli": MnliProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "snli": SnliProcessor,
    "aug-4": Aug4Processor,
    "aug-3": Aug3Processor,
    "aug-2": Aug2Processor,
}


DEFAULT_FOLDER_NAMES = {
    "cola": "CoLA",
    "sst": "SST-2",
    "sst-2": "SST-2",
    "mrpc": "MRPC",
    "stsb": "STS-B",
    "sts-b": "STS-B",
    "qqp": "QQP",
    "mnli": "MNLI",
    "qnli": "QNLI",
    "rte": "RTE",
    "wnli": "WNLI",
    "snli": "SNLI",
}

glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst": 2,
    "sst-2": 2,
    "stsb": 1,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "snli":3,
    'aug-4': 4,
    'aug-3': 3,
    'aug-2': 2,
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

    def get_train_df(self):
        return self.processor.get_train_df(self.data_dir)

    def get_dev_df(self, multi_testset_times=1):
        return self.processor.get_dev_df(self.data_dir, multi_testset_times)

    def get_labels(self):
        return self.processor.get_labels()


def get_task(task_name, data_dir):
    task_name = task_name.lower()
    task_processor = PROCESSORS[task_name]()
    if data_dir is None:
        data_dir = os.path.join(os.environ["GLUE_DIR"], DEFAULT_FOLDER_NAMES[task_name])
    return Task(task_name, task_processor, data_dir)
