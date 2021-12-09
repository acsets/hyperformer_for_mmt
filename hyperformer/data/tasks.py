"""Implements different tasks and defines the processors to convert each dataset
to a sequence to sequence format."""
from collections import OrderedDict

import abc
import datasets
import functools
import logging
import numpy as np
import torch
from hyperformer.metrics import metrics
from typing import Callable, Dict, Mapping, List

from .utils import round_stsb_target, compute_task_max_decoding_length

logger = logging.getLogger(__name__)
from datasets import set_caching_enabled
set_caching_enabled(False)

class AbstractTaskDataset(abc.ABC):
    """Defines the abstract class for all the tasks.
    name: the name of the task.
    task_specific_config: specifies the special configuration needs
        to be passed to encoder when decoding each task. Since different
        tasks, have different output space, the maximum decoding length
        varies based on the tasks.
    preprocessor: a processor to convert the given dataset to the sequence
        to sequence format.
    metrics: specifies the metrics to evaluate the task based on them.
    split_to_data_split: since not all the time, different splits of the
        datasets are available, we define a mapping from the wanted split
        to the existing dataset splits.
    small_datasets_without_all_splits: List of strings, defines the name
        of all low-resource tasks in which not all train/test/validation
        splits are available.
    large_data_without_all_splits: List of strings, defines the name of
        all high-resource tasks in which not all train/test/validation
        splits are available.
    """
    name = NotImplemented
    task_specific_config: Dict = NotImplemented
    preprocessor: Callable = NotImplemented
    metrics: List[Callable] = NotImplemented
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}

    small_datasets_without_all_splits = ["cola", "wnli", "rte", "trec", "superglue-cb", "sick",
                                         "mrpc", "stsb", "imdb", "commonsense_qa", "superglue-boolq"]
    large_data_without_all_splits = ["yelp_polarity", "qqp", "qnli",
                                     "social_i_qa", "cosmos_qa", "winogrande", "hellaswag", "sst2"]

    def __init__(self, seed=42):
        self.seed = seed

    def get_sampled_split(self, split: int, n_obs: int = None):
        # If the requested number of observation is more than dataset
        # size we reset it to the maximum available.
        split = self.split_to_data_split[split]
        dataset = self.load_dataset(split)
        total_size = len(dataset)
        n_obs = self.check_n_obs(n_obs, total_size)
        if n_obs is not None:
            split = split + "[:{}]".format(n_obs)
        return split

    def get_shuffled_sampled_split(self, split: int, n_obs: int = None):
        # Defines the random generator.
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # If the requested number of observation is more than dataset
        # size we reset it to the maximum available.
        mapped_split = self.split_to_data_split[split]
        dataset = self.load_dataset(mapped_split)
        # shuffle the dataset and get the random samples.
        train_size = len(dataset)
        indices = torch.randperm(train_size, generator=generator).tolist()
        dataset = self.select_dataset_samples(indices, dataset, n_obs=n_obs)
        return dataset

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def select_dataset_samples(self, indices, dataset, n_obs: int = None):
        """
        Given a dataset for the split, obtains the sample indices for this split
        and returns the subsampled dataset.
        :param indices: the selected indices.
        :param dataset: dataset corresponding to this split.
        :return: subsampled dataset.
        """
        n_obs = self.check_n_obs(n_obs, len(indices))
        indices = indices[:n_obs] if n_obs is not None else indices
        return dataset.select(indices)

    def load_dataset(self, split: int): #this will be overrided
        return datasets.load_dataset(self.name, split=split, script_version="master")

    def get_train_split_indices(self, split):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split["train"]
        dataset = self.load_dataset(mapped_split)
        train_size = len(dataset)
        indices = torch.randperm(train_size, generator=generator).tolist()
        validation_size = 1000
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def get_half_validation_indices(self, split):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split["validation"]
        dataset = self.load_dataset(mapped_split)
        validation_size = len(dataset)
        indices = torch.randperm(validation_size, generator=generator).tolist()
        if split == "validation":
            return indices[:(validation_size // 2)]
        else:
            return indices[validation_size // 2:]

    def get_dataset(self, split, n_obs=None, add_prefix=True, split_validation_test=False):
        
        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_half_validation_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_train_split_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        else:
            if n_obs == -1:
                dataset = self.load_dataset(split=split)
            else:
                # shuffles the data and samples it.
                dataset = self.get_shuffled_sampled_split(split, n_obs)
        
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                           remove_columns=dataset.column_names)
        


    def seq2seq_format(self, src_strs: List[str], tgt_strs: List[str],
                       add_prefix: bool = False, prefix: str = None):
        src_prefix = self.name if prefix is None else prefix
        src_strs = [src_prefix] + src_strs if add_prefix else src_strs
        return {"src_texts": ' '.join(src_strs),
                "tgt_texts": ' '.join(tgt_strs),
                "task": self.name}


class IWSLT2017RONL(AbstractTaskDataset):
    name = "iwslt2017-ro-nl"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-nl"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("iwslt2017", 'iwslt2017-ro-nl',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["ro"]]
        tgt_texts = [example['translation']["nl"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate Romanian to Dutch")


class IWSLT2017ENNL(AbstractTaskDataset):
    name = "iwslt2017-en-nl"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"en-nl"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("iwslt2017", 'iwslt2017-en-nl',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["nl"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate English to Dutch")


class WMT16ENROTaskDataset(AbstractTaskDataset):
    name = "wmt16-en-ro"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair,
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["ro"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate English to Romanian")

class WMT16ROENTaskDataset(AbstractTaskDataset):
    name = "wmt16-ro-en"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair,
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["ro"]]
        tgt_texts = [example['translation']["en"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate Romanian to English")


class WMT16ENCSTaskDataset(AbstractTaskDataset):
    name = "wmt16-en-cs"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"cs-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair,
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["cs"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate English to Czech")


class WMT16ENFITaskDataset(AbstractTaskDataset):
    name = "wmt16-en-fi"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"fi-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair,
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["fi"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate English to Finnish")

DATA_DIR = '' #put the directory where the data is stored here

class Americasnlp2021DatasetTemplate(AbstractTaskDataset):
    task_specific_config = {'max_length': 256, 'num_beams': 6, 'early_stopping': False} #cannot be placed inside __init__
    metrics = [metrics.bleu] #cannot be placed inside __init__
    def __init__(self, tgt, seed, dev_involve_training):
        super().__init__(seed)
        # self.task_specific_config = {'max_length': 256, 'num_beams': 6}
        self.src = 'es_XX'
        self.tgt = tgt
        self.lang_pair = f"{self.src}-{self.tgt}"
        self.name = f"americasnlp2021-{self.lang_pair}" #object name has to match the keys in TASK_MAPPING dict
        self.lang_pair_data_dir = f'{DATA_DIR}/{self.lang_pair}/bilingual_data'
        self.dev_involve_training = dev_involve_training
    def load_dataset(self, split):
        if split == 'train' and self.dev_involve_training == False:
            return datasets.load_dataset(path='json', name=self.name, split=split, data_files={'train':f'{self.lang_pair_data_dir}/train-{self.lang_pair}.jsonl'})
        elif split == 'train' and self.dev_involve_training == True:
            print(f'{self.lang_pair_data_dir}/train+0.9dev-{self.lang_pair}.jsonl')
            d = datasets.load_dataset(path='json', name=self.name, split=split, data_files={'train':f'{self.lang_pair_data_dir}/train+0.9dev-{self.lang_pair}.jsonl'})
            return d
        elif split == 'validation' and self.dev_involve_training == False:
            return datasets.load_dataset(path='json', name=self.name, split=split, data_files={'validation':f'{self.lang_pair_data_dir}/dev-{self.lang_pair}.jsonl'})
        elif split == 'validation' and self.dev_involve_training == True:
            return datasets.load_dataset(path='json', name=self.name, split=split, data_files={'validation':f'{self.lang_pair_data_dir}/0.1dev-{self.lang_pair}.jsonl'})
        elif split == 'test':
            return datasets.load_dataset(path='json', name=self.name, split=split, data_files = {'test':f'{self.lang_pair_data_dir}/test-{self.lang_pair}.jsonl'})
        else:
            raise ValueError('No such arguments')

    def preprocessor(self, example, add_prefix=True):
        try:
            src_texts = [example['translation'][self.src]]
            tgt_texts = [example['translation'][self.tgt]]
        except Exception as e:
            print(e)
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix=f"Translate {self.src} to {self.tgt}")

class AmericasNLP2021ESAYMDataset(Americasnlp2021DatasetTemplate):
    def __init__(self, seed):
        tgt = 'aym_XX'
        dev_involve_training = True
        super().__init__(tgt, seed, dev_involve_training)

class AmericasNLP2021ESBZDDataset(Americasnlp2021DatasetTemplate):
    def __init__(self, seed):
        tgt = 'bzd_XX'
        dev_involve_training = True
        super().__init__(tgt, seed, dev_involve_training)

class AmericasNLP2021ESCNIDataset(Americasnlp2021DatasetTemplate):
    def __init__(self, seed):
        tgt = 'cni_XX'
        dev_involve_training = True
        super().__init__(tgt, seed, dev_involve_training)

class AmericasNLP2021ESGNDataset(Americasnlp2021DatasetTemplate):
    def __init__(self, seed):
        tgt = 'gn_XX'
        dev_involve_training = True
        super().__init__(tgt, seed, dev_involve_training)

class AmericasNLP2021ESHCHDataset(Americasnlp2021DatasetTemplate):
    def __init__(self, seed):
        tgt = 'hch_XX'
        dev_involve_training = True
        super().__init__(tgt, seed, dev_involve_training)

class AmericasNLP2021ESNAHDataset(Americasnlp2021DatasetTemplate):
    def __init__(self, seed):
        tgt = 'nah_XX'
        dev_involve_training = True
        super().__init__(tgt, seed, dev_involve_training)

class AmericasNLP2021ESOTODataset(Americasnlp2021DatasetTemplate):
    def __init__(self, seed):
        tgt = 'oto_XX'
        dev_involve_training = True
        super().__init__(tgt, seed, dev_involve_training)

class AmericasNLP2021ESQUYDataset(Americasnlp2021DatasetTemplate):
    def __init__(self, seed):
        tgt = 'quy_XX'
        dev_involve_training = True
        super().__init__(tgt, seed, dev_involve_training)

class AmericasNLP2021ESSHPDataset(Americasnlp2021DatasetTemplate):
    def __init__(self, seed):
        tgt = 'shp_XX'
        dev_involve_training = True
        super().__init__(tgt, seed, dev_involve_training)

class AmericasNLP2021ESTARDataset(Americasnlp2021DatasetTemplate):
    def __init__(self, seed):
        tgt = 'tar_XX'
        dev_involve_training = True
        super().__init__(tgt, seed, dev_involve_training)



TASK_MAPPING = OrderedDict([
    ('americasnlp2021-es_XX-aym_XX', AmericasNLP2021ESAYMDataset),
    ('americasnlp2021-es_XX-bzd_XX', AmericasNLP2021ESBZDDataset),
    ('americasnlp2021-es_XX-cni_XX', AmericasNLP2021ESCNIDataset),
    ('americasnlp2021-es_XX-gn_XX', AmericasNLP2021ESGNDataset),
    ('americasnlp2021-es_XX-hch_XX', AmericasNLP2021ESHCHDataset),
    ('americasnlp2021-es_XX-nah_XX', AmericasNLP2021ESNAHDataset),
    ('americasnlp2021-es_XX-oto_XX', AmericasNLP2021ESOTODataset),
    ('americasnlp2021-es_XX-quy_XX', AmericasNLP2021ESQUYDataset),
    ('americasnlp2021-es_XX-shp_XX', AmericasNLP2021ESSHPDataset),
    ('americasnlp2021-es_XX-tar_XX', AmericasNLP2021ESTARDataset)
    ]
)


class AutoTask:
    @classmethod
    def get(self, task_name, seed=42):
        if task_name in TASK_MAPPING:
            return TASK_MAPPING[task_name](seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
