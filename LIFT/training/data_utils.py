import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
import psutil
import time
import subprocess
from dataclasses import dataclass
from multiprocessing import Value
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle, pipelinefilter
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample



class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value



class SimpleShardList(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(self, urls, seed=None):
        """Iterate through the list of shards.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = [os.path.join(urls, child) for child in os.listdir(urls)]
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.seed = seed

    def __len__(self):
        return len(self.urls)

    def __iter__(self):
        """Return an iterator over the shards."""
        urls = self.urls.copy()
        if self.seed is not None:
            random.Random(self.seed).shuffle(urls)
        for url in urls:
            yield dict(url=url)



@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)



def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image



def filter_nan_value(sample):
    for s in sample:
        if isinstance(s, np.ndarray):
            if np.isnan(s).any():
                return False
    return True



def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True



def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample



def parquet_to_samples_nothrow_wrapper(image_root=None, fetch=False, recaptioned=True):
    def parquet_to_samples_nothrow(src):
        """Return an iterator that yields samples from the raw caption and the image (optional) folders.
        This iterator is used for offline embeddings generation (not fetch images) and CLIP's training (fetch images).
        
        Key args:
            image_root: path to the image folder
            fetch: whether to fetch images or not; false for embeddings generation, true for CLIP's training
            src: the returned value of the iterator SimpleShardList, a dict containing the path to the raw caption .parquet file
        """
        parquet = parquets_opener(src)
        samples = parquet_content_iter(parquet, image_root=image_root, fetch=fetch, recaptioned=recaptioned)
        return samples
    return parquet_to_samples_nothrow



def parquets_opener(src) -> Iterator[Dict[str, Any]]:
    for parq in src:
        yield parq['url']



def parquet_content_iter(data, image_root=None, fetch=False, recaptioned=True) -> Iterator[Dict[str, Any]]:
    for parquet in data:
        try:
            # read chunk as stream, one line per time
            parquet_file = pq.ParquetFile(parquet)
        except Exception as e:
            logging.info(f"Cannot open {parquet}. {e}")
            continue

        for batch in parquet_file.iter_batches(batch_size=1024, columns=['key', 're_caption', 'org_caption', 'extension']):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                key = row['key']
                caption = (row['re_caption'] if recaptioned else row['org_caption'])
                ext = row['extension']
                parquet_name = parquet.split("/")[-1].split(".")[0]

                out = {
                    '__parqname__': parquet,
                    '__fname__': key,
                    '__txt__': caption,
                }
                if fetch:
                    image_path = os.path.join(image_root, parquet_name, key)
                    with open(f"{image_path}.{ext}", 'rb') as f:
                        out[ext] = f.read()
                        if len(out[ext]) > 50000:  # skip large images FIXME: fix the hardcode value
                            del out
                            continue
                else:
                    # For offline embedding generation, we include image extension in filenames.
                    # They will be saved with caption embeddings and make our life much easier
                    # when loading caption embeddings and determining image extension (line 241).
                    out['__fname__'] = f"{key}.{ext}"
                
                yield out

            # Explicitly delete the DataFrame to free up memory
            del df, batch

        # Explicitly delete parquet_file to free up memory
        del parquet_file



def dir_to_samples_nothrow_wrapper(image_root=None, fetch=False):
    """Return an iterator that yields samples from the caption embedding and the image folders.
        This iterator is used for LIFT's training.
        
        Key args:
            image_root: path to the image folder
            fetch: whether to fetch images or not; true for LIFT's training
            src: the returned value of the iterator SimpleShardList, a dict containing the path to the folders that contain number-indexed small parquet files.
    """
    def dir_to_samples_nothrow(src):
        dir = dirs_opener(src)
        samples = dir_content_iter(dir, image_root=image_root, fetch=fetch)
        return samples
    return dir_to_samples_nothrow



def dirs_opener(src) -> Iterator[Dict[str, Any]]:
    for dir in src:
        yield dir['url']



def dir_content_iter(data, image_root=None, fetch=False) -> Iterator[Dict[str, Any]]:
    for dir in data:
        for parquet in os.listdir(dir):
            parquet = os.path.join(dir, parquet)

            try:
                parquet_file = pq.ParquetFile(parquet)
            except Exception as e:
                logging.info(f"Cannot open {parquet}. {e}")
                continue

            for batch in parquet_file.iter_batches(batch_size=100, columns=['key', 'embedding']):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    key = row['key']
                    ext = key.split(".")[-1]
                    embedding = row['embedding']

                    out = {
                        '__fname__': key.split(".")[0], # get rid of extension
                        '__embedding__': embedding,
                    }
                    if fetch:
                        image_path = os.path.join(image_root, dir.split("/")[-1].split(".")[0], key)
                        with open(image_path, 'rb') as f:
                            out[ext] = f.read()
                            if len(out[ext]) > 50000:  # skip large images to save virtual memory FIXME: fix the hardcode value
                                del out
                                continue
                    yield out

                # Explicitly delete the DataFrame to free up memory
                del df, batch

        # Explicitly delete parquet_file to free up memory
        del parquet_file



def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()



_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000



class detshuffle2(wds.PipelineStage):
    """A deterministic shuffle stage for webdataset pipelines. Different epoches are shuffled differently."""
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)



def default_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a collection of samples (dictionaries) and create a batch.

    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.

    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    assert isinstance(samples[0], (list, tuple)), type(samples[0])
    batched = list(zip(*samples))
    result = []
    for b in batched:
        if isinstance(b[0], (int, float)):
            if combine_scalars:
                b = np.array(list(b))
        elif isinstance(b[0], torch.Tensor):
            if combine_tensors:
                b = torch.stack(list(b))
        elif isinstance(b[0], np.ndarray):
            if combine_tensors:
                b = np.array(list(b))
        else:
            b = list(b)
        result.append(b)
    return result



def simple_zip_collation_fn(samples):
    assert isinstance(samples[0], (list, tuple)), type(samples[0])
    return list(zip(*samples))



def pytorch_simple_collation_fn(batch):
    return batch



def _shard_boundary_batched(
    data,
    batchsize=20,
    collation_fn=default_collation_fn
):
    """An iterator to batch samples. This implementation mimics wds.batched, but also yields incomplete batches"""
    batch = []
    current_shard = None

    for sample in data:
        shard = sample[2]  # sample = (tokenized text, key, url)
        if current_shard is None:
            current_shard = shard

        # Detect shard change
        if shard != current_shard:
            if batch:
                yield collation_fn(batch)  # Output incomplete batch
                batch = []
            current_shard = shard

        # Add to batch
        batch.append(sample)
        if len(batch) == batchsize:
            yield collation_fn(batch)
            batch = []

    # Yield any remaining incomplete batch
    if batch:
        yield collation_fn(batch)

shard_boundary_batched = pipelinefilter(_shard_boundary_batched)



def get_available_cpus():
    """Get a list of SLURM's available CPUs."""
    # Run the taskset command to get the CPU affinity
    result = subprocess.run(['taskset', '-pc', str(os.getpid())], capture_output=True, text=True)
    # Extract the CPU list from the output
    cpu_list_str = result.stdout.split(':')[-1].strip()
    # Parse the CPU list string into a list of integers
    available_cpus = []
    for part in cpu_list_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            available_cpus.extend(range(start, end + 1))
        else:
            available_cpus.append(int(part))
    return available_cpus



def worker_init_fn_wrapper(available_cpus):
    """Create a pytorch worker initialization function that uniformly pins workers to CPUs"""
    def worker_init_fn(worker_id):
        pid = os.getpid()
        try:
            rank = torch.distributed.get_rank()
        except:
            rank = 0 # non DDP mode
        affinity_core = available_cpus[rank * 8 + worker_id] # assume 8 GPUs per node
        p = psutil.Process(pid)
        p.cpu_affinity([affinity_core])  # Pin process to specific core
    return worker_init_fn