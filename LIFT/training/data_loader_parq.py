import webdataset as wds
import math
import os
from .data_loader_tar import get_imagenet

from .data_utils import (
    SharedEpoch,
    detshuffle2,
    _SHARD_SHUFFLE_SIZE,
    _SHARD_SHUFFLE_INITIAL,
    _SAMPLE_SHUFFLE_SIZE,
    _SAMPLE_SHUFFLE_INITIAL,
    parquet_to_samples_nothrow_wrapper,
    dir_to_samples_nothrow_wrapper,
    log_and_continue,
    filter_nan_value,
    DataInfo,
    shard_boundary_batched,
    SimpleShardList,
    worker_init_fn_wrapper,
    simple_zip_collation_fn,
    pytorch_simple_collation_fn
)



def get_raw_text_dataset(args, available_cpus, epoch=0, floor=False):
    """Return the dataloader for text embeddings generation; output raw captions.
    Dataset structure:
        /raw_text_data (the raw caption folder)
            /train-XXX0-of-XXXX.parquet
            /train-XXX1-of-XXXX.parquet
            /train-XXX2-of-XXXX.parquet
    """
    input_parqs = args.raw_text_data
    assert input_parqs is not None

    num_parqs = len(os.listdir(args.raw_text_data))
    num_samples = args.raw_text_num_samples

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    pipeline = [SimpleShardList(input_parqs)]
    # at this point we have an iterator over all the shards
    pipeline.extend([
        wds.split_by_node,
        wds.split_by_worker,
    ])
    pipeline.extend([
        # at this point, we have an iterator over the shards assigned to each worker at each node
        parquet_to_samples_nothrow_wrapper(image_root=args.train_data, fetch=False, recaptioned=args.recaptioned),
    ])

    parq_lambda = lambda parqname: parqname.split("/")[-1].split(".")[0]

    pipeline.extend([
        wds.rename(text="__txt__", filename="__fname__", parqname="__parqname__"),
        wds.map_dict(parqname=parq_lambda),
        wds.to_tuple("text", "filename", "parqname"),
        shard_boundary_batched(args.batch_size) # In embedding generation, we need to go over every single data file, so accept partial batches
    ])

    dataset = wds.DataPipeline(*pipeline)

    assert num_parqs >= args.workers * args.world_size, 'number of shards must be >= total workers'
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_init_fn_wrapper(available_cpus),
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)



def get_clip_train_dataset(args, available_cpus, epoch=0, floor=False):
    """Return the dataloader for CLIP's training; output both raw captions and images.
    Dataset structure:
        /train_data (the image folder)
            /train-XXXX-of-XXXX
                /XXXXXXXXXXX0.jpg
                /XXXXXXXXXXX1.jpg
        /raw_text_data (the raw caption folder)
            /train-XXX0-of-XXXX.parquet
            /train-XXX1-of-XXXX.parquet
            /train-XXX2-of-XXXX.parquet
    """
    input_parqs = args.raw_text_data
    assert input_parqs is not None

    num_parqs = len(os.listdir(args.raw_text_data))
    num_samples = args.train_num_samples

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    pipeline = [SimpleShardList(input_parqs)]

    # at this point we have an iterator over all the shards
    pipeline.extend([
        detshuffle2(
            bufsize=_SHARD_SHUFFLE_SIZE,
            initial=_SHARD_SHUFFLE_INITIAL,
            seed=args.seed,
            epoch=shared_epoch,
        ),
        wds.split_by_node,
        wds.split_by_worker,
    ])
    pipeline.extend([
        # at this point, we have an iterator over the shards assigned to each worker at each node
        parquet_to_samples_nothrow_wrapper(image_root=args.train_data, fetch=True, recaptioned=args.recaptioned),
        wds.shuffle(
            bufsize=_SAMPLE_SHUFFLE_SIZE,
            initial=_SAMPLE_SHUFFLE_INITIAL,
        ),
    ])

    pipeline.extend([
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(text="__txt__", image="jpg;png;jpeg;webp"),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, collation_fn=simple_zip_collation_fn, partial=False)
    ])

    dataset = wds.DataPipeline(*pipeline)

    assert num_parqs >= args.workers * args.world_size, 'number of shards must be >= total workers'
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_init_fn_wrapper(available_cpus),
        collate_fn=pytorch_simple_collation_fn,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)



def get_lift_train_dataset(args, available_cpus, epoch=0, floor=False):
    """Return the dataloader for LIFT's training; output both caption embeddings and images.
    Dataset structure:
        /train_data (the image folder)
            /train-XXXX-of-XXXX
                /XXXXXXXXXXX0.jpg
                /XXXXXXXXXXX1.jpg
        /embedded_text_data (the caption embedding folder)
            /train-XXX0-of-XXXX
                /0.parquet
                /1.parquet
                /2.parquet
            /train-XXX1-of-XXXX
            /train-XXX2-of-XXXX
    """
    input_dirs = args.embedded_text_data
    assert input_dirs is not None

    num_dirs = len(os.listdir(args.embedded_text_data))
    num_samples = args.train_num_samples

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    pipeline = [SimpleShardList(input_dirs)]

    # at this point we have an iterator over all the shards
    pipeline.extend([
        detshuffle2(
            bufsize=_SHARD_SHUFFLE_SIZE,
            initial=_SHARD_SHUFFLE_INITIAL,
            seed=args.seed,
            epoch=shared_epoch,
        ),
        wds.split_by_node,
        wds.split_by_worker,
    ])
    pipeline.extend([
        # at this point, we have an iterator over the shards assigned to each worker at each node
        dir_to_samples_nothrow_wrapper(image_root=args.train_data, fetch=True),
        wds.shuffle(
            bufsize=_SAMPLE_SHUFFLE_SIZE,
            initial=_SAMPLE_SHUFFLE_INITIAL,
        ),
    ])

    pipeline.extend([
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", embedding="__embedding__",), 
        wds.to_tuple("image", "embedding"),
        wds.select(filter_nan_value),
        wds.batched(args.batch_size, collation_fn=simple_zip_collation_fn, partial=False)
    ])

    dataset = wds.DataPipeline(*pipeline)

    assert num_dirs >= args.workers * args.world_size, 'number of shards must be >= total workers'
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_init_fn_wrapper(available_cpus),
        collate_fn=pytorch_simple_collation_fn,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)



def get_data(args, available_cpus, preprocess_fns=None, epoch=0):
    data = {}

    if "llm_model" in args:
        # text embeddings offline generation
        data["raw_text"] = get_raw_text_dataset(args, available_cpus, epoch=epoch)
    else:
        if args.text_embed_dim:
            # LIFT
            data["train"] = get_lift_train_dataset(args, available_cpus, epoch=epoch)
        else:
            # CLIP
            data["train"] = get_clip_train_dataset(args, available_cpus, epoch=epoch)

    return data