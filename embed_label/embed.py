import logging
import os
import sys
from datetime import datetime
import time
import argparse

from LIFT.training.distributed import init_distributed_device
from LIFT.training.params import parse_args
from LIFT.training.data_loader_parq import get_data
from LIFT.training.data_utils import get_available_cpus
from LIFT.training.logger import setup_logging

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed

from utils.misc import load_llm_models_and_tokenizers, llm_model_embed_captions


def embed_parse_args(args):
    parser = argparse.ArgumentParser('Generate raw text embedding')

    # extra arguments in addition to default arguments in params.py
    parser.add_argument(
        "--embedded-labels",
        type=str,
        default=None,
        help="The path to the destination folder of caption embeddings",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="",
        help="The path or HF link to the LLM model used for embedding",
    )
    return parse_args(args, extra_arguments_parser=parser)



def save_tensor_as_parquet(tensor, filenames, parquet_save_path):
    # Convert rows to list format
    numpy_as_arrow = pa.array(tensor.tolist(), type=pa.list_(pa.float32()))
    # Convert filenames to a PyArrow array
    filenames_as_arrow = pa.array(filenames, type=pa.string())

    # Create a PyArrow table
    table = pa.Table.from_arrays([filenames_as_arrow, numpy_as_arrow], names=["key", "embedding"])
    pq.write_table(table, parquet_save_path)



def main(args):
    args = embed_parse_args(args)
    global_cpus = get_available_cpus()

    # Setup text logger
    current_time = datetime.now()
    log_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")+'.txt' # use current time as the log name
    os.makedirs(args.logs, exist_ok=True)
    args.log_path = os.path.join(args.logs, log_name)
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    if not os.path.exists(args.embedded_labels) and (args.world_size == 1 or torch.distributed.get_rank() == 0):
        logging.info("No directory found, making new one!")
        os.makedirs(args.embedded_labels, exist_ok=True)
    else:
        logging.info("Directory already exists, skipping creation.")

    logging.info(f"Arguments: {args}")

    try:
        assert args.global_batch_size % args.world_size == 0
    except AssertionError as e:
        logging.error("Global batch size cannot be divided evenly by number of machine")
    args.batch_size = int(args.global_batch_size / args.world_size)
    logging.info(f"Batch size per gpu is {args.batch_size}")

    model, tokenizer = load_llm_models_and_tokenizers(args.llm_model, device)
    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    
    data = get_data(args, global_cpus)
    dataloader = data["raw_text"].dataloader


    # ---------- Caption embedding generation ---------- 
    with torch.no_grad():
        end = time.time()
        total = 0
        last_parqname = None

        for step, batch in enumerate(dataloader):
            torch.cuda.empty_cache()

            text, keys, parqnames = batch
            batch_size = len(parqnames)

            # we divide a big parq to a bunch of small parq, index is the name for each small parq
            # filename is the key value in each small parq
            if not last_parqname or last_parqname != parqnames[0]:
                if last_parqname:
                    logging.info(f"Rank {torch.distributed.get_rank()}: {last_parqname} used {time.time() - end} seconds. Start to handle parq {parqnames[0]}") # we log each time we start to handle a new parq
                    end = time.time()
                    logging.info(f"Rank {torch.distributed.get_rank()}: {last_parqname} has {total} samples")
                    total = 0
                last_parqname = parqnames[0]
                index = 0
            else:
                index += 1

            small_parq_save_dir = os.path.join(args.embedded_labels, parqnames[0])
            if index == 0: # we start to handle a new parq
                if not os.path.exists(small_parq_save_dir):
                    os.mkdir(small_parq_save_dir)
                else:
                    # if all ranks have rolled over, we can stop manually
                    logging.info(f"Rank {torch.distributed.get_rank()} has rolled over {parqnames[0]}!")
            small_parq_save_path = os.path.join(small_parq_save_dir, f"{index}.parquet")

            # generate embeddings
            embeddings = llm_model_embed_captions(args.llm_model, model.module, tokenizer, text, max_length=args.caption_length)

            save_tensor_as_parquet(
                embeddings,
                keys,
                small_parq_save_path
            )
            total += batch_size


if __name__ == "__main__":
    main(sys.argv[1:])
