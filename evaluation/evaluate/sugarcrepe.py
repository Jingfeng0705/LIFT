import os
import torch
import sys
from tqdm import tqdm
import argparse
import json
from PIL import Image
import torchvision.transforms as transforms

from utils.misc import load_models, load_llm_models_and_tokenizers, llm_model_embed_captions
from LIFT.open_clip import get_tokenizer
from LIFT.training.params import parse_args



def zeroshot_parse_args(args):
    parser = argparse.ArgumentParser('SugarCrepe Zero-shot Evaluation')

    # extra arguments in addition to default arguments in params.py
    parser.add_argument("--llm-model", type=str, default="", help="The path or HF link to the LLM-based text encoder, only used in LIFT")
    parser.add_argument("--batch_size", default=512, type=int, help="The evaluation batch size")
    parser.add_argument("--data_path", default='/public/datasets/COCO2017/val2017', type=str, help="The path to the evaluation dataset")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action='store_true', help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU")
    
    return parse_args(args, extra_arguments_parser=parser)



@torch.no_grad()
def text_retrieval(pos_text, neg_text, image, model, transform, llm_model, tokenizer, device, args):
    if not llm_model:
        pos_text = tokenizer(pos_text).to(device)
        pos_text_embedding = model.encode_text(pos_text, normalize=True)
        neg_text = tokenizer(neg_text).to(device)
        neg_text_embedding = model.encode_text(neg_text, normalize=True)
    else:
        pos_text_embedding = llm_model_embed_captions(args.llm_model, llm_model, tokenizer, [pos_text])
        neg_text_embedding = llm_model_embed_captions(args.llm_model, llm_model, tokenizer, [neg_text])
    image_embedding = model.encode_image(transform(image).unsqueeze(dim=0).to(device), normalize=True)
    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    image_embedding = image_embedding.to(dtype=pos_text_embedding.dtype)

    pos_score = pos_text_embedding @ image_embedding.t()
    neg_score = neg_text_embedding @ image_embedding.t()
    return 1 if pos_score.item() > neg_score.item() else 0



def evaluate(dataset, model, transform, llm_model, tokenizer, device, args):
    metrics = {}

    for c, data_dict in dataset.items():
        correct_cnt = 0
        for i, data in tqdm(data_dict.items(), desc=f'Evaluating Task {c}'):
            image_path = os.path.join(args.data_path, data['filename'])
            image = Image.open(image_path)
            try:
                correct = text_retrieval(data['caption'], data['negative_caption'], image, model, transform, llm_model, tokenizer, device, args)
                correct_cnt += correct
            except:
                print("Text retrieval error")
        count = len(data_dict)
        metrics[c] = correct_cnt / count
        print(f"Task {c} Accuracy: {metrics[c]}")
    return metrics



def main(args):
    args = zeroshot_parse_args(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dict = {
        'add_obj'    : './extra_zeroshot_datasets/sugar_crepe/add_obj.json',
        'add_att'    : './extra_zeroshot_datasets/sugar_crepe/add_att.json',
        'replace_obj': './extra_zeroshot_datasets/sugar_crepe/replace_obj.json',
        'replace_att': './extra_zeroshot_datasets/sugar_crepe/replace_att.json',
        'replace_rel': './extra_zeroshot_datasets/sugar_crepe/replace_rel.json',
        'swap_obj'   : './extra_zeroshot_datasets/sugar_crepe/swap_obj.json',
        'swap_att'   : './extra_zeroshot_datasets/sugar_crepe/swap_att.json',
    }
    dataset = {}
    for c, data_path in data_dict.items():
        dataset[c] = json.load(open(data_path, 'r', encoding='utf-8'))

    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Load LIFT or CLIP model
    model = load_models(args, device)

    # Load LLM models and tokenizers
    is_clip = not (args.llm_model)
    tokenizer, llm_model = None, None
    if is_clip:
        tokenizer = get_tokenizer(args.model, context_length=model.context_length)
    else:
        llm_model, tokenizer = load_llm_models_and_tokenizers(args.llm_model, device)

    # Evaluate on the tasks
    metrics = evaluate(dataset, model, transform_val, llm_model, tokenizer, device, args)
    for k, v in metrics.items():
        print(f"{k}: {v}")



if __name__ == "__main__":
    main(sys.argv[1:])