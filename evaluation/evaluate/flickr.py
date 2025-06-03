import os
import torch
import sys
from tqdm import tqdm
import argparse
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

from utils.misc import load_models, load_llm_models_and_tokenizers, llm_model_embed_captions
from LIFT.open_clip import get_tokenizer
from LIFT.training.params import parse_args



def zeroshot_parse_args(args):
    parser = argparse.ArgumentParser('Flickr Zero-shot Evaluation')

    # extra arguments in addition to default arguments in params.py
    parser.add_argument("--llm-model", type=str, default="", help="The path or HF link to the LLM-based text encoder, only used in LIFT")
    parser.add_argument("--batch_size", default=512, type=int, help="The evaluation batch size")
    parser.add_argument("--data_path", default='/public/datasets/flickr30k/', type=str, help="The path to the evaluation dataset")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action='store_true', help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU")
    parser.add_argument("--topk", type=int, default=1, help="Top-k accuracy to compute")
    
    return parse_args(args, extra_arguments_parser=parser)



def load_embeddings(model, raws, device, args, batch_size=100, clip=False, text=True):
    embeddings = []

    if text:
        print("Embedding texts...")
        if not clip:
            llm_model, tokenizer = load_llm_models_and_tokenizers(args.llm_model, device)
            print("Extracting LLM embeddings for each zero-shot class in batches...")
            with torch.no_grad():
                for i in tqdm(range(0, len(raws), batch_size)):
                    batch_text = raws[i:i + batch_size]
                    batch_embeddings = llm_model_embed_captions(args.llm_model, llm_model, tokenizer, batch_text)
                    embeddings.append(batch_embeddings)

        else: # no llm model, using clip's text encoder
            tokenizer = get_tokenizer(args.model, context_length=model.context_length)

            with torch.no_grad():
                for i in tqdm(range(0, len(raws), batch_size)):
                    batch_text = raws[i:i + batch_size]
                    batch_tokens = tokenizer(batch_text).to(device)
                    batch_embeddings = model.encode_text(batch_tokens, normalize=True)
                    embeddings.append(batch_embeddings)
    
    else:
        print("Embedding images...")
        with torch.no_grad():
            for i in tqdm(range(0, len(raws), batch_size)):
                batch_image = torch.stack(raws[i:i + batch_size]).to(device)
                batch_embeddings = model.encode_image(batch_image, normalize=True)
                embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0).to(device)
    return embeddings



def calculate_topk_accuracy(querys, keys, k=1):
    querys /= querys.norm(dim=-1, keepdim=True)
    keys /= keys.norm(dim=-1, keepdim=True)

    total_correct = 0
    total = querys.size(0)

    with torch.no_grad():
        # Compute similarity and find top-k predictions
        querys = querys.to(dtype=keys.dtype)
        similarity = (100.0 * querys @ keys.T).softmax(dim=-1)
        _, topk_indices = similarity.topk(k, dim=-1)

        # Check if true labels are in top-k predictions
        for i in range(total):
            if i in topk_indices[i]:
                total_correct += 1
    
    topk_accuracy = total_correct / total
    return topk_accuracy



def main(args):
    args = zeroshot_parse_args(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Load the CSV file
    csv_path = os.path.join(args.data_path, "flickr_annotations_30k.csv")
    image_path = os.path.join(args.data_path, "flickr30k-images/")
    data = pd.read_csv(csv_path)
    
    # Extract captions and image paths from the CSV file
    captions = data.loc[data["split"] == "test", "raw"].tolist()
    captions = [eval(cap)[0] for cap in captions]
    print("Extracting and preprocessing images...")
    images = data.loc[data["split"] == "test", "filename"]
    images = [transform_val(Image.open(os.path.join(image_path, img))) for img in images]

    # Load LIFT or CLIP model
    model = load_models(args, device)

    # Generate class and image embeddings
    is_clip = not (args.llm_model)
    captions_embeddings = load_embeddings(model, captions, device, args, clip=is_clip, text=True)
    images_embeddings = load_embeddings(model, images, device, args, clip=is_clip, text=False)

    # Evaluate Top-k accuracy
    topk_accuracy = calculate_topk_accuracy(images_embeddings, captions_embeddings, k=args.topk)
    print(f"Top-{args.topk} I2T Flickr30k Accuracy: {topk_accuracy * 100:.2f}%")
    topk_accuracy = calculate_topk_accuracy(captions_embeddings, images_embeddings, k=args.topk)
    print(f"Top-{args.topk} T2I Flickr30k Accuracy: {topk_accuracy * 100:.2f}%")



if __name__ == "__main__":
    main(sys.argv[1:])