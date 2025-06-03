import os
import torch
import sys
from torchvision import datasets
from tqdm import tqdm
import argparse
import json
import torchvision.transforms as transforms

from utils.misc import load_models, load_llm_models_and_tokenizers, llm_model_embed_captions
from LIFT.open_clip import get_tokenizer
from LIFT.training.params import parse_args



def zeroshot_parse_args(args):
    parser = argparse.ArgumentParser('ImageNet Zero-shot Evaluation')

    # extra arguments in addition to default arguments in params.py
    parser.add_argument("--llm-model", type=str, default="", help="The path or HF link to the LLM-based text encoder, only used in LIFT")
    parser.add_argument("--batch_size", default=512, type=int, help="The evaluation batch size")
    parser.add_argument("--data_path", default='/public/datasets/imagenet1k', type=str, help="The path to the evaluation dataset")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action='store_true', help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU")
    parser.add_argument("--topk", type=int, default=1, help="Top-k accuracy to compute")
    parser.add_argument("--class_names", type=str, default='./extra_zeroshot_datasets/imagenet_class_index.json', help="The path to the class name JSON file for ImageNet")
    
    return parse_args(args, extra_arguments_parser=parser)



def load_class_embeddings(model, class_names, device, args, text_embedding_bs=100, clip=False):
    """
    Process sentences 'a photo of a {label}' and extract hidden states of the last token before <eos>.
    """
    sentences = [f"It is a photo of a {label}" for label in class_names] # using the simplistic CLIP prompt engineering method
    embeddings = []

    if not clip:
        llm_model, tokenizer = load_llm_models_and_tokenizers(args.llm_model, device)
        print("Extracting LLM embeddings for each zero-shot class in batches...")
        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), text_embedding_bs)):
                batch_sentences = sentences[i:i + text_embedding_bs]
                batch_embeddings = llm_model_embed_captions(args.llm_model, llm_model, tokenizer, batch_sentences)
                embeddings.append(batch_embeddings)

    else: # no llm model, using clip's text encoder
        tokenizer = get_tokenizer(args.model, context_length=model.context_length)

        with torch.no_grad():
            for i in tqdm(range(0, len(sentences), text_embedding_bs)):
                batch_sentences = sentences[i:i + text_embedding_bs]
                batch_tokens = tokenizer(batch_sentences).to(device)
                batch_embeddings = model.encode_text(batch_tokens, normalize=True)
                embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0).to(device)
    return embeddings



def calculate_topk_accuracy(model, dataloader, text_embeddings, device, k=1):
    """
    Calculate the top-k accuracy for a given model and dataloader.
    """
    total_correct = 0
    total_images = 0

    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            # Encode image features
            image_features = model.encode_image(images, normalize=True)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.to(dtype=text_embeddings.dtype)

            # Compute similarity and find top-k predictions
            similarity = (100.0 * image_features @ text_embeddings.T).softmax(dim=-1)
            _, topk_indices = similarity.topk(k, dim=-1)

            # Check if true labels are in top-k predictions
            for i in range(labels.size(0)):
                if labels[i] in topk_indices[i]:
                    total_correct += 1
            total_images += labels.size(0)
    
    topk_accuracy = total_correct / total_images
    return topk_accuracy



def main(args):
    args = zeroshot_parse_args(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if not 'extra_zeroshot_datasets' in args.data_path:
        args.data_path = os.path.join(args.data_path, 'val')
    dataset_val = datasets.ImageFolder(args.data_path, transform=transform_val)

    # we only support single machine zero-shot evaluation
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    with open(args.class_names) as f:
        class_idx = json.load(f)
        idx2label = {value[0]: value[1] for _, value in class_idx.items()}
    class_names = [idx2label[cls].replace("_", " ") for cls in dataset_val.classes]

    # Load LIFT or CLIP model
    model = load_models(args, device)
    is_clip = not (args.llm_model)

    # Generate class embeddings
    class_embeddings = load_class_embeddings(model, class_names, device, args, clip=is_clip)

    # Evaluate the model on ImageNet
    print(f"Evaluating the model on ImageNet...")
    topk_accuracy = calculate_topk_accuracy(model, data_loader_val, class_embeddings, device, k=args.topk)
    print(f"Top-{args.topk} ImageNet Classification Accuracy: {topk_accuracy * 100:.2f}%")



if __name__ == "__main__":
    main(sys.argv[1:])