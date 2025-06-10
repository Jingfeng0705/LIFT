import torch
import sys
import os
from torch import Tensor

sys.path.append(os.path.join(os.getcwd(), 'LIFT'))
from open_clip import create_model_and_transforms  # type: ignore



def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]



def load_models(args, device):
    """
    Load LIFT or CLIP model with the specified arguments.
    """
    model_kwargs = {}
    # Architecture configs controlled by training script
    if args.text_embed_dim:
        model_kwargs['text_embed_dim'] = args.text_embed_dim
    if args.projector_layers:
        model_kwargs['projector_layers'] = args.projector_layers
    # control caption length in training args, model_kwargs will be merged into CLIP's config
    if args.caption_length and 'LIFT' not in args.model:
        model_kwargs['context_length'] = args.caption_length
    if args.simplistic_cos:
        model_kwargs['simplistic_cos'] = True

    model, _, _ = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        **model_kwargs,
    )

    model.to(device)
    model.eval()
    return model



def llm_model_embed_captions(llm_model_path, llm_model, tokenizer, captions, max_length=None):
    """
    Generate embeddings for the given captions using the specified LLM model.
    If your LLM model is not supported, feel free to add it in this big if else condition.
    """

    if 'nv-embed' in llm_model_path:
        if max_length is None:
            embeddings = llm_model.encode(captions)
        else:
            embeddings = llm_model.encode(captions, max_length=max_length)


    elif "vicuna" in llm_model_path:
        if max_length is None:
            text = tokenizer( # inference time no caption length limit for LLM text encoder
                captions,
                padding=True,         # Pad to the longest sequence in the batch
                truncation=True,      # Truncate to the maximum length of the model
                return_tensors="pt"       # Return as PyTorch tensors
            ).input_ids
        else:
            text = tokenizer( # inference time no caption length limit for LLM text encoder
                captions,
                max_length=max_length,
                padding=True,         # Pad to the longest sequence in the batch
                truncation=True,      # Truncate to the maximum length of the model
                return_tensors="pt"       # Return as PyTorch tensors
            ).input_ids

        batch_size = len(captions) # for the last batch of each parq, batch_size is not the same as args.batch_size
        attention_mask = (text != tokenizer.pad_token_id).int()
        last_non_pad_indices = attention_mask.sum(dim=1) - 1 # -1 for zero indexing, no automatically added eos token

        text = text.to(device=llm_model.device)
        attention_mask = attention_mask.to(device=llm_model.device)
        outputs = llm_model(text, attention_mask=attention_mask, output_hidden_states=True)
        del text
        del attention_mask
        embeddings = outputs.hidden_states[-1][torch.arange(batch_size), last_non_pad_indices]
        del outputs


    elif "linq" in llm_model_path or "sfr" in llm_model_path or "mistral" in llm_model_path:
        if max_length is None:
            text = tokenizer( # inference time no caption length limit for LLM text encoder
                captions,
                padding=True,         # Pad to the longest sequence in the batch
                truncation=True,      # Truncate to the maximum length of the model
                return_tensors="pt"       # Return as PyTorch tensors
            )
        else:
            text = tokenizer( # inference time no caption length limit for LLM text encoder
                captions,
                max_length=max_length,
                padding=True,         # Pad to the longest sequence in the batch
                truncation=True,      # Truncate to the maximum length of the model
                return_tensors="pt"       # Return as PyTorch tensors
            )

        text = text.to(device=llm_model.device)
        outputs = llm_model(**text)
        embeddings = last_token_pool(outputs.last_hidden_state, text['attention_mask'])


    else:
        raise ValueError(f"Unsupported LLM model: {llm_model_path}")


    return embeddings



def load_llm_models_and_tokenizers(llm_model_path, device):
    """
    Load the appropriate LLM model and tokenizer based on the provided model name.
    If your LLM model is not supported, feel free to add it in this big if else condition.
    """
    print("Loading LLM model and tokenizer...")

    if 'nv-embed' in llm_model_path:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(llm_model_path, torch_dtype=torch.float16, trust_remote_code=True).to(device)
        tokenizer = None


    elif "vicuna" in llm_model_path:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        tokenizer.add_eos_token = True # add eos to make embedding extract easier, which is not the default of vicuna model
        model = AutoModelForCausalLM.from_pretrained(llm_model_path, torch_dtype=torch.float16, trust_remote_code=True).to(device)

    
    elif "linq" in llm_model_path or "sfr" in llm_model_path:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        model = AutoModel.from_pretrained(llm_model_path, torch_dtype=torch.float16, trust_remote_code=True).to(device)


    elif "mistral" in llm_model_path:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        tokenizer.add_eos_token = True # add eos to make embedding extract easier, which is not the default of vanilla mistral
        tokenizer.pad_token = tokenizer.eos_token # add pad token to make it compatible with our framework
        tokenizer.additional_special_tokens = ["<unk>", "<s>", "</s>"]
        model = AutoModel.from_pretrained(llm_model_path, torch_dtype=torch.float16, trust_remote_code=True).to(device)


    else:
        raise ValueError(f"Unsupported LLM model: {llm_model_path}")
    

    model.eval()
    return model, tokenizer