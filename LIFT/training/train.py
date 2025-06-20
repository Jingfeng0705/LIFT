import logging
import math
import os
import time
import torch

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from .distributed import is_master
from .precision import get_autocast



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model



def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()



def during_training_data_preprocess(tuple_images, tuple_texts, preprocess, args, tokenizer, device):
    """In our hardware setting, do the preprocessing in the training loop on GPU is faster than doing it in the dataloader on CPU. 
    Thus the dataloader only returns raw images and texts as tuples, and we do the preprocessing here.

    Key Args:
        tuple_texts: a tuple of texts, either str (CLIP) or numpy array (LIFT)
        preprocess: the preprocess function for images
        tokenizer: the tokenizer for texts, only for CLIP
    """
    images = []
    for img in tuple_images:
        images.append(preprocess(img))
    images = torch.stack(images, dim=0)
    del tuple_images
    
    if not args.text_embed_dim: # CLIP
        texts = tokenizer(list(tuple_texts))
    else: # LIFT
        texts = torch.stack([torch.from_numpy(text) for text in tuple_texts], dim=0)
    texts = texts.to(device=device, non_blocking=True) # text embedding in float32
    del tuple_texts

    return images, texts



def train_one_epoch(
        model, 
        data, 
        loss, 
        epoch,
        optimizer, 
        scaler, 
        scheduler, 
        dist_model, 
        preprocess_img,
        tokenizer,
        args, 
        tb_writer=None,
    ):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision) # if use amp mixed precision, input_dtype is None and autocast will handle everything
    preprocess = preprocess_img(device, input_dtype) # instantiate the image preprocess function with device and input_dtype

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        tuple_images, tuple_texts = batch # the dataloader returns a tuple of unpreprocessed images and texts (either str (CLIP) or numpy array (LIFT))
        images, texts = during_training_data_preprocess(tuple_images, tuple_texts, preprocess, args, tokenizer, device)
      
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, text=texts)
                logit_scale = model_out["logit_scale"] if 'logit_scale' in model_out else None
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)

            # clip the gradients for LIFT
            if 'LIFT' in args.model:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, text=texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            if hasattr(unwrap_model(model), 'logit_scale'):
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item() if logit_scale else None
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            log_msg = (
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
            )
            if logit_scale_scalar is not None:
                log_msg += f"Logit Scale: {logit_scale_scalar:.3f} "
            log_msg += loss_log
            logging.info(log_msg)

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": optimizer.param_groups[0]["lr"]
            } 
            if logit_scale_scalar is not None:
                log_data["scale"] = logit_scale_scalar            
            log_data.update({name:val.val for name,val in losses_m.items()})
            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

        #  adopt after-time step in clipa-v1
        completed_step = step + 1
        # Saving checkpoints every n step.
        if args.save_logs and args.save_every_n_steps != 0 and (completed_step % args.save_every_n_steps) == 0:
            checkpoint_dict = {
                "step": completed_step,
                "epoch": epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_path, f"step_{completed_step}.pt"),
            )

            if args.delete_prev_step_ckpt:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"step_{completed_step - args.save_every_n_steps}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

    # end for