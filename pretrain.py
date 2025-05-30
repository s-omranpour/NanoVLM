import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

import argparse
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.models.vlm import MultimodalLightningModel
from src.data.datasets import LlavaPretrainDataset
from src.data.utils import get_split_loaders


CACHE_DIR = "/scratch/s/soorism/hf_cache"

def main():
    parser = argparse.ArgumentParser(
        "OmniVLMPlus", add_help=False
    )
    parser.add_argument(
        "--lm_name", default='HuggingFaceTB/SmolLM2-360M', type=str, help="language model name"
    )
    parser.add_argument(
        "--token_reduction", default='shuffle', type=str, help="pool/shuffle"
    )
    parser.add_argument(
        "--reduction_factor", default=4, type=int, help="image token reduction factor"
    )
    parser.add_argument(
        "--image_pe", default='lape', type=str, help="image positional encoding"
    )
    parser.add_argument(
        "--image_size", default=224, type=int, help="image size"
    )
    parser.add_argument(
        "--max_image_length", default=50, type=int, help="max_image_length"
    )
    parser.add_argument(
        "--max_text_length", default=64, type=int, help="max_text_length"
    )
    parser.add_argument(
        "--batch_size", default=256, type=int, help="batch size"
    )
    parser.add_argument(
        "--num_workers", default=6, type=int, help="number of dataloader workers"
    )
    args = parser.parse_args()
    
    
    ## Model
    model = MultimodalLightningModel(
        enc_name="stabilityai/sd-vae-ft-ema", 
        lm_name=args.lm_name,
        token_reduction_mode=args.token_reduction,
        token_reduction_factor=args.reduction_factor,
        image_positional_encoding=args.image_pe,
        max_image_length=args.max_image_length,
        lr_lm=1e-4,
        lr_enc=1e-4,
        lr_proj=3e-3,
        freeze_encoder=True,
        freeze_lm=True,
        cache_dir=CACHE_DIR,
        max_lr_steps=50000
    )
    model.lm.train()
    model.image_encoder.train()
    
    ## Dataset
    root = '/home/mila/s/soroush.omranpour/scratch/hf_datasets/llava-pretrain/'
    dataset = LlavaPretrainDataset(
        meta_dir=root + 'blip_laion_cc_sbu_558k_meta.json', 
        img_dir=root + 'images/', 
        eos_token=model.tokenizer.eos_token, 
        max_length=args.max_text_length,
        image_size=args.image_size
    )
    train_loader, val_loader = get_split_loaders(
        dataset, 
        tokenizer=model.tokenizer,
        val_size=0.1, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    ## Trainer
    project = 'omnivlm+-pretrain-llava'
    name = f'{args.lm_name}-{args.token_reduction}-{args.image_pe}'
    wandb_logger = WandbLogger(
        project=project,
	name=name,
        save_dir='/home/mila/s/soroush.omranpour/scratch/wandb'
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"weights/{project}/",
        filename=name + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.01, patience=5, verbose=False, mode="min"
    )
    trainer = L.Trainer(
        max_epochs=50,
        devices=4,
        strategy='ddp_find_unused_parameters_true',
        accelerator="gpu", 
        logger=wandb_logger,
        accumulate_grad_batches=1,
        gradient_clip_val=1.,
        num_nodes=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=False
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
