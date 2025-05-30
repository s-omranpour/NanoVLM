import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

import argparse
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.models.vlm import MultimodalLightningModel
from src.data.datasets import MMStarDataset, CauldronDataset
from src.data.utils import get_split_loaders, get_test_loader


CACHE_DIR = "/home/mila/s/soroush.omranpour/scratch/hf_cache/"

def main():
    parser = argparse.ArgumentParser(
        "OmniVLMPlus", add_help=False
    )
    parser.add_argument(
        "--ckpt_path", default='weights/', type=str, help="checkpoint path"
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
        "--max_text_length", default=128, type=int, help="max_text_length"
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="batch size"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="number of dataloader workers"
    )
    args = parser.parse_args()
    
    
    ## Model
    model = MultimodalLightningModel.load_from_checkpoint(
        args.ckpt_path,
        token_reduction_mode=args.token_reduction,
        token_reduction_factor=args.reduction_factor,
        image_positional_encoding=args.image_pe,
        max_image_length=args.max_image_length,
        lr_lm=1e-4,
        lr_enc=1e-4,
        lr_proj=1e-3,
        freeze_encoder=True,
        freeze_lm=True,
    )
    model.lm.train()
    model.image_encoder.train()
    
    # Dataset
    dataset = CauldronDataset(
        subsets=[
            'ai2d', 'aokvqa', 'chart2text', 'chartqa', 'clevr', 'clevr_math', 'cocoqa', 'datikz',\
            'diagram_image_to_text', 'docvqa', 'figureqa', 'finqa', 'geomverse', 'hateful_memes',\
            'hitab', 'iam', 'iconqa', 'infographic_vqa', 'intergps', 'mapqa', 'multihiertt', 'okvqa',\
            'raven', 'robut_sqa', 'robut_wikisql', 'robut_wtq', 'scienceqa', 'screen2words',\
            'st_vqa', 'tabmwp', 'tallyqa', 'tat_qa', 'textcaps', 'textvqa', 'tqa', 'vistext',\
            'visual7w', 'visualmrc', 'vqarad', 'vqav2', 'vsr',\
            'dvqa', 'localized_narratives', 'ocrvqa', 'plotqa', 'rendered_text', 'websight'
        ],
        eos_token=model.tokenizer.eos_token, 
        max_length=args.max_text_length,
        image_size=args.image_size,
        cache_dir='/home/mila/s/soroush.omranpour/scratch/hf_cache'
    )
    train_loader, val_loader = get_split_loaders(
        dataset, 
        tokenizer=model.tokenizer,
        val_size=0.1, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    test_loader = get_test_loader(
        MMStarDataset(
            cache_dir=CACHE_DIR, 
            eos_token=model.tokenizer.eos_token, 
            image_size=224, 
            max_length=64
        ),
        tokenizer=model.tokenizer
    )
    
    ## Trainer
    project = 'omnivlm+-finetune'
    name = f'{args.ckpt_path.split("/")[-1]}--finetuned'
    wandb_logger = WandbLogger(
        project=project,
        name=name,
        save_dir='scratch/s/soorism/.wandb'
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"weights/{project}/",
        filename=name+"--{epoch:02d}-{val_loss:.2f}",
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
        enable_checkpointing=True,
        enable_progress_bar=False
    )
    trainer.fit(model, train_loader, val_loader)

    ## Eval on MMStar
    eval_trainer = L.Trainer(devices=1, accelerator='gpu', logger=wandb_logger)
    eval_trainer.test(model, dataloaders=test_loader)
    


if __name__ == '__main__':
    main()
