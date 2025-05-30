import torch
import lightning as L
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import AutoencoderKL
from tqdm.auto import tqdm

from src.models.pe import LAPE, SinPE, NoPE
from src.models.proj import ModalityProjector
from src.data.utils import check_multiple_choice_with_regex, top_k_top_p_filtering

class MultimodalLightningModel(L.LightningModule):
    def __init__(
        self, 
        enc_name="stabilityai/sd-vae-ft-ema", 
        lm_name="Qwen/Qwen-0.5B",
        token_reduction_mode='pool',
        token_reduction_factor=4,
        image_positional_encoding='none',
        max_image_length=256,
        lr_lm=1e-5,
        lr_enc=1e-5,
        lr_proj=1e-3,
        freeze_encoder=True,
        freeze_lm=True,
        cache_dir=None,
        max_lr_steps=1e6
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(lm_name, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name, trust_remote_code=True, cache_dir=cache_dir)
        self.lm_dim = self.lm.config.hidden_size
        self.image_encoder = AutoencoderKL.from_pretrained(enc_name, cache_dir=cache_dir).encoder
        self.image_pe = lambda x: torch.zeros_like(x).to(x.device)

        assert image_positional_encoding in ['sin', 'lape', 'none']
        if image_positional_encoding == 'sin':
            self.image_pe = SinPE(max_size=max_image_length, d_model=self.lm_dim)
        elif image_positional_encoding == 'lape':
            self.image_pe = LAPE(max_size=max_image_length, d_model=self.lm_dim)
        else:
            self.image_pe = NoPE()

        self.projector = ModalityProjector(
            input_dim=512, 
            hidden_dim=self.lm_dim,
            mode=token_reduction_mode,
            factor=token_reduction_factor
        )

        self.soi_token = nn.Parameter(torch.randn(1, 1, self.lm_dim), requires_grad=True)  
        self.eoi_token = nn.Parameter(torch.randn(1, 1, self.lm_dim), requires_grad=True)  

        self.lr_lm = lr_lm
        self.lr_enc = lr_enc
        self.lr_proj = lr_proj
        self.max_lr_steps = max_lr_steps

        if freeze_encoder:
            self.freeze_module(self.image_encoder)
        else:
            self.unfreeze_module(self.image_encoder)
        if freeze_lm:
            self.freeze_module(self.lm)
        else:
            self.unfreeze_module(self.lm)

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True
    

    def encode_image(self, x):
        r"""The forward method of the `Encoder` class."""
        h = self.image_encoder.conv_in(x)
        if self.image_encoder.training and self.image_encoder.gradient_checkpointing:
    
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
    
            # down
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.image_encoder.down_blocks:
                    h = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), h, use_reentrant=False
                    )
                # middle
                h = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.image_encoder.mid_block), h, use_reentrant=False
                )
            else:
                for down_block in self.image_encoder.down_blocks:
                    h = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), h)
                # middle
                h = torch.utils.checkpoint.checkpoint(create_custom_forward(self.image_encoder.mid_block), h)
    
        else:
            # down
            for down_block in self.image_encoder.down_blocks:
                h = down_block(h)
    
            # middle
            h = self.image_encoder.mid_block(h)
            h = self.image_encoder.conv_norm_out(h)
            h = self.image_encoder.conv_act(h)
            return h
        
    def forward(self, image, input_ids, attention_mask=None, targets=None):
        B = image.size(0)

        h_img = self.projector(
            self.encode_image(image)
        )
        
        # Add separator tokens
        soi_token = self.soi_token.expand(B, 1, -1)             # [B, 1, D_lm]
        eoi_token = self.eoi_token.expand(B, 1, -1)             # [B, 1, D_lm]

        h_text = self.lm.model.embed_tokens(input_ids) 
        h = torch.cat(
            [soi_token, h_img + self.image_pe(h_img), eoi_token, h_text], 
            dim=1
        )  # [B, 1+N_img+1+L, D]

        if attention_mask is not None:
            img_mask = torch.ones(B, h_img.size(1) + 2, device=image.device)
            attention_mask = torch.cat([img_mask, attention_mask], dim=1)

        output = self.lm(inputs_embeds=h, attention_mask=attention_mask)
        logits = output.logits[:, h_img.size(1)+2:, :]
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                targets.reshape(-1), 
                ignore_index=-100
            )

        return output.logits, loss

    def step(self, batch, mode='train'):
        logits, loss = self.forward(
            batch['image'], 
            batch['input_ids'], 
            batch['attention_mask'], 
            batch['labels']
        )
        self.log(f"{mode}_loss", loss.item(), sync_dist=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, mode='train')
    
    def validation_step(self, batch, batch_idx):
        self.step(batch, mode='val')

    def test_step(self, batch, batch_idx):
        correct_answer = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        c = 0
        n = 0
        pred = self.run(
            batch['input_ids'], 
            batch['image'], 
            attention_mask=batch['attention_mask'], 
            device=batch['image'].device,
            max_new_tokens=1,
            temperature=0.2, top_k=30, top_p=0.95,
            verbose=False
        )
        for p,g in zip(pred, correct_answer):
            p = p.strip().lower()
            g = g.strip().lower()
            if p.isdigit():
                p = int(p)
                if 0 < p < 5:
                    p = ['a', 'b', 'c', 'd'][int(p) - 1]
            c += int(p == g)
            n += 1
        acc = c/n
        self.log('mmstar_acc', acc, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {
                'params': list(self.projector.parameters()) +\
                    list(self.image_pe.parameters()) +\
                        [self.soi_token, self.eoi_token], 
                'lr': self.lr_proj
            },
            {'params': self.lm.parameters(), 'lr': self.lr_lm},
            {'params': self.image_encoder.parameters(), 'lr': self.lr_enc}
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_lr_steps,  # Number of steps per cycle (adjust based on your training)
            eta_min=1e-6  # Minimum learning rate
        )
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',  # This ensures the scheduler steps every batch
            'frequency': 1
        }
        return {'optimizer': optimizer , 'lr_scheduler': scheduler_config}
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @torch.no_grad
    def generate(
        self, 
        input_ids, image=None, attention_mask=None, 
        max_new_tokens=50, top_k=50, top_p=0.9, temperature=0.5, 
        greedy=False,
        verbose=False
    ):
        self.eval()
        
        h_text = self.lm.model.embed_tokens(input_ids)
        B = h_text.size(0)
        if image is not None:
            h_img = self.encode_image(image)
            h_img = self.projector(h_img)   
            # Add separator tokens
            soi_token = self.soi_token.expand(B, 1, -1)             # [B, 1, D_lm]
            eoi_token = self.eoi_token.expand(B, 1, -1)             # [B, 1, D_lm]
            if attention_mask is not None:
                img_mask = torch.ones(B, h_img.size(1) + 2, device=image.device)
                attention_mask = torch.cat([img_mask, attention_mask], dim=1)

            h = torch.cat(
                [soi_token, h_img + self.image_pe(h_img), eoi_token, h_text], 
                dim=1
            )  # [B, 1+N_img+1+L, D]
        else:
            h = h_text
        
        # Generate from combined embeddings using the lm
        # We need to use the lm's forward function and not its generate method
        # because we want to keep track of the image prefix
        outputs = h
        generated_tokens = torch.zeros((B, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype)

        prog = lambda x: x
        if verbose:
            prog = tqdm
        #Note: Here you could implement improvements like e.g. KV caching
        for i in prog(range(max_new_tokens)):
            last_token_logits = self.lm(inputs_embeds=outputs, attention_mask=attention_mask).logits[:, -1, :]
            if greedy:
                next_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(last_token_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits/temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            generated_tokens[:, i] = next_token.squeeze(-1)
            
            # Convert to embedding and append
            next_embd = self.lm.model.embed_tokens(next_token)
            outputs = torch.cat((outputs, next_embd), dim=1)

            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((B, 1), device=attention_mask.device)), dim=1)
        
        return generated_tokens

    def run(
        self, 
        input_ids, image=None, attention_mask=None, 
        max_new_tokens=50, top_k=50, top_p=0.9, 
        temperature=0.5, greedy=False,
        device='cuda',
        verbose=False
    ):
        self.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if image is not None:
            image = image.to(device)
        gen = self.generate(
            input_ids.to(device), image, attention_mask, 
            max_new_tokens=max_new_tokens, 
            top_k=top_k, top_p=top_p, 
            temperature=temperature, greedy=greedy, verbose=verbose
        )
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)
