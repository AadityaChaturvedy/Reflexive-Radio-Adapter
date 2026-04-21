import os
import sys
import shutil
import logging
import argparse
import re
import json
import random
import time
import gc
import ast
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import contextlib

# CRITICAL: Set memory optimization BEFORE any CUDA operations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_fscore_support
from tqdm.auto import tqdm

from log import StabilityMetricsTracker

# Update for PyTorch 2.0+ AMP syntax
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# Hugging Face & Peft
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("Bitsandbytes not found. Using standard AdamW.")

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    # System
    SEED = 42
    NUM_WORKERS = 2
    PIN_MEMORY = True

    # Paths
    DRIVE_ROOT = os.environ.get("RRA_DRIVE_ROOT", "data")
    DATA_SOURCE = os.environ.get("RRA_DATA_SOURCE", "data")
    LOCAL_DATA = os.environ.get("RRA_LOCAL_DATA", "data")
    OUTPUT_DIR = os.environ.get("RRA_OUTPUT_DIR", "outputs/run_default")

    # Model
    MODEL_ID = "google/gemma-2b"
    MAX_SEQ_LENGTH = 320
    VISION_CHECKPOINT = os.environ.get("RRA_VISION_CHECKPOINT", "")

    # Training
    EPOCHS = 10
    BATCH_SIZE = 4
    GRAD_ACCUMULATION = 4

    # Learning Rates
    LEARNING_RATE = 5e-6
    PROJ_LR = 5e-5
    VISION_LR = 1e-5

    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    WARMUP_RATIO = 0.15

    # LoRA
    LORA_R = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.05

    # Loss Weights
    LM_LOSS_WEIGHT = 1.0
    REFLEX_LOSS_WEIGHT = 2.0
    CLINICAL_LOSS_WEIGHT = 5.0

    # Validation & Checkpointing
    VAL_INTERVAL_STEPS = 300
    SANITY_GEN_TOKENS = 40
    EARLY_STOPPING_PATIENCE = 3

    # Architecture
    NUM_QUERIES = 32
    HIDDEN_DIM = 2048
    NUM_CLINICAL_CLASSES = 14

    # Focal Loss
    FOCAL_GAMMA_POS = 1
    FOCAL_GAMMA_NEG = 4

    # Generation
    GEN_TEMPERATURE = 0.7

cfg = Config()
logger = logging.getLogger(__name__)


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "training.log"))
        ],
        force=True,
    )

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed} (deterministic mode enabled)")

# ==============================================================================
# SPEED LAYER SETUP
# ==============================================================================
def setup_data_layer():
    """Copy data to local VM for faster I/O"""
    if os.path.exists(cfg.LOCAL_DATA):
        logger.info(f"Local data found at {cfg.LOCAL_DATA}")
        return cfg.LOCAL_DATA

    if os.path.exists(cfg.DATA_SOURCE):
        logger.info(f"Copying data from Drive to VM...")
        try:
            shutil.copytree(cfg.DATA_SOURCE, cfg.LOCAL_DATA)
            logger.info("Data copied successfully")
            return cfg.LOCAL_DATA
        except Exception as e:
            logger.warning(f"Copy failed: {e}. Using Drive path.")
            return cfg.DATA_SOURCE

    logger.warning("Data not found at default paths")
    return cfg.DATA_SOURCE

# ==============================================================================
# DATASET
# ==============================================================================
CHEXPERT_LABELS = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]

RARE_CONDITIONS = {'Pneumothorax', 'Fracture', 'Lung Lesion'}


class MIMICDataset(Dataset):
    """Dataset with pre-computed clinical labels and complexity from CSV"""

    def __init__(self, csv_path, data_root, tokenizer, transform):
        self.data = pd.read_csv(csv_path)
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.transform = transform

        self.prompt = "Findings: "
        self.prompt_ids = tokenizer(self.prompt, add_special_tokens=False).input_ids

        required_cols = ['image_path', 'report', 'clinical_labels', 'complexity']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info("Parsing clinical labels...")
        self.data['clinical_labels_parsed'] = self.data['clinical_labels'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        self.active_indices = list(range(len(self.data)))

        logger.info(f"Loaded {len(self.data)} samples from {csv_path}")
        logger.info(f"Complexity range: {self.data['complexity'].min():.2f} - {self.data['complexity'].max():.2f}")

    def _clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = str(text).strip()
        text = re.sub(r'(?i)^(findings|impression|indication)[:\s]*', '', text)
        text = re.sub(r'\d{2}[/-]\d{2}[/-]\d{2,4}', '[DATE]', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text if len(text.split()) >= 5 else "No significant findings."

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        actual_idx = self.active_indices[idx]
        row = self.data.iloc[actual_idx]

        img_path = os.path.join(self.data_root, row['image_path'])

        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            image = torch.zeros(3, 224, 224)

        text = self._clean_text(row['report'])
        full_text = self.prompt + text + self.tokenizer.eos_token

        encoding = self.tokenizer(
            full_text,
            max_length=cfg.MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()

        labels = input_ids.clone()
        labels[:len(self.prompt_ids)] = -100
        labels[attention_mask == 0] = -100

        clinical_labels = torch.tensor(row['clinical_labels_parsed'], dtype=torch.float32)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "clinical_labels": clinical_labels,
            "raw_text": text
        }

# ==============================================================================
# ARCHITECTURE
# ==============================================================================

class EfficientVisionEncoder(nn.Module):
    """ResNet50 with optional fine-tuning of last layers"""
    def __init__(self, pretrained_path=None, unfreeze_last_n_blocks=2):
        super().__init__()

        backbone = models.resnet50(weights=None)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"Loading BioViL weights from {pretrained_path}")
            state = torch.load(pretrained_path, map_location='cpu')

            new_state = {}
            for k, v in state.items():
                k = k.replace("module.", "").replace("image_encoder.", "").replace("encoder.encoder.", "").replace("encoder.", "")
                if "fc" not in k:
                    new_state[k] = v

            missing, unexpected = self.load_state_dict(new_state, strict=False)
            logger.info(f"Vision Weights Loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
            if len(missing) > 50:
                logger.warning(f"HIGH NUMBER OF MISSING VISION KEYS ({len(missing)})!")
                logger.warning(f"First 5 missing: {list(missing)[:5]}")
        else:
            logger.info("Using ImageNet initialization")
            backbone_pretrained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.load_state_dict(backbone_pretrained.state_dict(), strict=False)

        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False

        if unfreeze_last_n_blocks >= 1:
            for param in self.layer4.parameters():
                param.requires_grad = True
        if unfreeze_last_n_blocks >= 2:
            for param in self.layer3.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Vision Encoder: {trainable/1e6:.2f}M/{total/1e6:.2f}M trainable params")

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 2048, 7, 7]
        return x


class QFormerBottleneck(nn.Module):
    """Efficient Q-Former style bottleneck"""
    def __init__(self, visual_dim=2048, hidden_dim=2048, num_queries=32):
        super().__init__()

        self.num_queries = num_queries

        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.visual_proj = nn.Linear(visual_dim, hidden_dim)

        self.scale = hidden_dim ** -0.5

    def forward(self, visual_features):
        """
        Args:
            visual_features: [B, 2048, 7, 7]
        Returns:
            visual_summary: [B, num_queries, hidden_dim]
        """
        B, C, H, W = visual_features.shape

        visual_flat = visual_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        keys = values = self.visual_proj(visual_flat)              # [B, H*W, hidden_dim]

        queries = self.query_tokens.expand(B, -1, -1)             # [B, num_queries, hidden_dim]

        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        visual_summary = torch.matmul(attn_weights, values)       # [B, num_queries, hidden_dim]

        return visual_summary


class ReflexiveProjector(nn.Module):
    def __init__(self, hidden_dim=2048, num_queries=32):
        super().__init__()
        self.num_queries = num_queries
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256 * num_queries)
        )
        self.norm = nn.LayerNorm(256)
        self.out_proj = nn.Linear(256, hidden_dim)

    def forward(self, text_repr):
        B = text_repr.size(0)
        projected = self.projector(text_repr)
        recon_queries = projected.view(B, self.num_queries, 256)
        return self.out_proj(self.norm(recon_queries))


class ClinicalHead(nn.Module):
    def __init__(self, hidden_dim=2048, num_classes=14):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, visual_pooled):
        return self.classifier(visual_pooled)


class OptimizedRRA(nn.Module):
    """Optimized Reflexive Radio Adapter"""
    def __init__(self, llm, vision_encoder, config):
        super().__init__()

        self.llm = llm
        self.vision_encoder = vision_encoder
        self.config = config

        self.qformer = QFormerBottleneck(
            visual_dim=2048,
            hidden_dim=config.HIDDEN_DIM,
            num_queries=config.NUM_QUERIES
        )

        self.reflex_proj = ReflexiveProjector(
            hidden_dim=config.HIDDEN_DIM,
            num_queries=config.NUM_QUERIES
        )

        self.clinical_head = ClinicalHead(
            hidden_dim=config.HIDDEN_DIM,
            num_classes=config.NUM_CLINICAL_CLASSES
        )

    def forward(self, images, input_ids, attention_mask, labels=None, clinical_labels=None):
        B = images.size(0)

        # 1. Vision encoding
        visual_features = self.vision_encoder(images)       # [B, 2048, 7, 7]

        # 2. Q-Former
        visual_summary_clean = self.qformer(visual_features)      # [B, num_queries, hidden_dim]
        visual_summary = F.dropout(visual_summary_clean, p=0.1, training=self.training)

        # 3. Text embeddings
        text_emb = self.llm.get_input_embeddings()(input_ids)  # [B, seq_len, hidden_dim]

        # 4. Concatenate visual and text
        combined_emb = torch.cat([visual_summary, text_emb], dim=1)

        # 5. Extend attention mask
        visual_mask = torch.ones(B, cfg.NUM_QUERIES, device=attention_mask.device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([visual_mask, attention_mask], dim=1)

        # 6. Extend labels
        if labels is not None:
            visual_labels = torch.full((B, cfg.NUM_QUERIES), -100, device=labels.device, dtype=labels.dtype)
            extended_labels = torch.cat([visual_labels, labels], dim=1)
        else:
            extended_labels = None

        # 7. LLM forward
        outputs = self.llm(
            inputs_embeds=combined_emb,
            attention_mask=extended_mask,
            labels=extended_labels,
            output_hidden_states=True
        )

        # 8. Mean pool text hidden states
        last_hidden = outputs.hidden_states[-1]
        text_hidden = last_hidden[:, cfg.NUM_QUERIES:, :]

        mask_expanded = attention_mask.unsqueeze(-1).float()
        mask_sum = mask_expanded.sum(1).clamp(min=1e-6)
        text_repr = (text_hidden * mask_expanded).sum(1) / mask_sum

        # 9. Reflexive reconstruction
        recon_queries = self.reflex_proj(text_repr)

        # 10. Clinical classification
        clinical_logits = self.clinical_head(visual_summary.mean(dim=1))

        return {
            'lm_loss': outputs.loss,
            'logits': outputs.logits,
            'original_queries': visual_summary_clean.detach(),
            'recon_queries': recon_queries,
            'clinical_logits': clinical_logits,
            'text_repr': text_repr,
            'visual_summary': visual_summary,
            'text_emb': text_emb
        }

    @torch.no_grad()
    def generate_report(self, image, tokenizer, max_new_tokens=40, temperature=0.7):
        """Generate report from image with temperature sampling"""
        self.eval()

        visual_features = self.vision_encoder(image)
        visual_summary = self.qformer(visual_features)

        prompt = "Findings: "
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(image.device)

        for _ in range(max_new_tokens):
            text_emb = self.llm.get_input_embeddings()(input_ids)
            combined_emb = torch.cat([visual_summary, text_emb], dim=1)

            visual_mask = torch.ones(1, cfg.NUM_QUERIES, device=image.device, dtype=torch.long)
            text_mask = torch.ones(1, input_ids.size(1), device=image.device, dtype=torch.long)
            extended_mask = torch.cat([visual_mask, text_mask], dim=1)

            outputs = self.llm(inputs_embeds=combined_emb, attention_mask=extended_mask)

            next_token_logits = outputs.logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================

def contrastive_reflexive_loss(recon_queries, original_queries):
    """Stable Cosine Loss (Range 0-2)"""
    # Normalize
    recon_norm = F.normalize(recon_queries, dim=-1, eps=1e-6)
    orig_norm = F.normalize(original_queries, dim=-1, eps=1e-6)

    # Cosine similarity per query token
    # Shape: [B, N]
    cos_sim = torch.sum(recon_norm * orig_norm, dim=-1)

    # Loss: 1.0 - similarity (Minimize distance)
    return (1.0 - cos_sim).mean()


def asymmetric_focal_loss(logits, targets, pos_weight, gamma_pos=1, gamma_neg=4):
    p = torch.sigmoid(logits)
    
    loss_pos = -(1 - p)**gamma_pos * targets * torch.log(p + 1e-8) * pos_weight.to(logits.device)
    loss_neg = -p**gamma_neg * (1 - targets) * torch.log(1 - p + 1e-8)
    
    return (loss_pos + loss_neg).mean()


def compute_total_loss(outputs, clinical_labels, config, pos_weight):
    lm_loss = outputs['lm_loss']
    
    reflex_loss = contrastive_reflexive_loss(
        outputs['recon_queries'],
        outputs['original_queries'],
    )
    
    clinical_loss = asymmetric_focal_loss(
        outputs['clinical_logits'],
        clinical_labels,
        pos_weight=pos_weight,
        gamma_pos=config.FOCAL_GAMMA_POS,
        gamma_neg=config.FOCAL_GAMMA_NEG
    )
    
    total_loss = (
        config.LM_LOSS_WEIGHT * lm_loss +
        config.REFLEX_LOSS_WEIGHT * reflex_loss +
        config.CLINICAL_LOSS_WEIGHT * clinical_loss
    )
    
    return total_loss, {
        'lm': lm_loss.item(),
        'reflex': reflex_loss.item(),
        'clinical': clinical_loss.item(),
        'total': total_loss.item()
    }

# ==============================================================================
# CHECKPOINT MANAGEMENT
# ==============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, step, metrics, is_best=False):
    """Save complete checkpoint"""

    checkpoint_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'metrics': metrics,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'config': {
            'batch_size': cfg.BATCH_SIZE,
            'learning_rate': cfg.LEARNING_RATE,
            'proj_lr': cfg.PROJ_LR,
            'vision_lr': cfg.VISION_LR,
            'max_seq_length': cfg.MAX_SEQ_LENGTH,
            'lora_r': cfg.LORA_R,
            'lora_alpha': cfg.LORA_ALPHA,
        }
    }

    lora_dir = os.path.join(checkpoint_dir, f"epoch_{epoch}_lora")
    model.llm.save_pretrained(lora_dir)

    component_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_components.pt")
    torch.save({
        'vision_encoder': model.vision_encoder.state_dict(),
        'qformer': model.qformer.state_dict(),
        'reflex_proj': model.reflex_proj.state_dict(),
        'clinical_head': model.clinical_head.state_dict(),
    }, component_path)

    state_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_state.pt")
    torch.save(checkpoint, state_path)

    latest_dir = os.path.join(checkpoint_dir, "latest")
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    os.makedirs(latest_dir)

    shutil.copy(component_path, os.path.join(latest_dir, "components.pt"))
    shutil.copy(state_path, os.path.join(latest_dir, "state.pt"))
    shutil.copytree(lora_dir, os.path.join(latest_dir, "lora"))

    if is_best:
        best_dir = os.path.join(checkpoint_dir, "best")
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(latest_dir, best_dir)
        logger.info(f"New best model saved (F1: {metrics.get('val_f1', 0):.4f})")

    logger.info(f"Checkpoint saved: epoch {epoch}, step {step}")

    return checkpoint_dir


def load_checkpoint(model, optimizer, scheduler, checkpoint_dir=None):
    """Load checkpoint with error handling"""

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints", "latest")

    if not os.path.exists(checkpoint_dir):
        logger.info("No checkpoint found. Starting from scratch.")
        return 0, 0, {}

    logger.info(f"Loading checkpoint from {checkpoint_dir}")

    try:
        state_path = os.path.join(checkpoint_dir, "state.pt")
        checkpoint = torch.load(state_path, map_location='cpu')

        component_path = os.path.join(checkpoint_dir, "components.pt")
        components = torch.load(component_path, map_location='cpu')

        model.vision_encoder.load_state_dict(components['vision_encoder'])
        model.qformer.load_state_dict(components['qformer'])
        model.reflex_proj.load_state_dict(components['reflex_proj'])
        model.clinical_head.load_state_dict(components['clinical_head'])

        lora_dir = os.path.join(checkpoint_dir, "lora")
        from peft import PeftModel
        base_model = model.llm.base_model
        model.llm = PeftModel.from_pretrained(base_model, lora_dir, is_trainable=True)

        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])

        epoch = checkpoint['epoch']
        step = checkpoint['step']
        metrics = checkpoint.get('metrics', {})

        logger.info(f"Resumed from epoch {epoch}, step {step}")

        return epoch, step, metrics

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        logger.info("Starting from scratch instead.")
        return 0, 0, {}


# ==============================================================================
# VALIDATION
# ==============================================================================

@torch.no_grad()
def validate(model, val_loader, tokenizer, device, pos_weight, metrics_tracker=None):
    """Run validation with metrics tracking"""
    model.eval()

    all_clinical_preds = []
    all_clinical_targets = []
    total_lm_loss = 0
    total_reflex_loss = 0
    total_clinical_loss = 0

    try:
        sample_batch = next(iter(val_loader))
        sample_img = sample_batch['image'][0:1].to(device)
        gt_text = sample_batch['raw_text'][0]

        gen_text = model.generate_report(
            sample_img, tokenizer,
            max_new_tokens=cfg.SANITY_GEN_TOKENS,
            temperature=cfg.GEN_TEMPERATURE
        )

        logger.info("\n" + "="*70)
        logger.info("SAMPLE GENERATION CHECK")
        logger.info("="*70)
        logger.info(f"GT : {gt_text[:150]}...")
        logger.info(f"GEN: {gen_text}")
        logger.info("="*70 + "\n")

        if metrics_tracker is not None:
            gen_ids = tokenizer(gen_text, return_tensors='pt').input_ids
            metrics_tracker.log_generation_quality(gen_ids, tokenizer)

    except Exception as e:
        logger.warning(f"Sample generation failed: {e}")

    amp_context = autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else contextlib.nullcontext()

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        clinical_labels = batch['clinical_labels'].to(device)

        with amp_context:
            outputs = model(images, input_ids, attention_mask, labels, clinical_labels)
            _, losses = compute_total_loss(outputs, clinical_labels, cfg, pos_weight)

            total_lm_loss += losses['lm']
            total_reflex_loss += losses['reflex']
            total_clinical_loss += losses['clinical']

            preds = (torch.sigmoid(outputs['clinical_logits']) > 0.5).float().cpu()
            all_clinical_preds.append(preds)
            all_clinical_targets.append(clinical_labels.cpu())

            if metrics_tracker is not None:
                probs = torch.sigmoid(outputs['clinical_logits'])
                metrics_tracker.log_clinical_predictions(probs, clinical_labels, CHEXPERT_LABELS)
                metrics_tracker.log_confidence_distribution(outputs['clinical_logits'])

    all_preds = torch.cat(all_clinical_preds, dim=0).numpy()
    all_targets = torch.cat(all_clinical_targets, dim=0).numpy()

    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )

    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    rare_class_f1s = {}
    for i, label_name in enumerate(CHEXPERT_LABELS):
        if label_name in RARE_CONDITIONS:
            rare_class_f1s[label_name] = per_class_f1[i]

    metrics = {
        'val_lm_loss': total_lm_loss / len(val_loader),
        'val_reflex_loss': total_reflex_loss / len(val_loader),
        'val_clinical_loss': total_clinical_loss / len(val_loader),
        'val_f1': f1,
        'val_precision': precision,
        'val_recall': recall,
    }
    metrics.update({f'val_{k}_f1': v for k, v in rare_class_f1s.items()})

    logger.info(f"Val Metrics: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")
    for k, v in rare_class_f1s.items():
        logger.info(f" {k} F1: {v:.4f}")

    return metrics

from torch.utils.data import WeightedRandomSampler

def create_balanced_sampler(dataset):
    logger.info("Creating Balanced Sampler...")
    labels = np.array(dataset.data['clinical_labels_parsed'].tolist())
    class_counts = np.sum(labels, axis=0) + 1 
    class_weights = 1.0 / class_counts
    
    sample_weights = []
    for label_row in labels:
        if np.sum(label_row) == 0:
            sample_weights.append(0.01) # Penalize "Normal" images
        else:
            active_indices = np.where(label_row == 1)[0]
            sample_weights.append(np.max(class_weights[active_indices]))
            
    return WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights), replacement=True)

def compute_pos_weight(dataset, device):
    logger.info("Computing per-class positive weights...")
    labels = np.array(dataset.data['clinical_labels_parsed'].tolist())
    num_samples = len(labels)
    pos_counts = np.sum(labels, axis=0).clip(min=1)
    neg_counts = num_samples - pos_counts
    pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32).to(device)
    for i, name in enumerate(CHEXPERT_LABELS):
        logger.info(f"   {name}: pos_weight={pos_weight[i].item():.2f} ({int(pos_counts[i])} positives)")
    return pos_weight

# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Reflexive Radio Adapter (RRA)")
    parser.add_argument("--data_source", default=cfg.DATA_SOURCE)
    parser.add_argument("--local_data", default=cfg.LOCAL_DATA)
    parser.add_argument("--output_dir", default=cfg.OUTPUT_DIR)
    parser.add_argument("--vision_checkpoint", default=cfg.VISION_CHECKPOINT)
    parser.add_argument("--model_id", default=cfg.MODEL_ID)
    parser.add_argument("--train_csv", default=None,
                        help="Optional train CSV path. Default: <data_root>/metadata/train.csv")
    parser.add_argument("--val_csv", default=None,
                        help="Optional val CSV path. Default: <data_root>/metadata/val.csv")
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--grad_accumulation", type=int, default=cfg.GRAD_ACCUMULATION)
    parser.add_argument("--learning_rate", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--proj_lr", type=float, default=cfg.PROJ_LR)
    parser.add_argument("--vision_lr", type=float, default=cfg.VISION_LR)
    parser.add_argument("--num_workers", type=int, default=cfg.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--val_interval_steps", type=int, default=cfg.VAL_INTERVAL_STEPS)
    parser.add_argument("--early_stopping_patience", type=int, default=cfg.EARLY_STOPPING_PATIENCE)
    args = parser.parse_args()

    cfg.DATA_SOURCE = args.data_source
    cfg.LOCAL_DATA = args.local_data
    cfg.OUTPUT_DIR = args.output_dir
    cfg.VISION_CHECKPOINT = args.vision_checkpoint
    cfg.MODEL_ID = args.model_id
    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.GRAD_ACCUMULATION = args.grad_accumulation
    cfg.LEARNING_RATE = args.learning_rate
    cfg.PROJ_LR = args.proj_lr
    cfg.VISION_LR = args.vision_lr
    cfg.NUM_WORKERS = args.num_workers
    cfg.SEED = args.seed
    cfg.VAL_INTERVAL_STEPS = args.val_interval_steps
    cfg.EARLY_STOPPING_PATIENCE = args.early_stopping_patience

    setup_logging(cfg.OUTPUT_DIR)
    seed_everything(cfg.SEED)

    logger.info("="*70)
    logger.info("RRA TRAINING")
    logger.info("="*70)
    logger.info("="*70)

    gc.collect()
    torch.cuda.empty_cache()

    data_path = setup_data_layer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_csv = args.train_csv or os.path.join(data_path, "metadata", "train.csv")
    val_csv = args.val_csv or os.path.join(data_path, "metadata", "val.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    train_dataset = MIMICDataset(train_csv, data_path, tokenizer, train_transform)
    val_dataset = MIMICDataset(val_csv, data_path, tokenizer, val_transform)
    val_dataset.active_indices = list(range(len(val_dataset.data)))

    metrics_tracker = StabilityMetricsTracker(cfg.OUTPUT_DIR)

    logger.info("Building model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    base_llm = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    base_llm.gradient_checkpointing_enable()
    base_llm = prepare_model_for_kbit_training(base_llm, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=cfg.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    base_llm = get_peft_model(base_llm, lora_config)
    base_llm.print_trainable_parameters()

    vision_encoder = EfficientVisionEncoder(
        pretrained_path=cfg.VISION_CHECKPOINT,
        unfreeze_last_n_blocks=4
    ).to(device)

    model = OptimizedRRA(base_llm, vision_encoder, cfg).to(device)
    pos_weight = compute_pos_weight(train_dataset, device)
    logger.info(f"Pos Weights (min/max):   {pos_weight.min().item():.2f} / {pos_weight.max().item():.2f}")

    logger.info("Model built successfully")

    optimizer_params = [
        {'params': model.llm.parameters(), 'lr': cfg.LEARNING_RATE},
        {'params': model.vision_encoder.layer3.parameters(), 'lr': cfg.VISION_LR},
        {'params': model.vision_encoder.layer4.parameters(), 'lr': cfg.VISION_LR},
        {'params': model.qformer.parameters(), 'lr': cfg.PROJ_LR},
        {'params': model.reflex_proj.parameters(), 'lr': cfg.PROJ_LR},
        {'params': model.clinical_head.parameters(), 'lr': cfg.PROJ_LR},
    ]

    if HAS_BNB:
        optimizer = bnb.optim.AdamW8bit(optimizer_params, weight_decay=cfg.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(optimizer_params, weight_decay=cfg.WEIGHT_DECAY)

    full_steps_per_epoch = len(train_dataset.data) // (cfg.BATCH_SIZE * cfg.GRAD_ACCUMULATION)
    total_steps_for_scheduler = full_steps_per_epoch * cfg.EPOCHS
    warmup_steps = int(total_steps_for_scheduler * cfg.WARMUP_RATIO)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps_for_scheduler
    )

    logger.info(f"Scheduler: {total_steps_for_scheduler} total steps, {warmup_steps} warmup steps")

    # Load checkpoint if exists (pass the placeholder scheduler; it gets overwritten)
    start_epoch, global_step, history = load_checkpoint(model, optimizer, scheduler)

    if not history:
        history = defaultdict(list)

    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    best_f1 = history.get('best_f1', 0.0)

    patience_counter = 0

    logger.info("\n" + "="*70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*70)
    logger.info(f"Device:                  {device}")
    logger.info(f"Batch Size:              {cfg.BATCH_SIZE}")
    logger.info(f"Gradient Accumulation:   {cfg.GRAD_ACCUMULATION}")
    logger.info(f"Effective Batch:         {cfg.BATCH_SIZE * cfg.GRAD_ACCUMULATION}")
    logger.info(f"Learning Rates:")
    logger.info(f"  - LLM:                 {cfg.LEARNING_RATE}")
    logger.info(f"  - Vision:              {cfg.VISION_LR}")
    logger.info(f"  - Projection:          {cfg.PROJ_LR}")
    logger.info(f"LoRA Config:             r={cfg.LORA_R}, alpha={cfg.LORA_ALPHA}")
    logger.info(f"Max Grad Norm:           {cfg.MAX_GRAD_NORM}")
    logger.info(f"Warmup Ratio:            {cfg.WARMUP_RATIO}")
    logger.info(f"Max Seq Length:          {cfg.MAX_SEQ_LENGTH}")
    logger.info(f"Epochs:                  {cfg.EPOCHS}")
    logger.info(f"Loss Weights:            LM={cfg.LM_LOSS_WEIGHT}, "
                f"Reflex={cfg.REFLEX_LOSS_WEIGHT}, Clinical={cfg.CLINICAL_LOSS_WEIGHT}")
    logger.info(f"Training Samples:        {len(train_dataset)}")
    logger.info(f"Validation Samples:      {len(val_dataset)}")
    logger.info(f"Starting from:           Epoch {start_epoch}, Step {global_step}")
    logger.info(f"AMP Dtype:               {'float16 (T4 compatible)' if device.type == 'cuda' else 'N/A (CPU)'}")
    logger.info(f"Generation Temp:         {cfg.GEN_TEMPERATURE}")
    logger.info("="*70 + "\n")

    amp_context = autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else contextlib.nullcontext()

    for epoch in range(start_epoch, cfg.EPOCHS):
        logger.info(f"\n{'='*70}")
        logger.info(f"EPOCH {epoch + 1}/{cfg.EPOCHS}")
        logger.info(f"{'='*70}")

        epoch_improved = False

        train_sampler = create_balanced_sampler(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            sampler=train_sampler,     
            shuffle=False,             
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )

        steps_per_epoch = len(train_loader) // cfg.GRAD_ACCUMULATION
        logger.info(f"Epoch {epoch+1}: {steps_per_epoch} optimizer steps")

        model.train()
        epoch_losses = defaultdict(float)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(pbar):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            clinical_labels = batch['clinical_labels'].to(device)

            with amp_context:
                outputs = model(images, input_ids, attention_mask, labels, clinical_labels)
                total_loss, losses = compute_total_loss(outputs, clinical_labels, cfg, pos_weight)
                total_loss = total_loss / cfg.GRAD_ACCUMULATION

            loss_fp32 = total_loss.detach().float()
            has_issue = metrics_tracker.log_nan_inf_events(loss_fp32)

            if has_issue:
                logger.warning(f"NaN/Inf detected at step {global_step}. Skipping step.")
                metrics_tracker.skipped_steps.append(global_step)
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            for k, v in losses.items():
                epoch_losses[k] += v

            metrics_tracker.log_loss_components(
                losses['lm'], losses['reflex'], losses['clinical'], losses['total']
            )

            if (step + 1) % cfg.GRAD_ACCUMULATION == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                grad_norm, max_grad, was_clipped = metrics_tracker.log_gradient_stats(
                    model, cfg.MAX_GRAD_NORM
                )

                pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad and p.grad is not None),
                    cfg.MAX_GRAD_NORM
                )

                post_clip_norm = min(pre_clip_norm.item(), cfg.MAX_GRAD_NORM)
                metrics_tracker.log_post_clip_norm(post_clip_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1

                if 'visual_summary' in outputs and 'text_emb' in outputs:
                    metrics_tracker.log_multimodal_utilization(
                        outputs['visual_summary'], outputs['text_emb']
                    )

                pbar.set_postfix({
                    'loss': f"{losses['total']:.4f}",
                    'lm': f"{losses['lm']:.3f}",
                    'rfx': f"{losses['reflex']:.3f}",
                    'clin': f"{losses['clinical']:.3f}",
                    'gn': f"{grad_norm:.3f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })

                if global_step % cfg.VAL_INTERVAL_STEPS == 0:
                    logger.info(f"\nRunning mid-epoch validation at step {global_step}...")

                    val_metrics = validate(model, val_loader, tokenizer, device, pos_weight, metrics_tracker)

                    for k, v in val_metrics.items():
                        history[k].append(v)

                    current_f1 = val_metrics['val_f1']
                    is_best = current_f1 > best_f1

                    if is_best:
                        best_f1 = current_f1
                        epoch_improved = True  # Mark that this epoch improved
                        history['best_f1'] = best_f1

                    # Always save mid-epoch checkpoint; never touch patience here
                    save_checkpoint(
                        model, optimizer, scheduler,
                        epoch + 1, global_step, val_metrics, is_best
                    )

                    model.train()

        # End of epoch
        metrics_tracker.print_epoch_summary(epoch + 1)

        avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}

        logger.info(f"\nEpoch {epoch+1} Training Summary:")
        logger.info(f"   Total Loss:     {avg_losses['total']:.4f}")
        logger.info(f"   LM Loss:        {avg_losses['lm']:.4f}")
        logger.info(f"   Reflex Loss:    {avg_losses['reflex']:.4f}")
        logger.info(f"   Clinical Loss:  {avg_losses['clinical']:.4f}")

        for k, v in avg_losses.items():
            history[f'train_{k}'].append(v)

        logger.info("\nRunning end-of-epoch validation...")
        val_metrics = validate(model, val_loader, tokenizer, device, pos_weight, metrics_tracker)

        for k, v in val_metrics.items():
            history[k].append(v)

        current_f1 = val_metrics['val_f1']
        is_best = current_f1 > best_f1

        if is_best or epoch_improved:
            best_f1 = max(best_f1, current_f1)
            patience_counter = 0
            history['best_f1'] = best_f1
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{cfg.EARLY_STOPPING_PATIENCE}")

        save_checkpoint(
            model, optimizer, scheduler,
            epoch + 1, global_step, val_metrics, is_best
        )

        plot_training_progress(history)
        metrics_tracker.save_all_metrics()

        # Early stopping check
        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            logger.info(f"\nEarly stopping: No improvement for {patience_counter} consecutive epochs")
            break

        gc.collect()
        torch.cuda.empty_cache()

    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Best F1 Score:       {best_f1:.4f}")
    logger.info(f"Total Epochs:        {epoch + 1}")
    logger.info(f"Total Steps:         {global_step}")
    logger.info(f"Checkpoints saved:   {cfg.OUTPUT_DIR}/checkpoints")
    logger.info(f"Metrics saved:       {cfg.OUTPUT_DIR}/metrics")
    logger.info("="*70)

    metrics_tracker.save_all_metrics()


if __name__ == "__main__":
    main()
