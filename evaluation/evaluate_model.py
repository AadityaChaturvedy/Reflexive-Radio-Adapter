"""
RRA Model Evaluation Script
Metrics: F1, METEOR, CheXBERT, BLEU-4, ROUGE-L, ROUGE-1,
         AUC-ROC, Sensitivity, Specificity, Reflexive Score, BERTScore
"""

import os
import sys
import ast
import json
import logging
import argparse
import contextlib

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import bert_score

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, PeftModel

# ==============================================================================
# LOGGING — dual handler: console + persistent file
# Output dir created early so the log file has a home from the first line
# ==============================================================================

logger = logging.getLogger(__name__)


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(output_dir, "evaluation_log.txt"), mode='a'
            ),
        ],
        force=True,
    )

# ==============================================================================
# CONFIG — MUST MATCH TRAINING EXACTLY
# ==============================================================================
CHEXPERT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]
RARE_CONDITIONS = {'Pneumothorax', 'Fracture', 'Lung Lesion'}

MODEL_ID        = "google/gemma-2b"
VISION_CKPT     = os.environ.get("RRA_VISION_CKPT", "")
CHECKPOINT_DIR  = os.environ.get("RRA_CHECKPOINT_DIR", "outputs/checkpoints/best")
DATA_ROOT       = os.environ.get("RRA_DATA_ROOT", "data")
VAL_CSV         = os.environ.get("RRA_VAL_CSV", "data/metadata/test.csv")
OUTPUT_DIR      = os.environ.get("RRA_EVAL_OUTPUT_DIR", "outputs/evaluation")

NUM_QUERIES          = 32
HIDDEN_DIM           = 2048
NUM_CLINICAL_CLASSES = 14
MAX_SEQ_LENGTH       = 320

BATCH_SIZE           = 8

MAX_GEN_TOKENS       = 100


# ==============================================================================
# ARCHITECTURE
# ==============================================================================
class EfficientVisionEncoder(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        backbone    = models.resnet50(weights=None)
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if pretrained_path and os.path.exists(pretrained_path):
            state     = torch.load(pretrained_path, map_location='cpu')
            new_state = {}
            for k, v in state.items():
                k = (k.replace("module.", "")
                      .replace("image_encoder.", "")
                      .replace("encoder.encoder.", "")
                      .replace("encoder.", ""))
                if "fc" not in k:
                    new_state[k] = v
            self.load_state_dict(new_state, strict=False)
            logger.info("BioViL weights loaded")

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class QFormerBottleneck(nn.Module):
    def __init__(self, visual_dim=2048, hidden_dim=2048, num_queries=32):
        super().__init__()
        self.num_queries  = num_queries
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.visual_proj  = nn.Linear(visual_dim, hidden_dim)
        self.scale        = hidden_dim ** -0.5

    def forward(self, visual_features):
        B, C, H, W   = visual_features.shape
        visual_flat  = visual_features.flatten(2).transpose(1, 2)
        keys = values = self.visual_proj(visual_flat)
        queries       = self.query_tokens.expand(B, -1, -1)
        attn_weights  = F.softmax(
            torch.matmul(queries, keys.transpose(-2, -1)) * self.scale, dim=-1)
        return torch.matmul(attn_weights, values)


class ReflexiveProjector(nn.Module):
    def __init__(self, hidden_dim=2048, num_queries=32):
        super().__init__()
        self.num_queries = num_queries
        self.projector   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256 * num_queries)
        )
        self.norm     = nn.LayerNorm(256)
        self.out_proj = nn.Linear(256, hidden_dim)

    def forward(self, text_repr):
        B             = text_repr.size(0)
        projected     = self.projector(text_repr)
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
    def __init__(self, llm, vision_encoder):
        super().__init__()
        self.llm            = llm
        self.vision_encoder = vision_encoder
        self.qformer        = QFormerBottleneck(2048, HIDDEN_DIM, NUM_QUERIES)
        self.reflex_proj    = ReflexiveProjector(HIDDEN_DIM, NUM_QUERIES)
        self.clinical_head  = ClinicalHead(HIDDEN_DIM, NUM_CLINICAL_CLASSES)

    def forward(self, images, input_ids, attention_mask, labels=None):
        B = images.size(0)

        visual_features      = self.vision_encoder(images)
        visual_summary_clean = self.qformer(visual_features)

        # training=self.training ensures dropout is OFF at eval
        visual_summary = F.dropout(visual_summary_clean, p=0.1,
                                   training=self.training)

        text_emb     = self.llm.get_input_embeddings()(input_ids)
        combined_emb = torch.cat([visual_summary, text_emb], dim=1)

        visual_mask   = torch.ones(B, NUM_QUERIES,
                                   device=attention_mask.device,
                                   dtype=attention_mask.dtype)
        extended_mask = torch.cat([visual_mask, attention_mask], dim=1)

        if labels is not None:
            visual_labels   = torch.full((B, NUM_QUERIES), -100,
                                         device=labels.device, dtype=labels.dtype)
            extended_labels = torch.cat([visual_labels, labels], dim=1)
        else:
            extended_labels = None

        outputs = self.llm(
            inputs_embeds=combined_emb,
            attention_mask=extended_mask,
            labels=extended_labels,
            output_hidden_states=True
        )

        last_hidden = outputs.hidden_states[-1]
        text_hidden = last_hidden[:, NUM_QUERIES:, :]
        mask_exp    = attention_mask.unsqueeze(-1).float()
        text_repr   = (text_hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-6)

        recon_queries   = self.reflex_proj(text_repr)
        clinical_logits = self.clinical_head(visual_summary.mean(dim=1))

        return {
            'lm_loss':          outputs.loss,
            'original_queries': visual_summary_clean.detach(),
            'recon_queries':    recon_queries,
            'clinical_logits':  clinical_logits,
            'text_repr':        text_repr,
            'visual_summary':   visual_summary,
        }

    @torch.no_grad()
    def generate_report(self, image, tokenizer, max_new_tokens=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
        """
        Temperature sampling with nucleus filtering and repetition penalty.
        Args:
            temperature: Controls randomness (0.7 = balanced)
            top_p: Nucleus sampling threshold (0.9 = keep top 90% probability mass)
            repetition_penalty: Penalty for repeated tokens (1.2 = moderate penalty)
        """
        self.eval()
        visual_features = self.vision_encoder(image)
        visual_summary  = self.qformer(visual_features)

        prompt    = "Findings: "
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(image.device)
        
        generated_tokens = []

        for _ in range(max_new_tokens):
            text_emb     = self.llm.get_input_embeddings()(input_ids)
            combined_emb = torch.cat([visual_summary, text_emb], dim=1)
            
            # Create masks
            visual_mask  = torch.ones(1, NUM_QUERIES, device=image.device, dtype=torch.long)
            text_mask    = torch.ones(1, input_ids.size(1), device=image.device, dtype=torch.long)
            ext_mask     = torch.cat([visual_mask, text_mask], dim=1)

            out = self.llm(inputs_embeds=combined_emb, attention_mask=ext_mask)
            logits = out.logits[0, -1, :]

            # Apply repetition penalty to generated tokens only (not prompt)
            for token_id in set(generated_tokens):
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty
            
            # Temperature scaling
            logits = logits / temperature
            
            # Nucleus (top-p) sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens outside nucleus
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
            
            # Sample from filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_tokens.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break

        raw = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Clean the output
        if raw.lower().startswith("findings:"):
            raw = raw[len("findings:"):].strip()

        return raw


# ==============================================================================
# DATASET
# ==============================================================================
class EvalDataset(Dataset):
    def __init__(self, csv_path, data_root, tokenizer, transform):
        self.data       = pd.read_csv(csv_path)
        self.data_root  = data_root
        self.tokenizer  = tokenizer
        self.transform  = transform
        self.prompt     = "Findings: "
        self.prompt_ids = tokenizer(self.prompt, add_special_tokens=False).input_ids

        self.data['clinical_labels_parsed'] = self.data['clinical_labels'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        logger.info(f"Loaded {len(self.data)} evaluation samples")

    def _clean_text(self, text):
        import re
        if not isinstance(text, str):
            return ""
        text = re.sub(r'(?i)^(findings|impression|indication)[:\s]*',
                      '', str(text).strip())
        text = re.sub(r'\d{2}[/-]\d{2}[/-]\d{2,4}', '[DATE]', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text if len(text.split()) >= 5 else "No significant findings."

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row      = self.data.iloc[idx]
        img_path = os.path.join(self.data_root, row['image_path'])

        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 224, 224)

        text      = self._clean_text(row['report'])
        full_text = self.prompt + text + self.tokenizer.eos_token
        enc       = self.tokenizer(
            full_text, max_length=MAX_SEQ_LENGTH,
            padding="max_length", truncation=True, return_tensors="pt"
        )

        input_ids      = enc.input_ids.squeeze()
        attention_mask = enc.attention_mask.squeeze()
        labels         = input_ids.clone()
        labels[:len(self.prompt_ids)] = -100
        labels[attention_mask == 0]   = -100

        return {
            "image":           image,
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "labels":          labels,
            "clinical_labels": torch.tensor(row['clinical_labels_parsed'],
                                            dtype=torch.float32),
            "raw_text":        text,
        }


# ==============================================================================
# MODEL LOADER
# ==============================================================================
def load_model(checkpoint_dir, device, vision_ckpt):
    logger.info("Loading model from checkpoint...")

    for item in ['components.pt', 'state.pt', 'lora']:
        path = os.path.join(checkpoint_dir, item)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                "Verify checkpoint directory is complete (components.pt, state.pt, lora/)."
            )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    base_llm = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config,
        device_map="auto", low_cpu_mem_usage=True
    )
    base_llm = prepare_model_for_kbit_training(
        base_llm, use_gradient_checkpointing=False
    )

    lora_dir = os.path.join(checkpoint_dir, "lora")
    llm      = PeftModel.from_pretrained(base_llm, lora_dir, is_trainable=False)

    if not vision_ckpt:
        raise ValueError("vision_ckpt must be provided via --vision_ckpt or RRA_VISION_CKPT")

    if not os.path.exists(vision_ckpt):
        raise FileNotFoundError(f"Vision checkpoint not found: {vision_ckpt}")

    vision_encoder = EfficientVisionEncoder(pretrained_path=vision_ckpt).to(device)
    model          = OptimizedRRA(llm, vision_encoder).to(device)

    components = torch.load(
        os.path.join(checkpoint_dir, "components.pt"), map_location='cpu'
    )
    model.vision_encoder.load_state_dict(components['vision_encoder'])
    model.qformer.load_state_dict(components['qformer'])
    model.reflex_proj.load_state_dict(components['reflex_proj'])
    model.clinical_head.load_state_dict(components['clinical_head'])

    model.eval()
    logger.info("Model loaded and set to eval mode")
    return model


# ==============================================================================
# THRESHOLD OPTIMIZATION
# ==============================================================================
def optimize_thresholds(all_probs, all_targets):
    logger.info("Optimizing per-class thresholds on val set...")
    candidates  = np.arange(0.05, 0.95, 0.05)
    best_thresholds = []

    for i, label in enumerate(CHEXPERT_LABELS):
        best_t, best_f1 = 0.5, 0.0
        for t in candidates:
            preds = (all_probs[:, i] > t).astype(float)
            f1    = f1_score(all_targets[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds.append(best_t)
        logger.info(f"   {label:<35} thresh={best_t:.2f}  F1={best_f1:.4f}")

    return np.array(best_thresholds)


# ==============================================================================
# REFLEXIVE SCORE
# ==============================================================================
def compute_reflexive_score(recon_list, original_list):
    """
    Mean cosine similarity between text-reconstructed queries and original
    visual queries across the full val set. Range [-1, 1].
    This is your novel metric — report it in the paper.
    """
    recon    = torch.cat(recon_list,    dim=0).float()
    original = torch.cat(original_list, dim=0).float()
    cos_sim  = (F.normalize(recon, dim=-1) * F.normalize(original, dim=-1)).sum(dim=-1)
    return cos_sim.mean().item(), cos_sim.std().item()


# ==============================================================================
# CHEXBERT
# ==============================================================================
def compute_chexbert_score(generated_texts, reference_texts):
    try:
        from chexbert_eval import compute_scores
        return compute_scores(generated_texts, reference_texts)
    except ImportError:
        logger.warning(
            "WARNING: chexbert_eval not installed — using Bio_ClinicalBERT cosine proxy.\n"
            "   Official install: pip install git+https://github.com/stanfordmlgroup/CheXBert.git"
        )

    try:
        from transformers import AutoTokenizer as AT, AutoModel as AM
        tok = AT.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        mod = AM.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        mod.eval()

        def embed(texts):
            enc = tok(texts, return_tensors='pt', padding=True,
                      truncation=True, max_length=512)
            with torch.no_grad():
                return mod(**enc).last_hidden_state[:, 0, :]

        CHUNK    = 32
        gen_embs = [embed(generated_texts[i:i+CHUNK])
                    for i in range(0, len(generated_texts), CHUNK)]
        ref_embs = [embed(reference_texts[i:i+CHUNK])
                    for i in range(0, len(reference_texts), CHUNK)]

        gen_emb = torch.cat(gen_embs)
        ref_emb = torch.cat(ref_embs)
        cos     = (F.normalize(gen_emb, dim=-1) *
                   F.normalize(ref_emb, dim=-1)).sum(dim=-1)
        return {"chexbert_proxy_cosine": round(cos.mean().item(), 4)}

    except Exception as e:
        logger.error(f"ClinicalBERT proxy failed: {e}")
        return {"chexbert_proxy_cosine": None}


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================
@torch.no_grad()
def evaluate(model, dataloader, tokenizer, device, gen_samples, threshold_file=None):
    model.eval()
    amp_ctx = (autocast('cuda', dtype=torch.float16)
               if device.type == 'cuda' else contextlib.nullcontext())

    all_probs            = []
    all_targets          = []
    all_recon_queries    = []
    all_original_queries = []
    reference_texts      = []

    # ── Pass 1: Classification + Reflexive (full val set, batch=16) ──────────
    logger.info("Pass 1/2 — Classification & reflexive...")
    for batch in tqdm(dataloader, desc="Classification pass"):
        images          = batch['image'].to(device)
        input_ids       = batch['input_ids'].to(device)
        attention_mask  = batch['attention_mask'].to(device)
        labels          = batch['labels'].to(device)
        clinical_labels = batch['clinical_labels'].to(device)

        with amp_ctx:
            outputs = model(images, input_ids, attention_mask, labels)

        all_probs.append(torch.sigmoid(outputs['clinical_logits']).cpu().float().numpy())
        all_targets.append(clinical_labels.cpu().numpy())
        all_recon_queries.append(outputs['recon_queries'].cpu().float())
        all_original_queries.append(outputs['original_queries'].cpu().float())
        reference_texts.extend(batch['raw_text'])

    # Check if we should load thresholds from a file to avoid test set leakage
    if threshold_file and os.path.exists(threshold_file):
        logger.info(f"Loading thresholds from {threshold_file}...")
        with open(threshold_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "classification" in data and "thresholds_used" in data["classification"]:
                thresholds = np.array(data["classification"]["thresholds_used"])
            elif "thresholds_used" in data:
                thresholds = np.array(data["thresholds_used"])
            elif "Tuned_Thresholds" in data:
                thresholds = np.array([data["Tuned_Thresholds"].get(c, 0.5) for c in CHEXPERT_LABELS])
            else:
                raise ValueError("Could not find thresholds_used key in threshold JSON dict")
        elif isinstance(data, list):
            thresholds = np.array(data)
        else:
            raise ValueError(f"Invalid threshold format in {threshold_file}")
        logger.info(f"Using loaded thresholds: {thresholds.tolist()}")
    else:
        logger.info("Optimizing thresholds on the current dataset...")
        thresholds = optimize_thresholds(all_probs, all_targets)

    all_preds  = (all_probs > thresholds).astype(float)

    # ── Pass 2: Greedy generation ────────────────────────
    gen_subset          = min(gen_samples, len(dataloader.dataset))
    reference_texts_gen = reference_texts[:gen_subset]
    generated_texts     = []

    logger.info(f"Pass 2/2 — Greedy report generation ({gen_subset} samples)...")
    for i in tqdm(range(gen_subset), desc="Greedy generation"):
        sample = dataloader.dataset[i]
        img    = sample['image'].unsqueeze(0).to(device)
        gen    = model.generate_report(img, tokenizer,
                                       max_new_tokens=MAX_GEN_TOKENS)
        generated_texts.append(gen)

    # ── Classification Metrics ────────────────────────────────────────────────
    logger.info("Computing classification metrics...")

    macro_f1               = f1_score(all_targets, all_preds,
                                      average='macro', zero_division=0)
    macro_p, macro_r, _, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0)
    per_class_f1           = f1_score(all_targets, all_preds,
                                      average=None, zero_division=0)

    auc_scores = {}
    valid_aucs = []
    for i, label in enumerate(CHEXPERT_LABELS):
        pos = all_targets[:, i].sum()
        if 0 < pos < len(all_targets):
            auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
            auc_scores[label] = auc
            valid_aucs.append(auc)
        else:
            auc_scores[label] = None
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    sensitivities = {}
    specificities = {}
    for i, label in enumerate(CHEXPERT_LABELS):
        tp = ((all_preds[:, i] == 1) & (all_targets[:, i] == 1)).sum()
        fn = ((all_preds[:, i] == 0) & (all_targets[:, i] == 1)).sum()
        tn = ((all_preds[:, i] == 0) & (all_targets[:, i] == 0)).sum()
        fp = ((all_preds[:, i] == 1) & (all_targets[:, i] == 0)).sum()
        sensitivities[label] = float(tp / (tp + fn + 1e-8))
        specificities[label] = float(tn / (tn + fp + 1e-8))

    macro_sensitivity = float(np.mean(list(sensitivities.values())))
    macro_specificity = float(np.mean(list(specificities.values())))

    # ── Reflexive Score ───────────────────────────────────────────────────────
    logger.info("Computing reflexive score...")
    reflex_mean, reflex_std = compute_reflexive_score(
        all_recon_queries, all_original_queries
    )

    # ── Generation Metrics ────────────────────────────────────────────────────
    logger.info("Computing generation metrics...")

    import nltk
    for resource in ['wordnet', 'omw-1.4']:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource, quiet=True)

    smooth   = SmoothingFunction().method1
    refs_tok = [[ref.lower().split()] for ref in reference_texts_gen]
    hyps_tok = [gen.lower().split()  for gen in generated_texts]
    bleu4    = corpus_bleu(refs_tok, hyps_tok,
                           weights=(0.25, 0.25, 0.25, 0.25),
                           smoothing_function=smooth)
    bleu1    = corpus_bleu(refs_tok, hyps_tok,
                           weights=(1, 0, 0, 0),
                           smoothing_function=smooth)

    rouge_sc = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    r1_scores, rL_scores = [], []
    for gen, ref in zip(generated_texts, reference_texts_gen):
        s = rouge_sc.score(ref, gen)
        r1_scores.append(s['rouge1'].fmeasure)
        rL_scores.append(s['rougeL'].fmeasure)
    rouge1 = float(np.mean(r1_scores))
    rougeL = float(np.mean(rL_scores))

    meteor_scores = [
        meteor_score([ref.lower().split()], gen.lower().split())
        for gen, ref in zip(generated_texts, reference_texts_gen)
    ]
    meteor = float(np.mean(meteor_scores))

    # roberta-large: ~500MB, always cached, standard in medical NLP papers
    # deberta-xlarge-mnli (~3GB) stalls on download in restricted GPU environments
    logger.info("Computing BERTScore (roberta-large)...")
    try:
        P_bert, R_bert, F_bert = bert_score.score(
            generated_texts, reference_texts_gen,
            model_type="roberta-large",
            lang="en", verbose=False, device=str(device),
            rescale_with_baseline=True
        )
        bertscore_f1 = float(F_bert.mean())
        bertscore_p  = float(P_bert.mean())
        bertscore_r  = float(R_bert.mean())
        logger.info(f"   BERTScore F1 (roberta-large, rescaled): {bertscore_f1:.4f}")
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}. Setting to None.")
        bertscore_f1 = bertscore_p = bertscore_r = None

    logger.info("Computing CheXBERT score...")
    chexbert = compute_chexbert_score(generated_texts, reference_texts_gen)

    # ── Compile ───────────────────────────────────────────────────────────────
    results = {
        "classification": {
            "macro_F1":          round(float(macro_f1),          4),
            "macro_precision":   round(float(macro_p),           4),
            "macro_recall":      round(float(macro_r),           4),
            "macro_AUC_ROC":     round(macro_auc,                4),
            "macro_sensitivity": round(macro_sensitivity,        4),
            "macro_specificity": round(macro_specificity,        4),
            "thresholds_used":   thresholds.tolist(),
        },
        "per_class": {
            label: {
                "F1":          round(float(per_class_f1[i]),   4),
                "AUC_ROC":     round(auc_scores[label], 4) if auc_scores[label] is not None else None,
                "sensitivity": round(sensitivities[label],     4),
                "specificity": round(specificities[label],     4),
                "threshold":   round(float(thresholds[i]),     2),
            }
            for i, label in enumerate(CHEXPERT_LABELS)
        },
        "generation": {
            "BLEU_1":                round(bleu1,        4),
            "BLEU_4":                round(bleu4,        4),
            "ROUGE_1":               round(rouge1,       4),
            "ROUGE_L":               round(rougeL,       4),
            "METEOR":                round(meteor,       4),
            "BERTScore_P":           round(bertscore_p,  4) if bertscore_p  is not None else None,
            "BERTScore_R":           round(bertscore_r,  4) if bertscore_r  is not None else None,
            "BERTScore_F1":          round(bertscore_f1, 4) if bertscore_f1 is not None else None,
            "BERTScore_model":       "roberta-large (rescaled baseline)",
            "CheXBERT":              chexbert,
            "num_reports_evaluated": gen_subset,
            "decoding_strategy":     "nucleus sampling (top-p=0.9, temp=0.7, rep_penalty=1.2)",
            "max_new_tokens":        MAX_GEN_TOKENS,
        },
        "reflexive": {
            "reflexive_score_mean": round(reflex_mean, 4),
            "reflexive_score_std":  round(reflex_std,  4),
            "interpretation": (
                "Mean cosine similarity between text-reconstructed queries "
                "and original visual queries. Range [-1, 1]. "
                + ('Strong'   if reflex_mean > 0.7
                   else 'Moderate' if reflex_mean > 0.4
                   else 'Weak')
                + " visual-text grounding."
            ),
        },
        "rare_classes": {
            label: {
                "F1":          round(float(per_class_f1[i]), 4),
                "sensitivity": round(sensitivities[label],   4),
                "specificity": round(specificities[label],   4),
                "AUC_ROC":     round(auc_scores[label], 4) if auc_scores[label] is not None else None,
            }
            for i, label in enumerate(CHEXPERT_LABELS)
            if label in RARE_CONDITIONS
        },
    }

    return results, generated_texts, reference_texts_gen


# ==============================================================================
# PRINT
# ==============================================================================
def print_results(results):
    # All output goes through logger — captured in both console and evaluation_log.txt
    logger.info("=" * 70)
    logger.info("RRA EVALUATION RESULTS")
    logger.info("=" * 70)

    c = results['classification']
    logger.info("── CLASSIFICATION ──────────────────────────────────────────────────")
    logger.info(f"  Macro F1:          {c['macro_F1']}")
    logger.info(f"  Macro Precision:   {c['macro_precision']}")
    logger.info(f"  Macro Recall:      {c['macro_recall']}")
    logger.info(f"  Macro AUC-ROC:     {c['macro_AUC_ROC']}")
    logger.info(f"  Macro Sensitivity: {c['macro_sensitivity']}")
    logger.info(f"  Macro Specificity: {c['macro_specificity']}")

    logger.info("── RARE CLASSES ────────────────────────────────────────────────────")
    for label, m in results['rare_classes'].items():
        logger.info(f"  {label}:")
        logger.info(f"    F1={m['F1']}  Sens={m['sensitivity']}  "
                    f"Spec={m['specificity']}  AUC={m['AUC_ROC']}")

    logger.info("── PER-CLASS BREAKDOWN ─────────────────────────────────────────────")
    for label, m in results['per_class'].items():
        auc_str = f"{m['AUC_ROC']:.4f}" if m['AUC_ROC'] is not None else "N/A  "
        logger.info(f"  {label:<35} F1={m['F1']:.4f}  AUC={auc_str}  "
                    f"thresh={m['threshold']}")

    g = results['generation']
    logger.info("── GENERATION ──────────────────────────────────────────────────────")
    logger.info(f"  Decoding:          {g['decoding_strategy']}")
    logger.info(f"  Max tokens:        {g['max_new_tokens']}")
    logger.info(f"  Reports evaluated: {g['num_reports_evaluated']}")
    logger.info(f"  BLEU-1:            {g['BLEU_1']}")
    logger.info(f"  BLEU-4:            {g['BLEU_4']}")
    logger.info(f"  ROUGE-1:           {g['ROUGE_1']}")
    logger.info(f"  ROUGE-L:           {g['ROUGE_L']}")
    logger.info(f"  METEOR:            {g['METEOR']}")
    logger.info(f"  BERTScore P/R/F1:  "
                f"{g['BERTScore_P']} / {g['BERTScore_R']} / {g['BERTScore_F1']}")
    logger.info(f"  CheXBERT:          {g['CheXBERT']}")

    r = results['reflexive']
    logger.info("── REFLEXIVE SCORE ─────────────────────────────────────────────────")
    logger.info(f"  Mean cosine sim:   {r['reflexive_score_mean']} ± {r['reflexive_score_std']}")
    logger.info(f"  {r['interpretation']}")
    logger.info("=" * 70)


# ==============================================================================
# SAVE
# ==============================================================================
def save_results(results, generated_texts, reference_texts, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "eval_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {output_dir}/eval_results.json")

    samples = [
        {"id": i, "generated": gen, "reference": ref}
        for i, (gen, ref) in enumerate(zip(generated_texts, reference_texts))
    ]
    with open(os.path.join(output_dir, "generation_samples.json"), 'w') as f:
        json.dump(samples, f, indent=2)
    logger.info(f"Generation samples saved: {output_dir}/generation_samples.json")

    df = pd.DataFrame([
        {
            "label":       label,
            "F1":          results['per_class'][label]['F1'],
            "AUC_ROC":     results['per_class'][label]['AUC_ROC'],
            "sensitivity": results['per_class'][label]['sensitivity'],
            "specificity": results['per_class'][label]['specificity'],
            "threshold":   results['per_class'][label]['threshold'],
        }
        for label in CHEXPERT_LABELS
    ])
    df.to_csv(os.path.join(output_dir, "per_class_metrics.csv"), index=False)
    logger.info(f"Per-class metrics saved: {output_dir}/per_class_metrics.csv")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate RRA model")
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--vision_ckpt", default=VISION_CKPT)
    parser.add_argument("--val_csv",        default=VAL_CSV)
    parser.add_argument("--data_root",      default=DATA_ROOT)
    parser.add_argument("--output_dir",     default=OUTPUT_DIR)
    parser.add_argument("--gen_samples",    type=int, default=1000)
    parser.add_argument("--batch_size",     type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers",    type=int, default=2)
    parser.add_argument("--threshold_file", default=None, help="Optional JSON file containing classification thresholds to avoid test set leakage.")
    args = parser.parse_args()

    setup_logging(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Immutable run record — written to both console and evaluation_log.txt
    logger.info("=" * 60)
    logger.info("RRA EVALUATION — STARTING")
    logger.info("=" * 60)
    logger.info(f"   Checkpoint:    {args.checkpoint_dir}")
    logger.info(f"   Val CSV:       {args.val_csv}")
    logger.info(f"   Data root:     {args.data_root}")
    logger.info(f"   Output dir:    {args.output_dir}")
    logger.info(f"   Device:        {device}")
    logger.info(f"   Batch size:    {args.batch_size}  (classification pass, no grad)")
    logger.info(f"   Gen tokens:    {MAX_GEN_TOKENS}  (nucleus sampling: top-p=0.9, temp=0.7)")
    logger.info(f"   Gen samples:   {args.gen_samples}")
    logger.info(f"   Thresh method: " + (f"loaded from {args.threshold_file}" if args.threshold_file else "optimized on dataset (warning: test leakage if run on test set)"))
    logger.info(f"   Vision ckpt:   {args.vision_ckpt}")
    logger.info(f"   Model ID:      {MODEL_ID}")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = EvalDataset(args.val_csv, args.data_root, tokenizer, transform)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    model = load_model(args.checkpoint_dir, device, args.vision_ckpt)

    results, generated_texts, reference_texts = evaluate(
        model, loader, tokenizer, device, args.gen_samples,
        threshold_file=args.threshold_file
    )

    print_results(results)
    save_results(results, generated_texts, reference_texts, args.output_dir)


if __name__ == "__main__":
    main()
