import os
import json
import logging
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ==============================================================================
# METRICS TRACKER - COMPREHENSIVE LOGGING MODULE
# ==============================================================================
class StabilityMetricsTracker:
    """Comprehensive tracker for training stability and quality metrics"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Training Stability Metrics
        self.grad_norms = []
        self.grad_norms_after_clip = []
        self.max_grad_norms = []
        self.grad_clip_counts = []
        self.total_steps = []

        self.nan_loss_counts = []
        self.inf_loss_counts = []
        self.skipped_steps = []

        # Loss Component Tracking
        self.lm_losses = []
        self.reflex_losses = []
        self.clinical_losses = []
        self.total_losses = []

        # Multimodal Utilization
        self.visual_norms = []
        self.text_norms = []
        self.visual_text_ratios = []

        # Clinical Learning Behavior
        self.positive_pred_rates = defaultdict(list)
        self.rare_class_sensitivities = defaultdict(list)

        # Generation Quality
        self.unique_token_ratios = []
        self.repetition_rates = []
        self.report_lengths = []
        self.vocab_diversities = []

        # Calibration
        self.confidence_distributions = []

        # Per-epoch summaries
        self.epoch_summaries = []

        logger.info(f"Metrics tracker initialized. Saving to: {self.metrics_dir}")

    def log_gradient_stats(self, model, max_grad_norm):
        """Track gradient statistics before clipping.

        FIX #6: Skip inf/nan parameter gradients so they don't corrupt
        np.mean() / np.max() in the epoch summary.

        Returns:
            tuple: (pre_clip_norm, max_norm, was_clipped)
        """
        total_norm = 0.0
        max_norm = 0.0

        for p in model.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.data.norm(2).item()
                # FIX #6: Guard against inf/nan individual parameter gradients
                if not (np.isnan(param_norm) or np.isinf(param_norm)):
                    total_norm += param_norm ** 2
                    max_norm = max(max_norm, param_norm)

        total_norm = total_norm ** 0.5

        self.grad_norms.append(total_norm)
        self.max_grad_norms.append(max_norm)

        # Track if gradient was clipped
        was_clipped = 1 if total_norm > max_grad_norm else 0
        self.grad_clip_counts.append(was_clipped)

        return total_norm, max_norm, was_clipped

    def log_post_clip_norm(self, post_clip_norm):
        """Log the ACTUAL gradient norm after clipping"""
        self.grad_norms_after_clip.append(float(post_clip_norm))

    def log_loss_components(self, lm_loss, reflex_loss, clinical_loss, total_loss):
        """Track individual loss components"""
        self.lm_losses.append(float(lm_loss))
        self.reflex_losses.append(float(reflex_loss))
        self.clinical_losses.append(float(clinical_loss))
        self.total_losses.append(float(total_loss))

    def log_nan_inf_events(self, loss):
        """Track numerical instabilities.

        Args:
            loss: Should be detached and cast to float32 to avoid AMP artifacts
        """
        # Ensure loss is in float32 and detached
        if loss.dtype != torch.float32:
            loss = loss.float()
        if loss.requires_grad:
            loss = loss.detach()

        has_nan = torch.isnan(loss).any().item()
        has_inf = torch.isinf(loss).any().item()

        self.nan_loss_counts.append(int(has_nan))
        self.inf_loss_counts.append(int(has_inf))

        return has_nan or has_inf

    def log_multimodal_utilization(self, visual_features, text_embeddings):
        """Track visual vs text contribution"""
        with torch.no_grad():
            visual_norm = visual_features.norm(dim=-1).mean().item()
            text_norm = text_embeddings.norm(dim=-1).mean().item()

            self.visual_norms.append(visual_norm)
            self.text_norms.append(text_norm)

            ratio = visual_norm / (text_norm + 1e-8)
            self.visual_text_ratios.append(ratio)

    def log_clinical_predictions(self, predictions, targets, class_names):
        """Track positive prediction rates per class.

        Args:
            predictions: Probabilities [B, num_classes] - should be in [0, 1] range
            targets: Binary labels [B, num_classes]
            class_names: List of class names
        """
        with torch.no_grad():
            # Ensure predictions are probabilities
            if predictions.max() > 1.0 or predictions.min() < 0.0:
                logger.warning("Predictions not in [0,1] range - applying sigmoid")
                predictions = torch.sigmoid(predictions)

            preds_np = predictions.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for i, class_name in enumerate(class_names):
                pos_rate = (preds_np[:, i] > 0.5).mean()
                self.positive_pred_rates[class_name].append(float(pos_rate))

                # Sensitivity for positive cases
                if targets_np[:, i].sum() > 0:
                    sensitivity = ((preds_np[:, i] > 0.5) & (targets_np[:, i] > 0.5)).sum() / targets_np[:, i].sum()
                    self.rare_class_sensitivities[class_name].append(float(sensitivity))

    def log_generation_quality(self, generated_ids, tokenizer):
        """Track generation quality metrics"""
        with torch.no_grad():
            # Convert to tokens
            tokens = generated_ids.cpu().numpy().flatten()
            tokens = tokens[tokens != tokenizer.pad_token_id]

            # Length
            length = len(tokens)
            self.report_lengths.append(length)

            # Unique token ratio
            unique_ratio = len(set(tokens)) / max(length, 1)
            self.unique_token_ratios.append(unique_ratio)

            # Repetition rate (trigram repetition)
            trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
            if len(trigrams) > 0:
                repetition = 1.0 - len(set(trigrams)) / len(trigrams)
                self.repetition_rates.append(repetition)
            else:
                self.repetition_rates.append(0.0)

    def log_confidence_distribution(self, clinical_logits):
        """Track confidence/calibration"""
        with torch.no_grad():
            confidences = torch.sigmoid(clinical_logits).cpu().numpy()
            self.confidence_distributions.append(confidences.flatten())

    def compute_epoch_summary(self, epoch):
        """Compute summary statistics for the epoch"""
        if len(self.grad_norms) == 0:
            return {}

        # Calculate indices for this epoch's data
        steps_this_epoch = len(self.grad_norms) - sum([s['total_steps'] for s in self.epoch_summaries])
        if steps_this_epoch <= 0:
            steps_this_epoch = len(self.grad_norms)

        start_idx = len(self.grad_norms) - steps_this_epoch

        summary = {
            'epoch': epoch,
            'total_steps': steps_this_epoch,

            # Gradient Statistics
            'mean_grad_norm': np.mean(self.grad_norms[start_idx:]),
            'max_grad_norm': np.max(self.max_grad_norms[start_idx:]),
            'std_grad_norm': np.std(self.grad_norms[start_idx:]),
            'pct_clipped': 100 * np.mean(self.grad_clip_counts[start_idx:]),

            # Numerical Stability
            'nan_events': sum(self.nan_loss_counts[start_idx:]),
            'inf_events': sum(self.inf_loss_counts[start_idx:]),

            # Loss Trends
            'mean_lm_loss': np.mean(self.lm_losses[start_idx:]),
            'mean_reflex_loss': np.mean(self.reflex_losses[start_idx:]),
            'mean_clinical_loss': np.mean(self.clinical_losses[start_idx:]),
            'mean_total_loss': np.mean(self.total_losses[start_idx:]),

            # Multimodal Balance
            'mean_visual_norm': np.mean(self.visual_norms[start_idx:]) if self.visual_norms else 0.0,
            'mean_text_norm': np.mean(self.text_norms[start_idx:]) if self.text_norms else 0.0,
            'mean_visual_text_ratio': np.mean(self.visual_text_ratios[start_idx:]) if self.visual_text_ratios else 0.0,

            # Generation Quality
            'mean_unique_token_ratio': np.mean(self.unique_token_ratios[start_idx:]) if self.unique_token_ratios else 0.0,
            'mean_repetition_rate': np.mean(self.repetition_rates[start_idx:]) if self.repetition_rates else 0.0,
            'mean_report_length': np.mean(self.report_lengths[start_idx:]) if self.report_lengths else 0.0,
        }

        self.epoch_summaries.append(summary)
        return summary

    def save_all_metrics(self):
        """Save all tracked metrics to disk"""

        # 1. Save raw metrics
        metrics_data = {
            'grad_norms': self.grad_norms,
            'grad_norms_after_clip': self.grad_norms_after_clip,
            'max_grad_norms': self.max_grad_norms,
            'grad_clip_counts': self.grad_clip_counts,
            'lm_losses': self.lm_losses,
            'reflex_losses': self.reflex_losses,
            'clinical_losses': self.clinical_losses,
            'total_losses': self.total_losses,
            'visual_norms': self.visual_norms,
            'text_norms': self.text_norms,
            'visual_text_ratios': self.visual_text_ratios,
            'unique_token_ratios': self.unique_token_ratios,
            'repetition_rates': self.repetition_rates,
            'report_lengths': self.report_lengths,
            'nan_loss_counts': self.nan_loss_counts,
            'inf_loss_counts': self.inf_loss_counts,
        }

        with open(os.path.join(self.metrics_dir, 'raw_metrics.json'), 'w') as f:
            json.dump(metrics_data, f, indent=2)

        # 2. Save per-class metrics
        with open(os.path.join(self.metrics_dir, 'clinical_class_metrics.json'), 'w') as f:
            class_metrics = {
                'positive_prediction_rates': dict(self.positive_pred_rates),
                'rare_class_sensitivities': dict(self.rare_class_sensitivities),
            }
            json.dump(class_metrics, f, indent=2)

        # 3. Save epoch summaries
        with open(os.path.join(self.metrics_dir, 'epoch_summaries.json'), 'w') as f:
            json.dump(self.epoch_summaries, f, indent=2)

        # 4. Create comprehensive plots
        self.plot_comprehensive_metrics()

        logger.info(f"All metrics saved to {self.metrics_dir}")

    def plot_comprehensive_metrics(self):
        """Create comprehensive visualization of all metrics"""

        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Gradient Norm Statistics (with pre/post clip comparison)
        ax1 = fig.add_subplot(gs[0, 0])
        if self.grad_norms:
            steps = range(len(self.grad_norms))
            ax1.plot(steps, self.grad_norms, alpha=0.5, label='Pre-Clip', color='blue')
            if self.grad_norms_after_clip:
                ax1.plot(steps, self.grad_norms_after_clip, alpha=0.5,
                        label='Post-Clip', color='green')
            ax1.axhline(y=np.mean(self.grad_norms), color='blue',
                       linestyle='--', alpha=0.7, label=f'Mean Pre: {np.mean(self.grad_norms):.3f}')
            if self.grad_norms_after_clip:
                ax1.axhline(y=np.mean(self.grad_norms_after_clip), color='green',
                           linestyle='--', alpha=0.7, label=f'Mean Post: {np.mean(self.grad_norms_after_clip):.3f}')
            ax1.set_title('Gradient Norms: Pre vs Post Clipping')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Grad Norm')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

        # 2. Loss Components
        ax2 = fig.add_subplot(gs[0, 1])
        if self.lm_losses:
            steps = range(len(self.lm_losses))
            ax2.plot(steps, self.lm_losses, label='LM Loss', alpha=0.7)
            ax2.plot(steps, self.reflex_losses, label='Reflex Loss', alpha=0.7)
            ax2.plot(steps, self.clinical_losses, label='Clinical Loss', alpha=0.7)
            ax2.set_title('Loss Components')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Gradient Clipping Rate
        ax3 = fig.add_subplot(gs[0, 2])
        if self.grad_clip_counts:
            clip_rate = pd.Series(self.grad_clip_counts).rolling(100).mean() * 100
            ax3.plot(clip_rate, linewidth=2, color='orange')
            ax3.set_title('Gradient Clipping Rate (%)')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('% Steps Clipped')
            ax3.grid(True, alpha=0.3)

        # 4. Visual vs Text Norm Ratio
        ax4 = fig.add_subplot(gs[1, 0])
        if self.visual_text_ratios:
            ax4.plot(self.visual_text_ratios, alpha=0.6)
            ax4.plot(pd.Series(self.visual_text_ratios).rolling(50).mean(),
                    linewidth=2, color='purple', label='Smoothed')
            ax4.axhline(y=1.0, color='red', linestyle='--', label='Balance')
            ax4.set_title('Visual/Text Embedding Ratio')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Ratio')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. Generation Quality - Unique Token Ratio
        ax5 = fig.add_subplot(gs[1, 1])
        if self.unique_token_ratios:
            ax5.plot(self.unique_token_ratios, alpha=0.6, color='green')
            ax5.plot(pd.Series(self.unique_token_ratios).rolling(20).mean(),
                    linewidth=2, color='darkgreen', label='Smoothed')
            ax5.set_title('Unique Token Ratio (Diversity)')
            ax5.set_xlabel('Step')
            ax5.set_ylabel('Ratio')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. Repetition Rate
        ax6 = fig.add_subplot(gs[1, 2])
        if self.repetition_rates:
            ax6.plot(self.repetition_rates, alpha=0.6, color='red')
            ax6.plot(pd.Series(self.repetition_rates).rolling(20).mean(),
                    linewidth=2, color='darkred', label='Smoothed')
            ax6.set_title('Repetition Rate (Lower is Better)')
            ax6.set_xlabel('Step')
            ax6.set_ylabel('Rate')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # 7. Report Lengths
        ax7 = fig.add_subplot(gs[2, 0])
        if self.report_lengths:
            ax7.hist(self.report_lengths, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax7.axvline(np.mean(self.report_lengths), color='red',
                       linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.report_lengths):.1f}')
            ax7.set_title('Generated Report Length Distribution')
            ax7.set_xlabel('Length (tokens)')
            ax7.set_ylabel('Frequency')
            ax7.legend()
            ax7.grid(True, alpha=0.3)

        # 8. Confidence Distribution (latest)
        ax8 = fig.add_subplot(gs[2, 1])
        if self.confidence_distributions:
            latest_conf = np.concatenate(self.confidence_distributions[-10:])
            ax8.hist(latest_conf, bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax8.set_title('Clinical Classifier Confidence (Recent)')
            ax8.set_xlabel('Sigmoid Output')
            ax8.set_ylabel('Frequency')
            ax8.grid(True, alpha=0.3)

        # 9. Epoch Summary - Gradient Stats
        ax9 = fig.add_subplot(gs[2, 2])
        if self.epoch_summaries:
            epochs = [s['epoch'] for s in self.epoch_summaries]
            mean_norms = [s['mean_grad_norm'] for s in self.epoch_summaries]
            max_norms = [s['max_grad_norm'] for s in self.epoch_summaries]
            ax9.plot(epochs, mean_norms, marker='o', label='Mean', linewidth=2)
            ax9.plot(epochs, max_norms, marker='s', label='Max', linewidth=2)
            ax9.set_title('Per-Epoch Gradient Statistics')
            ax9.set_xlabel('Epoch')
            ax9.set_ylabel('Grad Norm')
            ax9.legend()
            ax9.grid(True, alpha=0.3)

        # 10. Per-Class Positive Prediction Rate
        ax10 = fig.add_subplot(gs[3, 0])
        if self.positive_pred_rates:
            for class_name, rates in list(self.positive_pred_rates.items())[:5]:
                if rates:
                    ax10.plot(rates, label=class_name, alpha=0.7)
            ax10.set_title('Positive Prediction Rate (Top 5 Classes)')
            ax10.set_xlabel('Step')
            ax10.set_ylabel('Rate')
            ax10.legend(fontsize=8)
            ax10.grid(True, alpha=0.3)

        # 11. Rare Class Sensitivity
        ax11 = fig.add_subplot(gs[3, 1])
        rare_classes = ['Pneumothorax', 'Fracture', 'Lung Lesion']
        if self.rare_class_sensitivities:
            for class_name in rare_classes:
                if class_name in self.rare_class_sensitivities and self.rare_class_sensitivities[class_name]:
                    ax11.plot(self.rare_class_sensitivities[class_name],
                             marker='o', label=class_name, linewidth=2)
            ax11.set_title('Rare Class Sensitivity Trend')
            ax11.set_xlabel('Validation Step')
            ax11.set_ylabel('Sensitivity')
            ax11.legend()
            ax11.grid(True, alpha=0.3)

        # 12. Numerical Stability Events
        ax12 = fig.add_subplot(gs[3, 2])
        if self.epoch_summaries:
            epochs = [s['epoch'] for s in self.epoch_summaries]
            nan_events = [s['nan_events'] for s in self.epoch_summaries]
            inf_events = [s['inf_events'] for s in self.epoch_summaries]

            x = np.arange(len(epochs))
            width = 0.35
            ax12.bar(x - width/2, nan_events, width, label='NaN', color='red', alpha=0.7)
            ax12.bar(x + width/2, inf_events, width, label='Inf', color='orange', alpha=0.7)
            ax12.set_title('Numerical Instability Events')
            ax12.set_xlabel('Epoch')
            ax12.set_ylabel('Count')
            ax12.set_xticks(x)
            ax12.set_xticklabels([f'E{e}' for e in epochs])
            ax12.legend()
            ax12.grid(True, alpha=0.3, axis='y')

        plt.savefig(os.path.join(self.metrics_dir, 'comprehensive_metrics.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

        # Additional plot: Epoch summary table
        if self.epoch_summaries:
            self.create_epoch_summary_table()

    def create_epoch_summary_table(self):
        """Create a formatted table of epoch summaries"""
        fig, ax = plt.subplots(figsize=(16, len(self.epoch_summaries) * 0.6 + 2))
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data
        headers = ['Epoch', 'Mean GradNorm', 'Max GradNorm', '% Clipped',
                  'LM Loss', 'Reflex Loss', 'Clinical Loss', 'V/T Ratio']

        table_data = []
        for s in self.epoch_summaries:
            row = [
                f"{s['epoch']}",
                f"{s['mean_grad_norm']:.4f}",
                f"{s['max_grad_norm']:.4f}",
                f"{s['pct_clipped']:.2f}%",
                f"{s['mean_lm_loss']:.4f}",
                f"{s['mean_reflex_loss']:.4f}",
                f"{s['mean_clinical_loss']:.4f}",
                f"{s.get('mean_visual_text_ratio', 0.0):.3f}",
            ]
            table_data.append(row)

        table = ax.table(cellText=table_data, colLabels=headers,
                        loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        plt.savefig(os.path.join(self.metrics_dir, 'epoch_summary_table.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

    def print_epoch_summary(self, epoch):
        """Print formatted epoch summary to console"""
        summary = self.compute_epoch_summary(epoch)

        logger.info("\n" + "="*80)
        logger.info(f"EPOCH {epoch} STABILITY & QUALITY METRICS")
        logger.info("="*80)

        logger.info("\nTraining Stability:")
        logger.info(f"   Mean Grad Norm:        {summary['mean_grad_norm']:.4f}")
        logger.info(f"   Max Grad Norm:         {summary['max_grad_norm']:.4f}")
        logger.info(f"   Std Grad Norm:         {summary['std_grad_norm']:.4f}")
        logger.info(f"   % Steps Clipped:       {summary['pct_clipped']:.2f}%")
        logger.info(f"   NaN Events:            {summary['nan_events']}")
        logger.info(f"   Inf Events:            {summary['inf_events']}")

        logger.info("\nLoss Components:")
        logger.info(f"   LM Loss:               {summary['mean_lm_loss']:.4f}")
        logger.info(f"   Reflex Loss:           {summary['mean_reflex_loss']:.4f}")
        logger.info(f"   Clinical Loss:         {summary['mean_clinical_loss']:.4f}")
        logger.info(f"   Total Loss:            {summary['mean_total_loss']:.4f}")

        if summary['mean_visual_norm'] > 0:
            logger.info("\nMultimodal Balance:")
            logger.info(f"   Visual Norm:           {summary['mean_visual_norm']:.4f}")
            logger.info(f"   Text Norm:             {summary['mean_text_norm']:.4f}")
            logger.info(f"   Visual/Text Ratio:     {summary['mean_visual_text_ratio']:.4f}")

        if summary['mean_unique_token_ratio'] > 0:
            logger.info("\nGeneration Quality:")
            logger.info(f"   Unique Token Ratio:    {summary['mean_unique_token_ratio']:.4f}")
            logger.info(f"   Repetition Rate:       {summary['mean_repetition_rate']:.4f}")
            logger.info(f"   Mean Report Length:    {summary['mean_report_length']:.1f} tokens")

        logger.info("="*80 + "\n")

    def plot_training_progress(history):
        """Plot and save training progress"""

        if not history or 'train_total' not in history:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(history['train_total'], label='Train Total', linewidth=2)
        if 'val_clinical_loss' in history and len(history['val_clinical_loss']) > 0:
            val_steps = np.linspace(0, len(history['train_total']), len(history['val_clinical_loss']))
            axes[0, 0].plot(val_steps, history['val_clinical_loss'], label='Val Clinical', linewidth=2, marker='o')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        if 'train_lm' in history:
            axes[0, 1].plot(history['train_lm'], label='LM', linewidth=2)
            axes[0, 1].plot(history['train_reflex'], label='Reflex', linewidth=2)
            axes[0, 1].plot(history['train_clinical'], label='Clinical', linewidth=2)
            axes[0, 1].set_title('Loss Components')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        if 'val_f1' in history and len(history['val_f1']) > 0:
            axes[1, 0].plot(history['val_f1'], marker='o', linewidth=2, color='green')
            axes[1, 0].set_title('Validation F1 Score')
            axes[1, 0].set_xlabel('Validation Steps')
            axes[1, 0].set_ylabel('F1')
            axes[1, 0].grid(True, alpha=0.3)

        if 'val_precision' in history and len(history['val_precision']) > 0:
            axes[1, 1].plot(history['val_precision'], marker='o', linewidth=2, label='Precision')
            axes[1, 1].plot(history['val_recall'], marker='s', linewidth=2, label='Recall')
            axes[1, 1].set_title('Precision & Recall')
            axes[1, 1].set_xlabel('Validation Steps')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'training_progress.png'), dpi=150)
        plt.close()

        with open(os.path.join(cfg.OUTPUT_DIR, 'training_history.json'), 'w') as f:
            history_serializable = {k: [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                                    for k, v in history.items() if isinstance(v, list)}
            json.dump(history_serializable, f, indent=2)
