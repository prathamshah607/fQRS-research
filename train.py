"""
train.py - Training Pipeline for CSIRO Image2Biomass Competition
Fixed: Handles row-to-image index mapping for CV splits
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
import pickle
from datetime import datetime
from PIL import Image

from model import MultimodalBiomassModel


# ============================================================
# Dataset for Multi-Target Format
# ============================================================

class BiomassDataset(Dataset):
    """
    Dataset that handles the multi-row format (5 rows per image).
    Returns all 5 targets for a single image.
    """
    def __init__(self, csv_path, image_root, transform=None, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform
        self.is_train = is_train
        
        # Group by base sample_id (without __target suffix)
        self.df['base_id'] = self.df['sample_id'].str.split('__').str[0]
        self.unique_samples = self.df['base_id'].unique()
        
        print(f"  Dataset: {len(self.unique_samples)} unique images ({len(self.df)} total rows)")
    
    def __len__(self):
        return len(self.unique_samples)
    
    def __getitem__(self, idx):
        base_id = self.unique_samples[idx]
        
        # Get all 5 rows for this image
        sample_rows = self.df[self.df['base_id'] == base_id]
        
        if len(sample_rows) != 5:
            raise ValueError(f"Expected 5 rows for {base_id}, got {len(sample_rows)}")
        
        # Get image
        image_path = sample_rows.iloc[0]['image_path']
        full_path = os.path.join(self.image_root, image_path)
        image = Image.open(full_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get numeric features
        numeric_cols = ['ndvi_norm', 'height_norm', 'date_sin', 'date_cos', 
                        'date_sin2', 'date_cos2', 'ndvi_height_interaction', 'seasonal_ndvi']
        numeric = torch.tensor(
            sample_rows.iloc[0][numeric_cols].values.astype(np.float32),
            dtype=torch.float32
        )
        
        # Get categorical features
        categorical = torch.tensor(
            [sample_rows.iloc[0]['state_id'], sample_rows.iloc[0]['species_id']],
            dtype=torch.long
        )
        
        if self.is_train:
            # Get all 5 targets
            targets = {}
            for _, row in sample_rows.iterrows():
                target_name = row['target_name']
                targets[target_name] = torch.tensor(row['target'], dtype=torch.float32)
            
            return image, numeric, categorical, targets
        else:
            return image, numeric, categorical


# ============================================================
# Transforms
# ============================================================

from torchvision import transforms

def get_train_transforms(img_size=518):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(img_size=518):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ============================================================
# Loss Function
# ============================================================

def weighted_r2_loss_with_constraints(preds, targets, weights, lambda_constraint=0.05):
    weighted_loss = 0.0
    metrics = {}
    
    for target, weight in weights.items():
        y_true = targets[target]
        y_pred = preds[target]
        
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
        
        loss_component = weight * (1.0 - r2)
        weighted_loss += loss_component
        
        metrics[f'r2_{target}'] = r2.detach().item()
    
    # Constraint penalty
    component_sum = (preds['Dry_Green_g'] + 
                     preds['Dry_Dead_g'] + 
                     preds['Dry_Clover_g'])
    
    violation = torch.relu(component_sum - preds['Dry_Total_g'] * 1.05)
    constraint_penalty = torch.mean(violation ** 2)
    
    total_loss = weighted_loss + lambda_constraint * constraint_penalty
    
    metrics['constraint_violation'] = constraint_penalty.detach().item()
    metrics['weighted_r2_loss'] = weighted_loss.detach().item()
    
    return total_loss, metrics


# ============================================================
# Training Functions
# ============================================================

def train_one_epoch(model, dataloader, optimizer, scheduler, device, weights, lambda_constraint=0.05):
    model.train()
    total_loss = 0.0
    metrics_sum = {}
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        images, numeric, categorical, targets = batch
        
        images = images.to(device)
        numeric = numeric.to(device)
        categorical = categorical.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        preds = model(images, numeric, categorical)
        
        loss, metrics = weighted_r2_loss_with_constraints(
            preds, targets, weights, lambda_constraint
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in metrics.items():
            metrics_sum[k] = metrics_sum.get(k, 0) + v
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'r2_Total': f"{metrics.get('r2_Dry_Total_g', 0):.4f}"
        })
    
    scheduler.step()
    
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}
    
    return avg_loss, avg_metrics


@torch.no_grad()
def validate(model, dataloader, device, weights):
    model.eval()
    all_preds = {t: [] for t in weights.keys()}
    all_targets = {t: [] for t in weights.keys()}
    
    for batch in tqdm(dataloader, desc='Validating', leave=False):
        images, numeric, categorical, targets = batch
        
        images = images.to(device)
        numeric = numeric.to(device)
        categorical = categorical.to(device)
        
        preds = model(images, numeric, categorical)
        
        for target in weights.keys():
            all_preds[target].append(preds[target].cpu().numpy())
            all_targets[target].append(targets[target].cpu().numpy())
    
    all_preds = {k: np.concatenate(v) for k, v in all_preds.items()}
    all_targets = {k: np.concatenate(v) for k, v in all_targets.items()}
    
    weighted_r2 = 0.0
    per_target_r2 = {}
    
    print("\n  Per-Target Performance:")
    for target, weight in weights.items():
        y_true = all_targets[target]
        y_pred = all_preds[target]
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
        
        weighted_r2 += weight * r2
        per_target_r2[target] = r2
        
        print(f"    {target:20s}: R² = {r2:.4f} (weight={weight})")
    
    print(f"\n    Weighted R²: {weighted_r2:.4f}")
    
    return weighted_r2, per_target_r2


# ============================================================
# Logging
# ============================================================

class TrainingLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.logs = {'folds': {}}
        os.makedirs(log_dir, exist_ok=True)
    
    def log_epoch(self, fold, epoch, train_loss, train_metrics, val_r2, per_target_r2):
        if fold not in self.logs['folds']:
            self.logs['folds'][fold] = []
        
        self.logs['folds'][fold].append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_metrics': train_metrics,
            'val_r2': val_r2,
            'per_target_r2': per_target_r2,
            'timestamp': datetime.now().isoformat()
        })
    
    def save(self):
        with open(f'{self.log_dir}/training_log.json', 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"\n✓ Logs saved to {self.log_dir}/training_log.json")


# ============================================================
# Fold Training
# ============================================================

def train_fold(fold, train_loader, val_loader, model, optimizer, scheduler,
               device, weights, config, logger):
    best_r2 = -np.inf
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*60}")
        print(f"Fold {fold} | Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*60}")
        
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, 
            device, weights, config['lambda_constraint']
        )
        
        val_r2, per_target_r2 = validate(model, val_loader, device, weights)
        
        logger.log_epoch(fold, epoch, train_loss, train_metrics, val_r2, per_target_r2)
        
        print(f"\n  Train Loss: {train_loss:.4f} | Val Weighted R²: {val_r2:.4f}")
        
        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(
                model.state_dict(), 
                f"{config['model_dir']}/best_model_fold{fold}.pt"
            )
            patience_counter = 0
            print(f"  ✓ New best R²: {best_r2:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break
        
        if per_target_r2['Dry_Total_g'] < 0.75 and epoch > 20:
            print(f"  ⚠️  WARNING: Dry_Total R² = {per_target_r2['Dry_Total_g']:.4f} (target: ≥0.90)")
    
    return best_r2


# ============================================================
# Main
# ============================================================

def main():
    config = {
        'img_size': 518,
        'batch_size': 16,
        'epochs': 120,
        'patience': 20,
        'lambda_constraint': 0.05,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'model_dir': 'models',
        'log_dir': 'logs',
        'train_csv': 'train_processed.csv',
        'preprocessor_pkl': 'preprocessor.pkl',
        'cv_splits_pkl': 'cv_splits.pkl',
        'train_images_dir': '',
        'lr_dino': 5e-5,
        'lr_other': 1e-4,
        'weight_decay': 1e-5,
        'scheduler_T0': 10,
        'scheduler_Tmult': 2,
        'scheduler_eta_min': 1e-6,
    }
    
    WEIGHTS = {
        'Dry_Green_g': 0.1,
        'Dry_Dead_g': 0.1,
        'Dry_Clover_g': 0.1,
        'GDM_g': 0.2,
        'Dry_Total_g': 0.5,
    }
    
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    print("="*60)
    print("CSIRO Image2Biomass - Training Pipeline")
    print("="*60)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Image size: {config['img_size']}")
    print("="*60)
    
    # Load preprocessor
    print("\nLoading preprocessor...")
    with open(config['preprocessor_pkl'], 'rb') as f:
        prep_state = pickle.load(f)
    
    num_states = prep_state['num_states']
    num_species = prep_state['num_species']
    print(f"  States: {num_states}")
    print(f"  Species: {num_species}")
    
    # Load CV splits
    print("\nLoading CV splits...")
    with open(config['cv_splits_pkl'], 'rb') as f:
        cv_splits = pickle.load(f)
    
    # Load data
    print("\nLoading training data...")
    train_df = pd.read_csv(config['train_csv'])
    train_df['base_id'] = train_df['sample_id'].str.split('__').str[0]
    unique_ids = train_df['base_id'].unique()
    
    print(f"  Total rows: {len(train_df)}")
    print(f"  Unique images: {len(unique_ids)}")
    
    # **FIX: Map row indices to image indices**
    print("\nMapping CV splits from row indices to image indices...")
    
    # Create mapping: base_id -> image_index
    id_to_idx = {base_id: i for i, base_id in enumerate(unique_ids)}
    
    # Get base_id for each row
    row_base_ids = train_df['base_id'].values
    
    # Convert fold indices
    for fold_info in cv_splits:
        train_row_idx = fold_info['train_idx']
        val_row_idx = fold_info['val_idx']
        
        # Get base_ids for these rows
        train_base_ids = row_base_ids[train_row_idx]
        val_base_ids = row_base_ids[val_row_idx]
        
        # Convert to image indices (unique)
        train_img_idx = sorted(set([id_to_idx[bid] for bid in train_base_ids]))
        val_img_idx = sorted(set([id_to_idx[bid] for bid in val_base_ids]))
        
        # Update fold info
        fold_info['train_idx'] = train_img_idx
        fold_info['val_idx'] = val_img_idx
        
        print(f"  Fold {fold_info['fold']}: train={len(train_img_idx)}, val={len(val_img_idx)} images")
    
    # Training loop
    logger = TrainingLogger(config['log_dir'])
    fold_results = []
    
    for fold_info in cv_splits:
        fold = fold_info['fold']
        train_idx = fold_info['train_idx']
        val_idx = fold_info['val_idx']
        
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"  Train: {len(train_idx)} images")
        print(f"  Val:   {len(val_idx)} images")
        print(f"{'='*60}")
        
        # Create datasets
        train_dataset = BiomassDataset(
            config['train_csv'],
            config['train_images_dir'],
            transform=get_train_transforms(config['img_size']),
            is_train=True
        )
        val_dataset = BiomassDataset(
            config['train_csv'],
            config['train_images_dir'],
            transform=get_val_transforms(config['img_size']),
            is_train=True
        )
        
        # Create subsets
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_subset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        # Initialize model
        print(f"\n  Initializing model...")
        model = MultimodalBiomassModel(
            num_states=num_states,
            num_species=num_species
        ).to(config['device'])
        
        # Optimizer
        optimizer = AdamW([
            {'params': model.dino_backbone.parameters(), 'lr': config['lr_dino']},
            {'params': model.tabular_embedder.parameters(), 'lr': config['lr_other']},
            {'params': model.fusion.parameters(), 'lr': config['lr_other']},
            {'params': model.heads.parameters(), 'lr': config['lr_other']},
        ], weight_decay=config['weight_decay'])
        
        # Scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['scheduler_T0'],
            T_mult=config['scheduler_Tmult'],
            eta_min=config['scheduler_eta_min']
        )
        
        # Train fold
        best_r2 = train_fold(
            fold, train_loader, val_loader, model, optimizer, scheduler,
            config['device'], WEIGHTS, config, logger
        )
        
        fold_results.append(best_r2)
        print(f"\n✓ Fold {fold} Best R²: {best_r2:.4f}")
    
    # Final results
    print("\n" + "="*60)
    print("5-Fold Cross-Validation Results")
    print("="*60)
    for fold, r2 in enumerate(fold_results):
        print(f"Fold {fold}: {r2:.4f}")
    print(f"\nMean R²: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print("="*60)
    
    logger.save()
    
    results_summary = {
        'fold_results': fold_results,
        'mean_r2': float(np.mean(fold_results)),
        'std_r2': float(np.std(fold_results)),
        'config': config,
        'weights': WEIGHTS,
    }
    
    with open(f"{config['log_dir']}/final_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✓ Results saved to {config['log_dir']}/final_results.json")
    
    mean_r2 = np.mean(fold_results)
    if mean_r2 >= 0.82:
        print(f"\n🎯 EXCELLENT! Mean R² = {mean_r2:.4f} (Target: 0.82-0.85)")
    elif mean_r2 >= 0.80:
        print(f"\n✓ GOOD! Mean R² = {mean_r2:.4f} (Above baseline 0.65)")
    else:
        print(f"\n⚠️  WARNING: Mean R² = {mean_r2:.4f} (Below target 0.80)")


if __name__ == '__main__':
    main()
