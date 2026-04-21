import pandas as pd
import numpy as np
import ast
import os
import re
import shutil
from pathlib import Path
from tqdm import tqdm

SOURCE_DIR = os.environ.get("RRA_SOURCE_DIR", "data/source")
DEST_DIR = os.environ.get("RRA_DEST_DIR", "data/split")

# Number of images requested per split
SPLITS = {
    'train': 10000,
    'val': 1000,
    'test': 1000
}

def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return "No significant findings."
    text = re.sub(r"(?i)^(findings|impression|indication|history)[:\s]*", "", text.strip())
    text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DATE]", text)
    text = re.sub(r"\b\d{1,2}[:.]\d{2}\s*(?:a\.?m\.?|p\.?m\.?)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:[01]\d|2[0-3]):[0-5]\d\b", "", text)
    text = re.sub(r"\b___+\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text.split()) >= 5 else "No significant findings."

def main():
    print("Loading datasets...")
    df_train = pd.read_csv(os.path.join(SOURCE_DIR, 'mimic_micro_train_portable.csv'))
    df_chexpert = pd.read_csv(os.path.join(SOURCE_DIR, 'mimic-cxr-2.0.0-chexpert.csv'))
    
    # Keep label order exactly aligned with CHEXPERT_LABELS in training/train_model.py
    label_cols = [
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
        'Support Devices',
    ]
    
    # Process chexpert labels
    # We will fill NA with 0.0 and replace -1.0 with 1.0 (positive)
    df_chexpert[label_cols] = df_chexpert[label_cols].fillna(0.0)
    df_chexpert[label_cols] = df_chexpert[label_cols].replace(-1.0, 1.0)
    
    print("Parsing train CSV and exploding images...")
    # Explode the original df so each row has 1 image
    records = []
    
    # Compile regex to extract study_id from image path
    # Example path: files/p15/p15734757/s57229866/7e24205a-26bc2f52.jpg
    study_re = re.compile(r'/s(\d+)/')
    
    for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
        try:
            images = ast.literal_eval(row['image'])
        except Exception:
            continue
            
        subject_id = row['subject_id']
        raw_report = row['text']
        cleaned_report = clean_text(raw_report)
        complexity = len(cleaned_report.split())
        
        for img_path in images:
            match = study_re.search(img_path)
            if match:
                study_id = int(match.group(1))
            else:
                study_id = -1
                
            records.append({
                'image_path': img_path,
                'report': raw_report, # keeping raw since dataloader cleans it, wait... dataloader uses clean_text() on row['report']. So raw is fine. Actually dataloader says `clean_text(row['report'])`.
                'subject_id': subject_id,
                'study_id': study_id,
                'complexity': complexity
            })
            
    df_exploded = pd.DataFrame(records)
    print(f"Total potential images: {len(df_exploded)}")
    
    # Merge with chexpert to get labels
    print("Merging with CheXpert labels...")
    df_merged = pd.merge(df_exploded, df_chexpert[['subject_id', 'study_id'] + label_cols], 
                         on=['subject_id', 'study_id'], how='inner')
    
    print(f"Images after merging with labels: {len(df_merged)}")
    
    # Format labels as string representations of lists
    # the EvalDataset does ast.literal_eval(x)
    def make_label_list(row):
        return str([float(row[col]) for col in label_cols])
        
    print("Formatting labels...")
    df_merged['clinical_labels'] = df_merged.apply(make_label_list, axis=1)
    
    # Group by subject
    print("Splitting by subject avoiding leakage...")
    subjects = df_merged['subject_id'].unique()
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(subjects)
    
    # Build out the requested split sizes while preventing subject leakage.
    split_data = {
        'train': [],
        'val': [],
        'test': []
    }
    
    split_counts = {
        'train': 0,
        'val': 0,
        'test': 0
    }
    
    subject_groups = df_merged.groupby('subject_id')
    
    # Assign subjects to buckets
    for subject_id in subjects:
        group_df = subject_groups.get_group(subject_id)
        
        # Fill train first, then val, then test
        for target_split in ['train', 'val', 'test']:
            needed = SPLITS[target_split] - split_counts[target_split]
            if needed > 0:
                if len(group_df) <= needed:
                    # add entire subject's images
                    group_df_assigned = group_df.copy()
                    group_df_assigned['split'] = target_split
                    split_data[target_split].append(group_df_assigned)
                    split_counts[target_split] += len(group_df)
                else:
                    # Truncate this subject's images to exactly meet the count
                    # The rest of the subject's images are DISCARDED to avoid leakage
                    group_df_assigned = group_df.iloc[:needed].copy()
                    group_df_assigned['split'] = target_split
                    split_data[target_split].append(group_df_assigned)
                    split_counts[target_split] += needed
                # Once a subject is assigned to a split, break! They can't be in multiple splits.
                break
                
        # Optimization: break early if all full
        if all(split_counts[k] == SPLITS[k] for k in SPLITS):
            break
            
    print("Split counts:")
    for k, v in split_counts.items():
        print(f"  {k}: {v} / {SPLITS[k]}")
        
    assert all(split_counts[k] == SPLITS[k] for k in SPLITS), "Not enough data to fulfill Exact counts!"
    
    # Create structure and copy files
    print("Creating destination directory and copying files (this might take a few minutes)...")
    metadata_dest_dir = os.path.join(DEST_DIR, 'metadata')
    images_dest_dir = DEST_DIR
    os.makedirs(metadata_dest_dir, exist_ok=True)
    os.makedirs(images_dest_dir, exist_ok=True)
    
    source_images_dir = os.path.join(SOURCE_DIR, 'images')
    
    final_dfs = {}
    for split_name in ['train', 'val', 'test']:
        df_split = pd.concat(split_data[split_name], ignore_index=True)
        final_dfs[split_name] = df_split
        
        # We need the following columns exactly
        output_cols = ['image_path', 'report', 'subject_id', 'study_id', 'split', 'clinical_labels', 'complexity']
        df_split_out = df_split[output_cols]
        
        csv_path = os.path.join(metadata_dest_dir, f"{split_name}.csv")
        df_split_out.to_csv(csv_path, index=False)
        print(f"Saved {csv_path} with {len(df_split_out)} rows.")
        
    print("Copying physical image files...")
    # To copy quickly, we iterate over all selected paths
    all_selected_paths = []
    for split_name in ['train', 'val', 'test']:
        all_selected_paths.extend(final_dfs[split_name]['image_path'].tolist())
        
    for rel_path in tqdm(all_selected_paths, desc="Copying images"):
        src_file = os.path.join(source_images_dir, rel_path)
        dst_file = os.path.join(images_dest_dir, rel_path)
        
        # Create directories if needed
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        
        if os.path.exists(src_file) and not os.path.exists(dst_file):
            shutil.copy2(src_file, dst_file)
            
    print("Done!")

if __name__ == "__main__":
    main()
