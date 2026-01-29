import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class FraudDataset(Dataset):
    def __init__(self, data: pd.DataFrame, target_col: str = 'is_fraud',
                 feature_cols: Optional[List[str]] = None,
                 image_col: Optional[str] = None,
                 transform=None):
        self.data = data.reset_index(drop=True)
        self.target_col = target_col
        self.transform = transform
        
        if feature_cols is None:
            self.feature_cols = [col for col in data.columns 
                                if col not in [target_col, 'transaction_id', 'timestamp', image_col] 
                                and data[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        else:
            self.feature_cols = feature_cols
        
        self.image_col = image_col
        
        if target_col in self.data.columns:
            self.targets = self.data[target_col].values
        else:
            self.targets = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {}
        
        if self.feature_cols:
            features = self.data.loc[idx, self.feature_cols].values.astype(np.float32)
            sample['data'] = torch.FloatTensor(features)
        
        if self.image_col and self.image_col in self.data.columns:
            image_path = self.data.loc[idx, self.image_col]
            if isinstance(image_path, str) and Path(image_path).exists():
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)
                if self.transform:
                    image = self.transform(image)
                else:
                    image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
                sample['pixel_values'] = image
            else:
                sample['pixel_values'] = torch.zeros(3, 224, 224)
        
        if self.targets is not None:
            sample['target'] = torch.LongTensor([self.targets[idx]])[0]
        
        sample['transaction_id'] = str(self.data.loc[idx, 'transaction_id']) if 'transaction_id' in self.data.columns else str(idx)
        
        return sample


class GraphFraudDataset(Dataset):
    def __init__(self, graphs: List[Dict], targets: Optional[List[int]] = None):
        self.graphs = graphs
        self.targets = targets if targets is not None else [0] * len(graphs)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        sample = {
            'x': graph['x'],
            'edge_index': graph['edge_index'],
            'edge_attr': graph.get('edge_attr', None),
            'target': torch.LongTensor([self.targets[idx]])[0]
        }
        if 'batch' in graph:
            sample['batch'] = graph['batch']
        return sample


class DataManager:
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_cols = None
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        logger.info(f"Loading data from {data_path}")
        
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        return df
    
    def preprocess_data(self, df: pd.DataFrame, 
                       target_col: str = 'is_fraud',
                       test_size: float = 0.2,
                       val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Preprocessing data...")
        
        df = df.copy()
        
        if target_col in df.columns:
            df[target_col] = df[target_col].astype(int)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if col != 'transaction_id' and col != 'timestamp':
                df[col] = df[col].fillna('unknown')
                if col != target_col:
                    df[col] = self.label_encoder.fit_transform(df[col].astype(str))
        
        if target_col in df.columns:
            train_df, temp_df = train_test_split(
                df, test_size=test_size + val_size, 
                stratify=df[target_col], random_state=42
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=test_size / (test_size + val_size),
                stratify=temp_df[target_col], random_state=42
            )
        else:
            train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=test_size / (test_size + val_size), random_state=42)
        
        self.feature_cols = numeric_cols + [c for c in categorical_cols if c != target_col]
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_dataloaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          test_df: pd.DataFrame,
                          batch_size: int = 32,
                          num_workers: int = 4,
                          image_col: Optional[str] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        logger.info("Creating data loaders...")
        
        train_dataset = FraudDataset(train_df, feature_cols=self.feature_cols, image_col=image_col)
        val_dataset = FraudDataset(val_df, feature_cols=self.feature_cols, image_col=image_col)
        test_dataset = FraudDataset(test_df, feature_cols=self.feature_cols, image_col=image_col)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
