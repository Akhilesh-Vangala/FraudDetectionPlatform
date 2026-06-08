import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, FastICA
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class AdvancedDataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.scalers = {}
        self.feature_selectors = {}
        self.pca_models = {}
    
    def process_transaction_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.info("Processing transaction data...")
        
        feature_cols = [col for col in df.columns if col not in ['transaction_id', 'is_fraud', 'timestamp']]
        
        X = df[feature_cols].values
        y = df['is_fraud'].values if 'is_fraud' in df.columns else None
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        if 'feature_selection' in self.config and self.config['feature_selection'].get('enabled', False):
            selector = SelectKBest(
                score_func=f_classif if y is not None else mutual_info_classif,
                k=self.config['feature_selection'].get('k', 20)
            )
            X_scaled = selector.fit_transform(X_scaled, y)
            self.feature_selectors['transaction'] = selector
        
        if 'pca' in self.config and self.config['pca'].get('enabled', False):
            pca = PCA(n_components=self.config['pca'].get('n_components', 50))
            X_scaled = pca.fit_transform(X_scaled)
            self.pca_models['transaction'] = pca
        
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.LongTensor(y) if y is not None else None
        
        logger.info(f"Processed data: {X_tensor.shape}")
        return X_tensor, y_tensor
    
    def process_image_data(self, image_paths: List[str], target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        logger.info(f"Processing {len(image_paths)} images...")
        
        images = []
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                image = image.resize(target_size)
                image_array = np.array(image) / 255.0
                images.append(image_array)
            except Exception as e:
                logger.warning(f"Error loading image {path}: {e}")
                images.append(np.zeros((*target_size, 3)))
        
        images_tensor = torch.FloatTensor(np.array(images)).permute(0, 3, 1, 2)
        
        logger.info(f"Processed images: {images_tensor.shape}")
        return images_tensor
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating temporal features...")
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating interaction features...")
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['is_fraud', 'transaction_id']]
        
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols[:5]):
                for col2 in numeric_cols[i+1:6]:
                    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
        logger.info("Creating statistical features...")
        df = df.copy()
        
        if 'account_id' in df.columns:
            for col in ['amount', 'transaction_count']:
                if col in df.columns:
                    df[f'{col}_rolling_mean'] = df.groupby('account_id')[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    df[f'{col}_rolling_std'] = df.groupby('account_id')[col].transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    )
                    df[f'{col}_rolling_max'] = df.groupby('account_id')[col].transform(
                        lambda x: x.rolling(window, min_periods=1).max()
                    )
        
        return df
