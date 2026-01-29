import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer,
    PowerTransformer, KBinsDiscretizer
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, chi2,
    RFE, SelectFromModel
)
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, FactorAnalysis
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.ensemble import RandomForestClassifier
import logging
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineering:
    def __init__(self, config: Dict):
        self.config = config
        self.scalers = {}
        self.transformers = {}
        self.feature_selectors = {}
        self.dimensionality_reducers = {}
    
    def create_statistical_features(self, df: pd.DataFrame, 
                                   group_cols: List[str],
                                   value_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        
        for group_col in group_cols:
            for value_col in value_cols:
                if group_col in df.columns and value_col in df.columns:
                    grouped = df.groupby(group_col)[value_col]
                    
                    df[f'{value_col}_{group_col}_mean'] = grouped.transform('mean')
                    df[f'{value_col}_{group_col}_std'] = grouped.transform('std')
                    df[f'{value_col}_{group_col}_min'] = grouped.transform('min')
                    df[f'{value_col}_{group_col}_max'] = grouped.transform('max')
                    df[f'{value_col}_{group_col}_median'] = grouped.transform('median')
                    df[f'{value_col}_{group_col}_skew'] = grouped.transform('skew')
                    df[f'{value_col}_{group_col}_kurt'] = grouped.transform(lambda x: x.kurtosis())
                    
                    df[f'{value_col}_{group_col}_q25'] = grouped.transform(lambda x: x.quantile(0.25))
                    df[f'{value_col}_{group_col}_q75'] = grouped.transform(lambda x: x.quantile(0.75))
                    df[f'{value_col}_{group_col}_iqr'] = (
                        df[f'{value_col}_{group_col}_q75'] - df[f'{value_col}_{group_col}_q25']
                    )
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame, 
                                time_col: str = 'timestamp') -> pd.DataFrame:
        df = df.copy()
        
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            
            df['hour'] = df[time_col].dt.hour
            df['day_of_week'] = df[time_col].dt.dayofweek
            df['day_of_month'] = df[time_col].dt.day
            df['month'] = df[time_col].dt.month
            df['quarter'] = df[time_col].dt.quarter
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_month_start'] = df[time_col].dt.is_month_start.astype(int)
            df['is_month_end'] = df[time_col].dt.is_month_end.astype(int)
            
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame,
                                   numeric_cols: List[str],
                                   max_interactions: int = 10) -> pd.DataFrame:
        df = df.copy()
        
        interaction_count = 0
        for i, col1 in enumerate(numeric_cols):
            if interaction_count >= max_interactions:
                break
            for col2 in numeric_cols[i+1:]:
                if interaction_count >= max_interactions:
                    break
                if col1 in df.columns and col2 in df.columns:
                    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                    df[f'{col1}_add_{col2}'] = df[col1] + df[col2]
                    df[f'{col1}_sub_{col2}'] = df[col1] - df[col2]
                    interaction_count += 4
        
        return df
    
    def create_polynomial_features(self, df: pd.DataFrame,
                                  numeric_cols: List[str],
                                  degree: int = 2) -> pd.DataFrame:
        df = df.copy()
        
        for col in numeric_cols:
            if col in df.columns:
                for d in range(2, degree + 1):
                    df[f'{col}_pow_{d}'] = df[col] ** d
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                               group_col: str,
                               value_col: str,
                               windows: List[int] = [3, 7, 14, 30]) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values([group_col, 'timestamp'] if 'timestamp' in df.columns else group_col)
        
        for window in windows:
            if group_col in df.columns and value_col in df.columns:
                grouped = df.groupby(group_col)[value_col]
                
                df[f'{value_col}_rolling_mean_{window}'] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f'{value_col}_rolling_std_{window}'] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
                df[f'{value_col}_rolling_min_{window}'] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).min()
                )
                df[f'{value_col}_rolling_max_{window}'] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).max()
                )
                df[f'{value_col}_rolling_median_{window}'] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).median()
                )
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame,
                           group_col: str,
                           value_cols: List[str],
                           lags: List[int] = [1, 2, 3, 7, 14]) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values([group_col, 'timestamp'] if 'timestamp' in df.columns else group_col)
        
        for value_col in value_cols:
            if group_col in df.columns and value_col in df.columns:
                grouped = df.groupby(group_col)[value_col]
                for lag in lags:
                    df[f'{value_col}_lag_{lag}'] = grouped.shift(lag)
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame,
                       numeric_cols: List[str],
                       method: str = 'iqr') -> pd.DataFrame:
        df = df.copy()
        
        for col in numeric_cols:
            if col in df.columns:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[f'{col}_is_outlier'] = (
                        (df[col] < lower_bound) | (df[col] > upper_bound)
                    ).astype(int)
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(df[col].fillna(0)))
                    df[f'{col}_is_outlier'] = (z_scores > 3).astype(int)
                elif method == 'isolation':
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers = iso_forest.fit_predict(df[[col]].fillna(0))
                    df[f'{col}_is_outlier'] = (outliers == -1).astype(int)
        
        return df
    
    def apply_robust_scaling(self, df: pd.DataFrame,
                            numeric_cols: List[str],
                            scaler_type: str = 'robust') -> Tuple[pd.DataFrame, Dict]:
        df = df.copy()
        scalers = {}
        
        for col in numeric_cols:
            if col in df.columns:
                if scaler_type == 'robust':
                    scaler = RobustScaler()
                elif scaler_type == 'standard':
                    scaler = StandardScaler()
                elif scaler_type == 'minmax':
                    scaler = MinMaxScaler()
                elif scaler_type == 'quantile':
                    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
                elif scaler_type == 'power':
                    scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                else:
                    scaler = RobustScaler()
                
                df[col] = scaler.fit_transform(df[[col]].fillna(0))
                scalers[col] = scaler
        
        return df, scalers
    
    def apply_feature_selection(self, X: np.ndarray, y: np.ndarray,
                               method: str = 'mutual_info',
                               k: int = 50) -> Tuple[np.ndarray, object]:
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'chi2':
            selector = SelectKBest(score_func=chi2, k=k)
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            selector = RFE(estimator, n_features_to_select=k)
        elif method == 'select_from_model':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            selector = SelectFromModel(estimator, max_features=k)
        else:
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        X_selected = selector.fit_transform(X, y)
        return X_selected, selector
    
    def apply_dimensionality_reduction(self, X: np.ndarray,
                                      method: str = 'pca',
                                      n_components: int = 50) -> Tuple[np.ndarray, object]:
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'ica':
            reducer = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
        elif method == 'factor':
            reducer = FactorAnalysis(n_components=n_components, random_state=42)
        else:
            reducer = PCA(n_components=n_components, random_state=42)
        
        X_reduced = reducer.fit_transform(X)
        return X_reduced, reducer
    
    def create_target_encoding(self, df: pd.DataFrame,
                              cat_col: str,
                              target_col: str,
                              smoothing: float = 1.0) -> pd.DataFrame:
        df = df.copy()
        
        if cat_col in df.columns and target_col in df.columns:
            global_mean = df[target_col].mean()
            n = df.groupby(cat_col).size()
            means = df.groupby(cat_col)[target_col].mean()
            
            df[f'{cat_col}_target_encoded'] = (
                (means[df[cat_col]] * n[df[cat_col]] + global_mean * smoothing) /
                (n[df[cat_col]] + smoothing)
            )
        
        return df
    
    def create_frequency_encoding(self, df: pd.DataFrame,
                                 cat_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        
        for col in cat_cols:
            if col in df.columns:
                freq = df[col].value_counts()
                df[f'{col}_freq'] = df[col].map(freq)
        
        return df
