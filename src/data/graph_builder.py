import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data, Batch
import networkx as nx
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class TransactionGraphBuilder:
    def __init__(self, config: Dict):
        self.config = config
        self.node_features = []
        self.edge_indices = []
        self.edge_weights = []
        self.node_mapping = {}
        self.reverse_mapping = {}
    
    def build_graph(self, transactions: pd.DataFrame) -> Data:
        logger.info("Building transaction graph...")
        
        self._create_nodes(transactions)
        self._create_edges(transactions)
        self._create_features(transactions)
        
        edge_index = torch.tensor(self.edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(self.edge_weights, dtype=torch.float)
        x = torch.tensor(self.node_features, dtype=torch.float)
        
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        logger.info(f"Graph built: {graph.num_nodes} nodes, {graph.num_edges} edges")
        return graph
    
    def _create_nodes(self, transactions: pd.DataFrame):
        unique_entities = set()
        
        if 'account_id' in transactions.columns:
            unique_entities.update(transactions['account_id'].unique())
        if 'merchant_id' in transactions.columns:
            unique_entities.update(transactions['merchant_id'].unique())
        if 'card_id' in transactions.columns:
            unique_entities.update(transactions['card_id'].unique())
        
        self.node_mapping = {entity: idx for idx, entity in enumerate(unique_entities)}
        self.reverse_mapping = {idx: entity for entity, idx in self.node_mapping.items()}
        
        logger.info(f"Created {len(self.node_mapping)} nodes")
    
    def _create_edges(self, transactions: pd.DataFrame):
        edge_types = []
        
        if 'account_id' in transactions.columns and 'merchant_id' in transactions.columns:
            for _, row in transactions.iterrows():
                account = self.node_mapping.get(row['account_id'])
                merchant = self.node_mapping.get(row['merchant_id'])
                
                if account is not None and merchant is not None:
                    self.edge_indices.append([account, merchant])
                    weight = row.get('amount', 1.0) / (row.get('amount', 1.0).max() + 1e-8) if 'amount' in transactions.columns else 1.0
                    self.edge_weights.append(weight)
                    edge_types.append('account_merchant')
        
        if 'account_id' in transactions.columns and 'card_id' in transactions.columns:
            for _, row in transactions.iterrows():
                account = self.node_mapping.get(row['account_id'])
                card = self.node_mapping.get(row['card_id'])
                
                if account is not None and card is not None:
                    self.edge_indices.append([account, card])
                    weight = 1.0
                    self.edge_weights.append(weight)
                    edge_types.append('account_card')
        
        logger.info(f"Created {len(self.edge_indices)} edges")
    
    def _create_features(self, transactions: pd.DataFrame):
        feature_dim = 10
        
        for entity_id in range(len(self.node_mapping)):
            entity = self.reverse_mapping[entity_id]
            
            features = np.zeros(feature_dim)
            
            entity_transactions = transactions[
                (transactions.get('account_id', pd.Series()) == entity) |
                (transactions.get('merchant_id', pd.Series()) == entity) |
                (transactions.get('card_id', pd.Series()) == entity)
            ]
            
            if len(entity_transactions) > 0:
                features[0] = len(entity_transactions)
                features[1] = entity_transactions.get('amount', pd.Series([0])).mean()
                features[2] = entity_transactions.get('amount', pd.Series([0])).std()
                features[3] = entity_transactions.get('is_fraud', pd.Series([0])).sum()
                features[4] = entity_transactions.get('is_fraud', pd.Series([0])).mean()
                features[5] = len(entity_transactions.get('merchant_id', pd.Series()).unique())
                features[6] = len(entity_transactions.get('location', pd.Series()).unique())
                features[7] = (entity_transactions.get('hour', pd.Series([12])) < 6).sum()
                features[8] = entity_transactions.get('hour', pd.Series([12])).mean()
                features[9] = entity_transactions.get('hour', pd.Series([12])).std()
            
            self.node_features.append(features)
        
        scaler = StandardScaler()
        self.node_features = scaler.fit_transform(self.node_features)
        
        logger.info(f"Created features: {len(self.node_features)} nodes, {len(self.node_features[0])} features per node")


class TemporalGraphBuilder:
    def __init__(self, config: Dict):
        self.config = config
        self.graphs = []
        self.temporal_edges = []
    
    def build_temporal_graphs(self, transactions: pd.DataFrame, 
                             time_window: str = '1H') -> List[Data]:
        logger.info("Building temporal transaction graphs...")
        
        transactions['timestamp'] = pd.to_datetime(transactions.get('timestamp', transactions.index))
        transactions = transactions.sort_values('timestamp')
        
        time_windows = pd.Grouper(key='timestamp', freq=time_window)
        
        for time_group, group_df in transactions.groupby(time_windows):
            builder = TransactionGraphBuilder(self.config)
            graph = builder.build_graph(group_df)
            graph.timestamp = time_group
            self.graphs.append(graph)
        
        logger.info(f"Created {len(self.graphs)} temporal graphs")
        return self.graphs
    
    def create_temporal_edges(self, graphs: List[Data]) -> torch.Tensor:
        temporal_edges = []
        
        for i in range(len(graphs) - 1):
            current_graph = graphs[i]
            next_graph = graphs[i + 1]
            
            for node_idx in range(current_graph.num_nodes):
                if node_idx < next_graph.num_nodes:
                    temporal_edges.append([node_idx, node_idx])
        
        return torch.tensor(temporal_edges, dtype=torch.long).t().contiguous() if temporal_edges else None


class HeterogeneousGraphBuilder:
    def __init__(self, config: Dict):
        self.config = config
        self.node_types = ['account', 'merchant', 'card', 'location']
        self.edge_types = [
            ('account', 'transacts_with', 'merchant'),
            ('account', 'uses', 'card'),
            ('merchant', 'located_in', 'location')
        ]
    
    def build_heterogeneous_graph(self, transactions: pd.DataFrame) -> Dict:
        logger.info("Building heterogeneous transaction graph...")
        
        node_dict = {}
        edge_dict = {}
        
        for node_type in self.node_types:
            if node_type == 'account':
                nodes = transactions['account_id'].unique()
            elif node_type == 'merchant':
                nodes = transactions['merchant_id'].unique()
            elif node_type == 'card':
                nodes = transactions['card_id'].unique()
            else:
                nodes = transactions['location'].unique()
            
            node_dict[node_type] = {node: idx for idx, node in enumerate(nodes)}
        
        for edge_type in self.edge_types:
            src_type, relation, dst_type = edge_type
            edges = []
            
            if relation == 'transacts_with':
                for _, row in transactions.iterrows():
                    src = node_dict[src_type].get(row['account_id'])
                    dst = node_dict[dst_type].get(row['merchant_id'])
                    if src is not None and dst is not None:
                        edges.append([src, dst])
            
            elif relation == 'uses':
                for _, row in transactions.iterrows():
                    src = node_dict[src_type].get(row['account_id'])
                    dst = node_dict[dst_type].get(row['card_id'])
                    if src is not None and dst is not None:
                        edges.append([src, dst])
            
            elif relation == 'located_in':
                for _, row in transactions.iterrows():
                    src = node_dict[src_type].get(row['merchant_id'])
                    dst = node_dict[dst_type].get(row['location'])
                    if src is not None and dst is not None:
                        edges.append([src, dst])
            
            if edges:
                edge_dict[edge_type] = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return {
            'node_dict': node_dict,
            'edge_dict': edge_dict
        }
