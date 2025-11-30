"""
Recommender System with LightGCN Implementation
Replaces item-based CF with graph neural network collaborative filtering
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import warnings
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
from eda_analysis import perform_eda

warnings.filterwarnings('ignore')


def check_gpu_availability():
    """Check GPU availability for PyTorch"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_available = torch.cuda.is_available()
    
    device_info = {
        'pytorch': {
            'available': gpu_available,
            'device': str(device),
            'device_name': torch.cuda.get_device_name(0) if gpu_available else 'CPU'
        }
    }
    
    return gpu_available, device_info


class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
    https://arxiv.org/abs/2002.02126
    """
    def __init__(self, num_users, num_items, embedding_dim, num_layers, adj_matrix):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Adjacency matrix (sparse)
        self.adj_matrix = adj_matrix
        
    def forward(self, users, items):
        """Forward pass for training"""
        # Get final embeddings
        all_users_emb, all_items_emb = self.get_embeddings()
        
        # Get user and item embeddings
        users_emb = all_users_emb[users]
        items_emb = all_items_emb[items]
        
        # Calculate scores
        scores = torch.sum(users_emb * items_emb, dim=1)
        return scores
    
    def get_embeddings(self):
        """
        Propagate embeddings through graph layers
        Returns final user and item embeddings
        """
        # Initial embeddings
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        
        # List to store embeddings at each layer
        embs = [all_emb]
        
        # Graph convolution layers
        for layer in range(self.num_layers):
            all_emb = torch.sparse.mm(self.adj_matrix, all_emb)
            embs.append(all_emb)
        
        # Average over all layers (including layer 0)
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        
        # Split back to users and items
        users_emb_final, items_emb_final = torch.split(final_emb, [self.num_users, self.num_items])
        
        return users_emb_final, items_emb_final


class BPRDataset(Dataset):
    """Dataset for Bayesian Personalized Ranking (BPR) loss"""
    def __init__(self, user_items, num_items):
        self.user_items = user_items
        self.num_items = num_items
        self.users = list(user_items.keys())
        
    def __len__(self):
        return len(self.users) * 10  # Multiple samples per user
    
    def __getitem__(self, idx):
        # Random user
        user = self.users[idx % len(self.users)]
        
        # Positive item (item user interacted with)
        pos_items = list(self.user_items[user])
        pos_item = random.choice(pos_items)
        
        # Negative item (item user didn't interact with)
        neg_item = random.randint(0, self.num_items - 1)
        while neg_item in self.user_items[user]:
            neg_item = random.randint(0, self.num_items - 1)
        
        return user, pos_item, neg_item


class RecommenderSystem:
    def __init__(self):
        self.user_items = defaultdict(set)  # user_id -> set of item_ids (training data)
        self.test_user_items = defaultdict(set)  # user_id -> set of item_ids (test data)
        self.item_users = defaultdict(set)  # item_id -> set of user_ids
        self.user_item_matrix = None
        self.user_ids = []
        self.item_ids = []
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        self.item_popularity = {}
        
        # LightGCN model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_raw_data(self, filename):
        """Load full user-item interactions without splitting."""
        self.user_items.clear()
        self.test_user_items.clear()
        self.item_users.clear()

        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                user_id = int(parts[0])
                items = [int(x) for x in parts[1:]]
                self.user_items[user_id] = set(items)
                for it in items:
                    self.item_users[it].add(user_id)

        print(f"Loaded {len(self.user_items)} users and {len(self.item_users)} items")
    
    def perform_eda(self):
        """Perform exploratory data analysis"""
        perform_eda(self.user_items, self.item_users)
    
    def split_train_test(self, test_ratio=0.2, random_seed=42):
        """Create train/test split from current user_items."""
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.test_user_items.clear()
        new_train = {}

        for user_id, items in self.user_items.items():
            items = list(items)
            if len(items) > 1:
                num_test = max(1, int(len(items) * test_ratio))
                test_items = set(random.sample(items, num_test))
                train_items = set(items) - test_items
                new_train[user_id] = train_items
                self.test_user_items[user_id] = test_items
            else:
                new_train[user_id] = set(items)

        self.user_items = new_train

        # rebuild item_users from train
        self.item_users.clear()
        for u, items in self.user_items.items():
            for it in items:
                self.item_users[it].add(u)

        total_train = sum(len(v) for v in self.user_items.values())
        total_test = sum(len(v) for v in self.test_user_items.values())
        print(f"Train interactions: {total_train}, Test interactions: {total_test}")
    
    def build_user_item_matrix(self):
        """Build user-item interaction matrix and create mappings"""
        print("Building user-item matrix for LightGCN...")
        
        # Create mappings
        self.user_ids = sorted(self.user_items.keys())
        self.item_ids = sorted(self.item_users.keys())
        
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        self.idx_to_user = {idx: user_id for user_id, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item_id for item_id, idx in self.item_to_idx.items()}
        
        # Build sparse matrix
        rows, cols, data = [], [], []
        for user_id, items in self.user_items.items():
            user_idx = self.user_to_idx[user_id]
            for item_id in items:
                if item_id in self.item_to_idx:
                    item_idx = self.item_to_idx[item_id]
                    rows.append(user_idx)
                    cols.append(item_idx)
                    data.append(1.0)
        
        self.user_item_matrix = csr_matrix((data, (rows, cols)), 
                                          shape=(len(self.user_ids), len(self.item_ids)))
        
        # Compute item popularity
        for item_id, users in self.item_users.items():
            self.item_popularity[item_id] = len(users)
        
        print(f"Matrix shape: {self.user_item_matrix.shape}")
        print(f"Number of interactions: {self.user_item_matrix.nnz}")
    
    def _create_adjacency_matrix(self):
        """
        Create normalized adjacency matrix for LightGCN
        A = D^(-1/2) * (R || R^T) * D^(-1/2)
        """
        num_users = len(self.user_ids)
        num_items = len(self.item_ids)
        
        # Create user-item interaction indices
        user_indices = []
        item_indices = []
        
        for user_id, items in self.user_items.items():
            user_idx = self.user_to_idx[user_id]
            for item_id in items:
                if item_id in self.item_to_idx:
                    item_idx = self.item_to_idx[item_id]
                    user_indices.append(user_idx)
                    item_indices.append(item_idx)
        
        # Create bipartite graph adjacency matrix
        # Format: [Users | Items]
        #         [Items | Users]
        
        # Edges from users to items
        row_indices_ui = user_indices
        col_indices_ui = [idx + num_users for idx in item_indices]
        
        # Edges from items to users (symmetric)
        row_indices_iu = [idx + num_users for idx in item_indices]
        col_indices_iu = user_indices
        
        # Combine all edges
        row_indices = row_indices_ui + row_indices_iu
        col_indices = col_indices_ui + col_indices_iu
        values = [1.0] * len(row_indices)
        
        # Create sparse adjacency matrix
        adj_mat = torch.sparse_coo_tensor(
            torch.LongTensor([row_indices, col_indices]),
            torch.FloatTensor(values),
            (num_users + num_items, num_users + num_items)
        )
        
        # Normalize: D^(-1/2) * A * D^(-1/2)
        # Calculate degree for each node
        degrees = torch.sparse.sum(adj_mat, dim=1).to_dense()
        degrees = torch.pow(degrees, -0.5)
        degrees[torch.isinf(degrees)] = 0
        
        # Create diagonal degree matrix
        degree_indices = torch.arange(num_users + num_items).repeat(2, 1)
        degree_mat = torch.sparse_coo_tensor(
            degree_indices,
            degrees,
            (num_users + num_items, num_users + num_items)
        )
        
        # Normalize adjacency matrix
        adj_mat = torch.sparse.mm(degree_mat, adj_mat)
        adj_mat = torch.sparse.mm(adj_mat, degree_mat)
        
        return adj_mat.to(self.device)
    
    def train_lightgcn(self, embedding_dim=64, num_layers=3, reg=0.0001, 
                       lr=0.001, epochs=20, batch_size=4096):
        """
        Train LightGCN model
        
        Args:
            embedding_dim: Dimension of embeddings
            num_layers: Number of graph convolution layers
            reg: L2 regularization weight
            lr: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print("\n" + "="*60)
        print("TRAINING LIGHTGCN")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Number of layers: {num_layers}")
        print(f"Learning rate: {lr}")
        print(f"Regularization: {reg}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print("="*60)
        
        # Create adjacency matrix
        print("\nCreating normalized adjacency matrix...")
        adj_matrix = self._create_adjacency_matrix()
        
        # Initialize model
        num_users = len(self.user_ids)
        num_items = len(self.item_ids)
        
        self.model = LightGCN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            adj_matrix=adj_matrix
        ).to(self.device)
        
        # Create dataset with mapped indices
        user_items_mapped = {}
        for user_id, items in self.user_items.items():
            user_idx = self.user_to_idx[user_id]
            item_indices = [self.item_to_idx[item_id] for item_id in items if item_id in self.item_to_idx]
            user_items_mapped[user_idx] = set(item_indices)
        
        dataset = BPRDataset(user_items_mapped, num_items)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            for batch_users, batch_pos_items, batch_neg_items in dataloader:
                batch_users = batch_users.to(self.device)
                batch_pos_items = batch_pos_items.to(self.device)
                batch_neg_items = batch_neg_items.to(self.device)
                
                # Forward pass
                pos_scores = self.model(batch_users, batch_pos_items)
                neg_scores = self.model(batch_users, batch_neg_items)
                
                # BPR loss
                bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
                
                # L2 regularization
                reg_loss = reg * (
                    torch.norm(self.model.user_embedding.weight) + 
                    torch.norm(self.model.item_embedding.weight)
                )
                
                loss = bpr_loss + reg_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        print("\n✓ Training completed!")
        self.model.eval()
    
    def recommend_items_lightgcn(self, user_id, n_recommendations=20):
        """Generate recommendations using trained LightGCN model"""
        if self.model is None:
            raise ValueError("Model not trained! Call train_lightgcn() first.")
        
        # Handle new users
        if user_id not in self.user_to_idx:
            # Return popular items for new users
            popular_items = sorted(self.item_popularity.items(), key=lambda x: x[1], reverse=True)
            return [item_id for item_id, _ in popular_items[:n_recommendations]]
        
        user_idx = self.user_to_idx[user_id]
        user_items = self.user_items.get(user_id, set())
        
        # Get embeddings
        with torch.no_grad():
            all_users_emb, all_items_emb = self.model.get_embeddings()
            user_emb = all_users_emb[user_idx].unsqueeze(0)  # (1, embedding_dim)
            
            # Calculate scores for all items
            scores = torch.matmul(user_emb, all_items_emb.T).squeeze(0)  # (num_items,)
            scores = scores.cpu().numpy()
        
        # Mask already interacted items
        for item_id in user_items:
            if item_id in self.item_to_idx:
                item_idx = self.item_to_idx[item_id]
                scores[item_idx] = -np.inf
        
        # Get top-k items
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [self.idx_to_item[idx] for idx in top_indices]
        
        return recommendations
    
    # ========== EVALUATION METHODS ==========
    
    def precision_at_k(self, recommendations, test_items, k):
        """Calculate Precision@K"""
        if k == 0:
            return 0.0
        top_k = set(recommendations[:k])
        if len(top_k) == 0:
            return 0.0
        return len(top_k.intersection(test_items)) / len(top_k)
    
    def recall_at_k(self, recommendations, test_items, k):
        """Calculate Recall@K"""
        if len(test_items) == 0:
            return 0.0
        top_k = set(recommendations[:k])
        return len(top_k.intersection(test_items)) / len(test_items)
    
    def ndcg_at_k(self, recommendations, test_items, k):
        """Calculate Normalized Discounted Cumulative Gain (NDCG)@K"""
        if len(test_items) == 0:
            return 0.0
        
        top_k = recommendations[:k]
        if len(top_k) == 0:
            return 0.0
        
        dcg = 0.0
        for i, item in enumerate(top_k):
            if item in test_items:
                relevance = 1.0
                dcg += relevance / np.log2(i + 2)
        
        ideal_relevance = sorted([1.0 if item in test_items else 0.0 
                                  for item in top_k], reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance) if rel > 0)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate(self, k_values=[5, 10, 20], max_users=None, sample_ratio=1.0):
        """Evaluate the recommender system on test set"""
        if len(self.test_user_items) == 0:
            print("Error: No test split available. Run split_train_test() first.")
            return None
        
        print(f"\nEvaluating on test set...")
        print("="*60)
        
        test_users = [uid for uid in self.test_user_items.keys() 
                     if len(self.test_user_items[uid]) > 0 and uid in self.user_items]
        
        if sample_ratio < 1.0:
            random.seed(42)
            num_sample = int(len(test_users) * sample_ratio)
            test_users = random.sample(test_users, num_sample)
            print(f"Sampling {len(test_users)} users ({sample_ratio*100:.1f}%) for evaluation")
        
        if max_users:
            test_users = test_users[:max_users]
        
        metrics = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in k_values}
        
        print(f"Evaluating on {len(test_users)} users...")
        
        for idx, user_id in enumerate(test_users):
            test_items = self.test_user_items[user_id]
            recommendations = self.recommend_items_lightgcn(user_id, n_recommendations=max(k_values))
            
            for k in k_values:
                precision = self.precision_at_k(recommendations, test_items, k)
                recall = self.recall_at_k(recommendations, test_items, k)
                ndcg = self.ndcg_at_k(recommendations, test_items, k)
                
                metrics[k]['precision'].append(precision)
                metrics[k]['recall'].append(recall)
                metrics[k]['ndcg'].append(ndcg)
            
            if (idx + 1) % 1000 == 0:
                print(f"  Evaluated {idx + 1}/{len(test_users)} users...")
        
        print("\nEvaluation Results:")
        print("="*60)
        for k in k_values:
            avg_precision = np.mean(metrics[k]['precision'])
            avg_recall = np.mean(metrics[k]['recall'])
            avg_ndcg = np.mean(metrics[k]['ndcg'])
            
            print(f"K={k}:")
            print(f"  Precision@{k}: {avg_precision:.4f}")
            print(f"  Recall@{k}: {avg_recall:.4f}")
            print(f"  NDCG@{k}: {avg_ndcg:.4f}")
        
        print("="*60)
        
        return metrics

    def generate_all_recommendations(self, output_file='recommendations.txt', n_recommendations=20):
        """Generate recommendations for all users (for submission)"""
        print(f"\nGenerating {n_recommendations} recommendations for all users...")
        print("  Using LightGCN Model")
        
        all_user_ids = sorted(self.user_items.keys())
        print(f"  Processing {len(all_user_ids)} users...")
        
        # Get popular items for fallback
        popular_items = sorted(self.item_popularity.items(), key=lambda x: x[1], reverse=True)
        popular_items = [item_id for item_id, _ in popular_items[:1000]]
        
        with open(output_file, 'w') as f:
            for idx, user_id in enumerate(all_user_ids):
                recommendations = self.recommend_items_lightgcn(user_id, n_recommendations)
                
                user_items = self.user_items.get(user_id, set())
                
                # Ensure exactly n_recommendations items
                if len(recommendations) < n_recommendations:
                    for item in popular_items:
                        if item not in user_items and item not in recommendations:
                            recommendations.append(item)
                            if len(recommendations) >= n_recommendations:
                                break
                
                # Final fallback
                if len(recommendations) < n_recommendations:
                    for item_id in sorted(self.item_ids):
                        if item_id not in user_items and item_id not in recommendations:
                            recommendations.append(item_id)
                            if len(recommendations) >= n_recommendations:
                                break
                
                recommendations = recommendations[:n_recommendations]
                assert len(recommendations) == n_recommendations, \
                    f"User {user_id}: Only {len(recommendations)} recommendations"
                
                line = str(user_id) + ' ' + ' '.join(map(str, recommendations)) + '\n'
                f.write(line)
                
                if (idx + 1) % 5000 == 0:
                    print(f"  Processed {idx + 1}/{len(all_user_ids)} users "
                          f"({100*(idx+1)/len(all_user_ids):.1f}%)...")
        
        print(f"\n✓ Recommendations saved to '{output_file}'")
        print(f"  Format: user_id item1 item2 ... item{n_recommendations}")
        print(f"  Total users: {len(all_user_ids)}")


def main():
    import sys
    
    # Check GPU availability
    print("="*60)
    print("GPU DETECTION")
    print("="*60)
    gpu_available, device_info = check_gpu_availability()
    if gpu_available:
        print("✓ GPU detected and will be used for training")
        print(f"  Device: {device_info['pytorch']['device_name']}")
    else:
        print("✗ No GPU detected - will use CPU (training will be slower)")
    print("="*60)
    print()
    
    recommender = RecommenderSystem()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--evaluate':
        # EVALUATION MODE
        print("\n" + "="*60)
        print("EVALUATION MODE")
        print("="*60)
        
        recommender.load_raw_data("train-2.txt")
        recommender.perform_eda()
        recommender.split_train_test(test_ratio=0.2, random_seed=42)
        recommender.build_user_item_matrix()
        
        # Train LightGCN
        recommender.train_lightgcn(
            embedding_dim=64,
            num_layers=3,
            reg=0.0001,
            lr=0.005,
            epochs=20,
            batch_size=4096
        )
        
        # Evaluate
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        recommender.evaluate(k_values=[5, 10, 20], sample_ratio=1.0)
        
    else:
        # SUBMISSION MODE
        print("\n" + "="*60)
        print("SUBMISSION MODE")
        print("="*60)
        
        recommender.load_raw_data("train-2.txt")
        recommender.perform_eda()
        recommender.build_user_item_matrix()
        
        # Train LightGCN on full data
        recommender.train_lightgcn(
            embedding_dim=64,
            num_layers=3,
            reg=0.0001,
            lr=0.005,
            epochs=20,
            batch_size=4096
        )
        
        # Generate recommendations
        print("\n" + "="*60)
        print("GENERATING RECOMMENDATIONS")
        print("="*60)
        recommender.generate_all_recommendations(
            output_file='recommendations.txt',
            n_recommendations=20
        )

if __name__ == "__main__":
    main()