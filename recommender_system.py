"""
Recommender System for User-Item Interactions
This script includes EDA and implements a collaborative filtering recommender system.
"""

import numpy as np
import random
import warnings
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix, diags
from eda_analysis import perform_eda

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

warnings.filterwarnings('ignore')


def check_gpu_availability():
    """Check GPU availability and return device info"""
    gpu_available = False
    device_info = {}
    
    # Check PyTorch GPU
    if torch is not None:
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            device_info['pytorch'] = {
                'device': torch.cuda.get_device_name(0),
                'memory': f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            }
    
    # Check cuPy (for cosine similarity)
    try:
        import cupy as cp
        device_info['cupy'] = {'available': True}
        gpu_available = True
    except ImportError:
        device_info['cupy'] = {'available': False}
    
    return gpu_available, device_info


class RecommenderSystem:
    def __init__(self):
        self.user_items = defaultdict(set)  # user_id -> set of item_ids (training data)
        self.test_user_items = defaultdict(set)  # user_id -> set of item_ids (test data)
        self.item_users = defaultdict(set)  # item_id -> set of user_ids
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.user_ids = []
        self.item_ids = []
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.use_test_split = False  # Flag for using train/test split
        self.item_popularity = {}  # Track item popularity
        self.popular_items_cache = None  # Cache popular items
        self.reverse_similarity_index = None  # Reverse index for similarity
        self.neucf_model = None  # NeuCF neural model
        self.neucf_device = None  # Device for NeuCF model
        self.use_neucf = False  # Flag for using NeuCF
        self.gmf_model = None  # GMF neural model
        self.gmf_device = None  # Device for GMF model
        self.use_gmf = False  # Flag for using GMF
        self.cosine_similarity_matrix = None  # Store cosine similarity for ensemble
        self.use_ensemble = False  # Flag for using ensemble
        
    def load_data(self, filename, test_ratio=0.0, random_seed=42):
        """
        Load data from train-2.txt file
        If test_ratio > 0, splits data into train/test sets
        """
        if test_ratio > 0:
            self.use_test_split = True
            print(f"Loading data and splitting into train/test (test_ratio={test_ratio})...")
            random.seed(random_seed)
            np.random.seed(random_seed)
        else:
            print("Loading data...")
        
        all_user_items = defaultdict(list)
        
        # First, load all data
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                user_id = int(parts[0])
                items = [int(item) for item in parts[1:]]
                all_user_items[user_id] = items
        
        # Split into train/test if requested
        for user_id, items in all_user_items.items():
            if test_ratio > 0 and len(items) > 1:
                # Split items for this user
                num_test = max(1, int(len(items) * test_ratio))
                test_items = set(random.sample(items, num_test))
                train_items = set(items) - test_items
                
                self.user_items[user_id] = train_items
                self.test_user_items[user_id] = test_items
                
                for item in train_items:
                    self.item_users[item].add(user_id)
            else:
                # No split - all items go to training
                self.user_items[user_id] = set(items)
                for item in items:
                    self.item_users[item].add(user_id)
        
        print(f"Loaded {len(self.user_items)} users and {len(self.item_users)} items")
        if test_ratio > 0:
            total_train = sum(len(items) for items in self.user_items.values())
            total_test = sum(len(items) for items in self.test_user_items.values())
            print(f"Train interactions: {total_train}, Test interactions: {total_test}")
    
    def perform_eda(self):
        """Perform exploratory data analysis"""
        perform_eda(self.user_items, self.item_users)
    
    def build_user_item_matrix(self, normalize=False):
        """
        Build user-item interaction matrix
        Args:
            normalize: If True, applies TF-IDF-like normalization to reduce power user bias
                      (weight = 1/sqrt(user_item_count))
        """
        print("Building user-item matrix...")
        
        self.user_ids = sorted(self.user_items.keys())
        self.item_ids = sorted(self.item_users.keys())
        
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        self.idx_to_item = {idx: item_id for item_id, idx in self.item_to_idx.items()}
        
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
        
        if normalize:
                    # TF-IDF-like normalization: 1 / sqrt(user_item_count)
                    # This gives less weight to power users
                    weight = 1.0 / np.sqrt(num_user_items)
        else:
            weight = 1.0  # Binary interaction
        data.append(weight)
        
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        print(f"Sparsity: {100 * (1 - self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])):.2f}%")
    
    def compute_item_similarity(self, top_k_similar=200, use_gpu=None, chunk_size=1000):
        """
        Compute item-item similarity using cosine similarity (OPTIMIZED)
        
        Args:
            top_k_similar: Only store top-k most similar items per item
            use_gpu: Use GPU if available (None = auto-detect, True = force GPU, False = CPU)
            chunk_size: Process items in chunks
        """
        print("Computing item-item similarity using cosine similarity...")
        print(f"  Using top-{top_k_similar} similar items per item (approximate)")
        
        # Auto-detect GPU if not specified
        if use_gpu is None:
            use_gpu = torch is not None and torch.cuda.is_available()
        
        # Adjust chunk size for GPU
        if use_gpu and chunk_size < 2000:
            chunk_size = 2000
            print(f"  Increased chunk_size to {chunk_size} for GPU")
        
        item_user_matrix = self.user_item_matrix.T
        num_items = len(self.item_ids)
        
        # Try using cuPy for GPU acceleration
        if use_gpu:
            try:
                import cupy as cp
                from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
                print("  Using GPU (cuPy) for cosine similarity computation")
                
                item_user_matrix_gpu = cupy_csr_matrix(item_user_matrix)
                norms = cp.sqrt(cp.array(item_user_matrix_gpu.multiply(item_user_matrix_gpu).sum(axis=1))).flatten()
                norms[norms == 0] = 1
                
                similarity_dict = {}
                for i in range(0, num_items, chunk_size):
                    end_i = min(i + chunk_size, num_items)
                    chunk_matrix = item_user_matrix_gpu[i:end_i]
                    similarities = (chunk_matrix @ item_user_matrix_gpu.T).toarray()
                    similarities = cp.asnumpy(similarities)
                    
                    chunk_norms = cp.asnumpy(norms[i:end_i])
                    for local_idx, item_idx in enumerate(range(i, end_i)):
                        scores = similarities[local_idx] / (chunk_norms[local_idx] * cp.asnumpy(norms))
                        scores[item_idx] = -1
                        top_indices = np.argsort(scores)[::-1][:top_k_similar]
                        similarity_dict[item_idx] = {top_idx: float(scores[top_idx]) for top_idx in top_indices if scores[top_idx] > 0}
                    
                    if (end_i) % (chunk_size * 5) == 0:
                        print(f"  Processed {end_i}/{num_items} items...")
                
                use_gpu_success = True
            except Exception as e:
                print(f"  GPU computation failed ({e}), falling back to CPU")
                use_gpu_success = False
        else:
            use_gpu_success = False
        
        # CPU fallback
        if not use_gpu_success:
            print("  Using CPU for cosine similarity computation")
            norms = np.sqrt(np.array(item_user_matrix.multiply(item_user_matrix).sum(axis=1))).flatten()
            norms[norms == 0] = 1
            
            similarity_dict = {}
            for i in range(0, num_items, chunk_size):
                end_i = min(i + chunk_size, num_items)
                chunk_matrix = item_user_matrix[i:end_i]
                similarities = (chunk_matrix @ item_user_matrix.T).toarray()
                
                for local_idx, item_idx in enumerate(range(i, end_i)):
                    scores = similarities[local_idx] / (norms[item_idx] * norms)
                    scores[item_idx] = -1
                    top_indices = np.argsort(scores)[::-1][:top_k_similar]
                    similarity_dict[item_idx] = {top_idx: float(scores[top_idx]) for top_idx in top_indices if scores[top_idx] > 0}
                
                if (end_i) % (chunk_size * 5) == 0:
                    print(f"  Processed {end_i}/{num_items} items...")
        
        self.item_similarity_matrix = similarity_dict
        
        # Store cosine similarity for ensemble
        self.cosine_similarity_matrix = self.item_similarity_matrix.copy()
        
        print("  Building reverse similarity index...")
        self.reverse_similarity_index = defaultdict(set)
        for item_idx, similar_items_dict in self.item_similarity_matrix.items():
            for similar_item_idx in similar_items_dict.keys():
                self.reverse_similarity_index[similar_item_idx].add(item_idx)
        print("Item similarity matrix computed!")
    
    def compute_neucf_factors(self, embedding_dim=64, mlp_layers=None, 
                              epochs=20, batch_size=256, lr=0.001, num_negatives=4):
        """
        Compute NeuCF (Neural Collaborative Filtering) factors
        
        Args:
            embedding_dim: Embedding dimension (default: 64)
            mlp_layers: MLP layer sizes (default: [64, 32, 16])
            epochs: Training epochs (default: 20)
            batch_size: Batch size (default: 256)
            lr: Learning rate (default: 0.001)
            num_negatives: Number of negative samples per positive interaction
        
        Expected: NDCG@20: 35-45%
        """
        if mlp_layers is None:
            mlp_layers = [64, 32, 16]
        
        if torch is None or nn is None:
            print("ERROR: PyTorch not found!")
            print("Install with: pip install torch")
            return
        
        try:
            from torch.utils.data import Dataset, DataLoader
        except ImportError:
            print("ERROR: torch.utils.data not available!")
            print("Install PyTorch with: pip install torch")
            return
        
        print(f"Computing NeuCF factors (embedding_dim={embedding_dim}, epochs={epochs})...")
        print("  NeuCF combines GMF and MLP for non-linear patterns!")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
            if batch_size < 1024:
                batch_size = 1024
                print(f"  Increased batch_size to {batch_size} for GPU optimization")
            use_amp = True
            scaler = torch.cuda.amp.GradScaler()
            print("  Enabled mixed precision training (FP16) for faster training")
        else:
            print("  Using CPU")
            use_amp = False
        
        # NeuCF Model
        class NeuCF(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim, mlp_layers, dropout=0.2):
                super().__init__()
                self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
                self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
                self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
                self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
                
                mlp_input_dim = embedding_dim * 2
                layers = []
                for i, layer_size in enumerate(mlp_layers):
                    input_dim = mlp_input_dim if i == 0 else mlp_layers[i-1]
                    layers.append(nn.Linear(input_dim, layer_size))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                self.mlp = nn.Sequential(*layers)
                
                self.predict_layer = nn.Linear(embedding_dim + mlp_layers[-1], 1)
            
            def forward(self, user_ids, item_ids):
                user_emb_gmf = self.user_embedding_gmf(user_ids)
                item_emb_gmf = self.item_embedding_gmf(item_ids)
                gmf_vector = user_emb_gmf * item_emb_gmf
                
                user_emb_mlp = self.user_embedding_mlp(user_ids)
                item_emb_mlp = self.item_embedding_mlp(item_ids)
                mlp_vector = torch.cat([user_emb_mlp, item_emb_mlp], dim=1)
                mlp_vector = self.mlp(mlp_vector)
                
                concat_vector = torch.cat([gmf_vector, mlp_vector], dim=1)
                prediction = self.predict_layer(concat_vector)
                return prediction.squeeze(-1)
        
        class InteractionDataset(Dataset):
            def __init__(self, user_items, user_to_idx, item_to_idx, num_items, num_negatives):
                self.samples = []
                all_item_indices = np.arange(num_items)
                rng = np.random.default_rng(seed=42)
                
                for user_id, items in user_items.items():
                    if not items:
                        continue
                    user_idx = user_to_idx[user_id]
                    item_indices = [item_to_idx[item_id] for item_id in items if item_id in item_to_idx]
                    if not item_indices:
                        continue
                    positive_set = set(item_indices)
                    
                    for pos_item_idx in item_indices:
                        self.samples.append((user_idx, pos_item_idx, 1.0))
                        
                        for _ in range(num_negatives):
                            neg_item_idx = rng.choice(all_item_indices)
                            while neg_item_idx in positive_set:
                                neg_item_idx = rng.choice(all_item_indices)
                            self.samples.append((user_idx, neg_item_idx, 0.0))
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        dataset = InteractionDataset(self.user_items, self.user_to_idx, self.item_to_idx, 
                                     len(self.item_ids), num_negatives)
        
        model = NeuCF(len(self.user_ids), len(self.item_ids), embedding_dim, mlp_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
        criterion = nn.BCEWithLogitsLoss()
        
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 4 if torch.cuda.is_available() else 0,
            'pin_memory': torch.cuda.is_available()
        }
        dataloader = DataLoader(dataset, **dataloader_kwargs)
        
        print("  Training NeuCF model")
        model.train()
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                user_ids, item_ids, labels = batch
                user_ids = user_ids.to(device, non_blocking=True)
                item_ids = item_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = model(user_ids, item_ids)
                        loss = criterion(predictions, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    predictions = model(user_ids, item_ids)
                    loss = criterion(predictions, labels)
                    loss.backward()
                    optimizer.step()
                
                total_loss += float(loss.item())
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            if avg_loss < 0.0001 or patience_counter >= patience:
                if avg_loss < 0.0001:
                    print(f"    Early stopping: Loss converged to {avg_loss:.6f} at epoch {epoch+1}")
                else:
                    print(f"    Early stopping: No improvement for {patience} epochs (best loss: {best_loss:.6f})")
                break
        
        self.neucf_model = model
        self.neucf_device = device
        self.use_neucf = True
        self.use_gmf = False
        
        print("  NeuCF model trained!")
        print("  Expected NDCG@20: 35-45%")
    
    def _recommend_items_neucf(self, user_id, n_recommendations=20):
        """Generate recommendations using NeuCF"""
        if self.neucf_model is None or not self.use_neucf:
            raise ValueError("NeuCF model not trained. Call compute_neucf_factors() first.")
        if torch is None:
            raise ImportError("PyTorch not available for NeuCF recommendations.")
        
        user_idx = self.user_to_idx[user_id]
        user_items = self.user_items[user_id]
        
        self.neucf_model.eval()
        num_items = len(self.item_ids)
        with torch.no_grad():
            user_tensor = torch.full((num_items,), user_idx, dtype=torch.long, device=self.neucf_device)
            item_tensor = torch.arange(num_items, dtype=torch.long, device=self.neucf_device)
            scores = torch.sigmoid(self.neucf_model(user_tensor, item_tensor)).cpu().numpy()
        
        top_indices = np.argsort(scores)[::-1][:n_recommendations * 2]
        
        recommendations = []
        for idx in top_indices:
            item_id = self.idx_to_item[idx]
            if item_id not in user_items:
                recommendations.append(item_id)
                if len(recommendations) >= n_recommendations:
                    break
        
        if len(recommendations) < n_recommendations:
            popular_items = self._get_popular_items(1000)
            for item in popular_items:
                if item not in user_items and item not in recommendations:
                    recommendations.append(item)
                    if len(recommendations) >= n_recommendations:
                        break
        
        return recommendations[:n_recommendations]

    def compute_gmf_factors(self, embedding_dim=64, epochs=20, batch_size=256, lr=0.001, num_negatives=4):
        """
        Compute GMF (Generalized Matrix Factorization) factors
        GMF is a simpler neural approach than NeuCF
        
        Args:
            embedding_dim: Embedding dimension (default: 64)
            epochs: Training epochs (default: 20)
            batch_size: Batch size (default: 256)
            lr: Learning rate (default: 0.001)
            num_negatives: Number of negative samples per positive interaction
        
        Expected: NDCG@20: 32-38%
        """
        if torch is None or nn is None:
            print("ERROR: PyTorch not found!")
            print("Install with: pip install torch")
            return
        
        try:
            from torch.utils.data import Dataset, DataLoader
        except ImportError:
            print("ERROR: torch.utils.data not available!")
            print("Install PyTorch with: pip install torch")
            return
        
        print(f"Computing GMF factors (embedding_dim={embedding_dim}, epochs={epochs})...")
        print("  GMF is a simpler neural method than NeuCF!")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
            if batch_size < 1024:
                batch_size = 1024
                print(f"  Increased batch_size to {batch_size} for GPU optimization")
            use_amp = True
            scaler = torch.cuda.amp.GradScaler()
            print("  Enabled mixed precision training (FP16) for faster training")
        else:
            print("  Using CPU")
            use_amp = False
        
        # GMF Model
        class GMF(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim):
                super().__init__()
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                self.output_layer = nn.Linear(embedding_dim, 1)
                
                nn.init.normal_(self.user_embedding.weight, std=0.01)
                nn.init.normal_(self.item_embedding.weight, std=0.01)
            
            def forward(self, user_ids, item_ids):
                user_emb = self.user_embedding(user_ids)
                item_emb = self.item_embedding(item_ids)
                element_product = user_emb * item_emb
                output = self.output_layer(element_product)
                return output.squeeze(-1)
        
        class InteractionDataset(Dataset):
            def __init__(self, user_items, user_to_idx, item_to_idx, num_items, num_negatives):
                self.samples = []
                all_item_indices = np.arange(num_items)
                rng = np.random.default_rng(seed=42)
                
                for user_id, items in user_items.items():
                    if not items:
                        continue
                    user_idx = user_to_idx[user_id]
                    item_indices = [item_to_idx[item_id] for item_id in items if item_id in item_to_idx]
                    if not item_indices:
                        continue
                    positive_set = set(item_indices)
                    
                    for pos_item_idx in item_indices:
                        self.samples.append((user_idx, pos_item_idx, 1.0))
                        
                        for _ in range(num_negatives):
                            neg_item_idx = rng.choice(all_item_indices)
                            while neg_item_idx in positive_set:
                                neg_item_idx = rng.choice(all_item_indices)
                            self.samples.append((user_idx, neg_item_idx, 0.0))
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        dataset = InteractionDataset(self.user_items, self.user_to_idx, self.item_to_idx, 
                                     len(self.item_ids), num_negatives)
        
        model = GMF(len(self.user_ids), len(self.item_ids), embedding_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
        criterion = nn.BCEWithLogitsLoss()
        
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 4 if torch.cuda.is_available() else 0,
            'pin_memory': torch.cuda.is_available()
        }
        dataloader = DataLoader(dataset, **dataloader_kwargs)
        
        print("  Training GMF model")
        model.train()
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                user_ids, item_ids, labels = batch
                user_ids = user_ids.to(device, non_blocking=True)
                item_ids = item_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = model(user_ids, item_ids)
                        loss = criterion(predictions, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    predictions = model(user_ids, item_ids)
                    loss = criterion(predictions, labels)
                    loss.backward()
                    optimizer.step()
                
                total_loss += float(loss.item())
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            if avg_loss < 0.0001 or patience_counter >= patience:
                if avg_loss < 0.0001:
                    print(f"    Early stopping: Loss converged to {avg_loss:.6f} at epoch {epoch+1}")
                else:
                    print(f"    Early stopping: No improvement for {patience} epochs (best loss: {best_loss:.6f})")
                break
        
        self.gmf_model = model
        self.gmf_device = device
        self.use_gmf = True
        self.use_neucf = False
        
        print("  GMF model trained!")
        print("  Expected NDCG@20: 32-38%")
    
    def _recommend_items_gmf(self, user_id, n_recommendations=20):
        """Generate recommendations using GMF"""
        if self.gmf_model is None or not self.use_gmf:
            raise ValueError("GMF model not trained. Call compute_gmf_factors() first.")
        if torch is None:
            raise ImportError("PyTorch not available for GMF recommendations.")
        
        user_idx = self.user_to_idx[user_id]
        user_items = self.user_items[user_id]
        
        self.gmf_model.eval()
        num_items = len(self.item_ids)
        with torch.no_grad():
            user_tensor = torch.full((num_items,), user_idx, dtype=torch.long, device=self.gmf_device)
            item_tensor = torch.arange(num_items, dtype=torch.long, device=self.gmf_device)
            scores = self.gmf_model(user_tensor, item_tensor).cpu().numpy()
        
        top_indices = np.argsort(scores)[::-1][:n_recommendations * 2]
        
        recommendations = []
        for idx in top_indices:
            item_id = self.idx_to_item[idx]
            if item_id not in user_items:
                recommendations.append(item_id)
                if len(recommendations) >= n_recommendations:
                    break
        
        if len(recommendations) < n_recommendations:
            popular_items = self._get_popular_items(1000)
            for item in popular_items:
                if item not in user_items and item not in recommendations:
                    recommendations.append(item)
                    if len(recommendations) >= n_recommendations:
                        break
        
        return recommendations[:n_recommendations]

    def _recommend_items_ensemble(self, user_id, n_recommendations=20):
        """
        Ensemble method: Combines NeuCF, GMF, and Cosine similarity
        """
        if user_id not in self.user_items:
            popular_items = self._get_popular_items(n_recommendations)
            return popular_items[:n_recommendations]
        
        user_items = self.user_items[user_id]
        all_scores = {}
        method_count = 0
        
        # Method 1: Cosine similarity
        if self.cosine_similarity_matrix:
            cosine_recs = self._get_item_based_scores(user_id, self.cosine_similarity_matrix)
            max_cosine = max(cosine_recs.values()) if cosine_recs else 1.0
            for item_id, score in cosine_recs.items():
                normalized_score = score / max_cosine if max_cosine > 0 else 0
                all_scores[item_id] = all_scores.get(item_id, 0) + 0.25 * normalized_score
            method_count += 1
        
        # Method 2: NeuCF
        if hasattr(self, 'use_neucf') and self.use_neucf and user_id in self.user_to_idx:
            try:
                neucf_recs = self._recommend_items_neucf(user_id, n_recommendations * 3)
                for rank, item_id in enumerate(neucf_recs):
                    if item_id not in user_items:
                        score = np.exp(-rank * 0.1)
                        all_scores[item_id] = all_scores.get(item_id, 0) + 0.45 * score
                method_count += 1
            except:
                pass
        
        # Method 3: GMF
        if hasattr(self, 'use_gmf') and self.use_gmf and user_id in self.user_to_idx:
            try:
                gmf_recs = self._recommend_items_gmf(user_id, n_recommendations * 3)
                for rank, item_id in enumerate(gmf_recs):
                    if item_id not in user_items:
                        score = np.exp(-rank * 0.1)
                        all_scores[item_id] = all_scores.get(item_id, 0) + 0.30 * score
                method_count += 1
            except:
                pass
        
        # Normalize scores
        if method_count > 1:
            max_score = max(all_scores.values()) if all_scores else 1.0
            if max_score > 0:
                all_scores = {item: score / max_score for item, score in all_scores.items()}
        
        # Sort and return
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item_id for item_id, score in sorted_items[:n_recommendations]]
        
        # Fill with popular items if needed
        if len(recommendations) < n_recommendations:
            popular_items = self._get_popular_items(1000)
            for item in popular_items:
                if item not in user_items and item not in recommendations:
                    recommendations.append(item)
                    if len(recommendations) >= n_recommendations:
                        break
        
        return recommendations[:n_recommendations]
    
    def _get_item_based_scores(self, user_id, similarity_matrix):
        """Helper: Get item-based CF scores for ensemble"""
        if user_id not in self.user_items:
            return {}
        
        user_items = self.user_items[user_id]
        interacted_item_indices = [self.item_to_idx[item] for item in user_items if item in self.item_to_idx]
        
        if not interacted_item_indices:
            return {}
        
        # Get candidate items
        candidate_items = set()
        for interacted_item_idx in interacted_item_indices:
            if isinstance(similarity_matrix, dict):
                if interacted_item_idx in similarity_matrix:
                    for similar_item_idx in similarity_matrix[interacted_item_idx].keys():
                        candidate_items.add(self.idx_to_item[similar_item_idx])
        
        candidate_items -= user_items
        
        # Compute scores
        scores = {}
        for item_id in candidate_items:
            item_idx = self.item_to_idx[item_id]
            similarity_scores = []
            for interacted_item_idx in interacted_item_indices:
                if isinstance(similarity_matrix, dict):
                    if item_idx in similarity_matrix:
                        similarity = similarity_matrix[item_idx].get(interacted_item_idx, 0.0)
                    elif interacted_item_idx in similarity_matrix:
                        similarity = similarity_matrix[interacted_item_idx].get(item_idx, 0.0)
                    else:
                        similarity = 0.0
                else:
                    similarity = similarity_matrix[item_idx, interacted_item_idx]
                
                if similarity > 0:
                    similarity_scores.append(similarity)
            
            scores[item_id] = sum(similarity_scores) if similarity_scores else 0.0
        
        return scores
        
    def _get_popular_items(self, n=1000):
        """Cache popular items to avoid recomputing"""
        if self.popular_items_cache is None:
            item_counts = Counter()
            for items in self.user_items.values():
                item_counts.update(items)
            self.popular_items_cache = [item_id for item_id, _ in item_counts.most_common(n)]
        return self.popular_items_cache
    
    def recommend_items_item_based(self, user_id, n_recommendations=20, popularity_penalty=0.0, use_svd=False, use_als=False, use_ensemble=False):
        """
        Generate recommendations
        """
        # Use ensemble if requested
        if use_ensemble and self.use_ensemble:
            return self._recommend_items_ensemble(user_id, n_recommendations)
        
        # Use NeuCF if available
        if hasattr(self, 'use_neucf') and self.use_neucf and user_id in self.user_to_idx:
            return self._recommend_items_neucf(user_id, n_recommendations)
        
        # Use GMF if available
        if hasattr(self, 'use_gmf') and self.use_gmf and user_id in self.user_to_idx:
            return self._recommend_items_gmf(user_id, n_recommendations)
        
        # Fallback to item-based CF or popular items
        if self.item_similarity_matrix is None:
            if hasattr(self, 'use_gmf') and self.use_gmf and user_id in self.user_to_idx:
                return self._recommend_items_gmf(user_id, n_recommendations)
            if hasattr(self, 'use_neucf') and self.use_neucf and user_id in self.user_to_idx:
                return self._recommend_items_neucf(user_id, n_recommendations)
            popular_items = self._get_popular_items(n_recommendations)
            return popular_items[:n_recommendations]
        
        # Item-based CF using similarity matrix
        if user_id not in self.user_items:
            popular_items = self._get_popular_items(n_recommendations)
            return popular_items[:n_recommendations]
        
        user_items = self.user_items[user_id]
        interacted_item_indices = []
        for item in user_items:
            if item in self.item_to_idx:
                interacted_item_indices.append(self.item_to_idx[item])
        
        if not interacted_item_indices:
            popular_items = self._get_popular_items(n_recommendations)
            return [item for item in popular_items if item not in user_items][:n_recommendations]
        
        max_popularity = max(self.item_popularity.values()) if popularity_penalty > 0 else 1
        
        candidate_items = set()
        if isinstance(self.item_similarity_matrix, dict):
            for interacted_item_idx in interacted_item_indices:
                if interacted_item_idx in self.item_similarity_matrix:
                    similar_items_dict = self.item_similarity_matrix[interacted_item_idx]
                    for similar_item_idx in similar_items_dict.keys():
                        candidate_items.add(self.idx_to_item[similar_item_idx])
                
                if self.reverse_similarity_index and interacted_item_idx in self.reverse_similarity_index:
                    for item_idx in self.reverse_similarity_index[interacted_item_idx]:
                        candidate_items.add(self.idx_to_item[item_idx])
            
            if len(candidate_items) > 5000:
                candidate_scores = {}
                for item_id in candidate_items:
                    item_idx = self.item_to_idx[item_id]
                    score = 0
                    for interacted_item_idx in interacted_item_indices:
                        if item_idx in self.item_similarity_matrix:
                            score += self.item_similarity_matrix[item_idx].get(interacted_item_idx, 0.0)
                        elif self.reverse_similarity_index and interacted_item_idx in self.reverse_similarity_index:
                            if item_idx in self.reverse_similarity_index[interacted_item_idx]:
                                score += 0.1
                    candidate_scores[item_id] = score
                sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
                candidate_items = set([item_id for item_id, _ in sorted_candidates[:5000]])
            
            if len(candidate_items) < n_recommendations * 2:
                popular_items = self._get_popular_items(1000)
                for item in popular_items[:500]:
                    candidate_items.add(item)
        
        candidate_items -= user_items
        
        item_scores = {}
        for item_id in candidate_items:
            if item_id not in self.item_to_idx:
                continue
            item_idx = self.item_to_idx[item_id]
            
            similarity_sum = 0.0
            for interacted_item_idx in interacted_item_indices:
                if isinstance(self.item_similarity_matrix, dict):
                    if item_idx in self.item_similarity_matrix:
                        sim = self.item_similarity_matrix[item_idx].get(interacted_item_idx, 0.0)
                    elif interacted_item_idx in self.item_similarity_matrix:
                        sim = self.item_similarity_matrix[interacted_item_idx].get(item_idx, 0.0)
                    else:
                        sim = 0.0
                else:
                    sim = self.item_similarity_matrix[item_idx, interacted_item_idx]
                similarity_sum += sim
            
            if popularity_penalty > 0:
                pop_score = self.item_popularity.get(item_id, 0) / max_popularity
                similarity_sum *= (1 - popularity_penalty * pop_score)
            
            item_scores[item_id] = similarity_sum
        
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item_id for item_id, score in sorted_items if score > 0][:n_recommendations]
        
        if len(recommendations) < n_recommendations:
            popular_items = self._get_popular_items(1000)
            for item in popular_items:
                if item not in user_items and item not in recommendations:
                    recommendations.append(item)
                    if len(recommendations) >= n_recommendations:
                        break
        
        return recommendations[:n_recommendations]
    
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
    
    def mean_average_precision(self, recommendations, test_items):
        """Calculate Mean Average Precision (MAP)"""
        if len(test_items) == 0:
            return 0.0
        
        relevant_items = set(test_items)
        if len(relevant_items) == 0:
            return 0.0
        
        precision_sum = 0.0
        num_relevant_found = 0
        
        for i, item in enumerate(recommendations):
            if item in relevant_items:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / (i + 1)
                precision_sum += precision_at_i
        
        if num_relevant_found == 0:
            return 0.0
        
        return precision_sum / len(relevant_items)
    
    def evaluate(self, k_values=[5, 10, 20], max_users=None, sample_ratio=1.0):
        """Evaluate the recommender system on test set"""
        if not self.use_test_split:
            print("Error: Cannot evaluate without train/test split. Use load_data with test_ratio > 0")
            return None
        
        print(f"\nEvaluating on test set...")
        print("="*60)
        
        test_users = [uid for uid in self.test_user_items.keys() 
                     if len(self.test_user_items[uid]) > 0 and uid in self.user_items]
        
        if sample_ratio < 1.0:
            import random
            random.seed(42)
            num_sample = int(len(test_users) * sample_ratio)
            test_users = random.sample(test_users, num_sample)
            print(f"Sampling {len(test_users)} users ({sample_ratio*100:.1f}%) for evaluation")
        
        if max_users:
            test_users = test_users[:max_users]
        
        metrics = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in k_values}
        map_scores = []
        
        print(f"Evaluating on {len(test_users)} users...")
        
        for idx, user_id in enumerate(test_users):
            test_items = self.test_user_items[user_id]
            recommendations = self.recommend_items_item_based(user_id, n_recommendations=max(k_values))
            
            for k in k_values:
                precision = self.precision_at_k(recommendations, test_items, k)
                recall = self.recall_at_k(recommendations, test_items, k)
                ndcg = self.ndcg_at_k(recommendations, test_items, k)
                
                metrics[k]['precision'].append(precision)
                metrics[k]['recall'].append(recall)
                metrics[k]['ndcg'].append(ndcg)
            
            map_score = self.mean_average_precision(recommendations, test_items)
            map_scores.append(map_score)
            
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
        
        avg_map = np.mean(map_scores)
        print(f"\nMAP: {avg_map:.4f}")
        print("="*60)
        
        return metrics

    def generate_all_recommendations(self, output_file='recommendations.txt', n_recommendations=20, popularity_penalty=0.0, use_svd=False, use_als=False, use_ensemble=False):
        """Generate recommendations for all users (for submission)"""
        print(f"\nGenerating {n_recommendations} recommendations for all users...")
        if use_ensemble and self.use_ensemble:
            print("  Using ENSEMBLE method")
        else:
            print("  Using neural models")
        
        all_user_ids = sorted(self.user_items.keys())
        print(f"  Processing {len(all_user_ids)} users...")
        
        popular_items = self._get_popular_items(2000)
        
        with open(output_file, 'w') as f:
            for idx, user_id in enumerate(all_user_ids):
                recommendations = self.recommend_items_item_based(user_id, n_recommendations, popularity_penalty, 
                                                                 use_svd=use_svd, use_als=use_als, 
                                                                 use_ensemble=use_ensemble)
                
                user_items = self.user_items.get(user_id, set())
                
                # Ensure exactly n_recommendations items
                if len(recommendations) < n_recommendations:
                    for item in popular_items:
                        if item not in user_items and item not in recommendations:
                            recommendations.append(item)
                            if len(recommendations) >= n_recommendations:
                                break
                
                if len(recommendations) < n_recommendations:
                    for item_id in sorted(self.item_ids):
                        if item_id not in user_items and item_id not in recommendations:
                            recommendations.append(item_id)
                            if len(recommendations) >= n_recommendations:
                                break
                
                recommendations = recommendations[:n_recommendations]
                assert len(recommendations) == n_recommendations, f"User {user_id}: Only {len(recommendations)} recommendations (need {n_recommendations})"
                
                line = str(user_id) + ' ' + ' '.join(map(str, recommendations)) + '\n'
                f.write(line)
                
                if (idx + 1) % 5000 == 0:
                    print(f"  Processed {idx + 1}/{len(all_user_ids)} users ({100*(idx+1)/len(all_user_ids):.1f}%)...")
        
        print(f"\n✓ Recommendations saved to '{output_file}'")
        print(f"  Format: user_id item1 item2 ... item{n_recommendations}")
        print(f"  Total users: {len(all_user_ids)}")
        print(f"  Items per user: {n_recommendations}")


def main():
    import sys
    
    # Check GPU availability
    print("="*60)
    print("GPU DETECTION")
    print("="*60)
    gpu_available, device_info = check_gpu_availability()
    if gpu_available:
        print("✓ GPU detected and will be used for acceleration")
        if 'pytorch' in device_info:
            print(f"  PyTorch GPU: {device_info['pytorch']['device']}")
            print(f"  Memory: {device_info['pytorch']['memory']}")
        if 'cupy' in device_info and device_info['cupy']['available']:
            print("  cuPy: Available")
        else:
            print("  cuPy: Not installed")
    else:
        print("✗ No GPU detected - will use CPU (slower)")
    print("="*60)
    print()
    
    recommender = RecommenderSystem()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--evaluate':
        # EVALUATION MODE
        print("\n" + "="*60)
        print("EVALUATION MODE")
        print("="*60)
        
        recommender.load_data('train-2.txt', test_ratio=0.2, random_seed=42)
        recommender.build_user_item_matrix()
        
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Step 1: Cosine similarity (baseline)
        print("\n[1/3] Computing Cosine Similarity...")
        recommender.compute_item_similarity(top_k_similar=200, use_gpu=None, chunk_size=2000)
        
        # Step 2: GMF
        print("\n[2/3] Training GMF...")
        try:
            recommender.compute_gmf_factors(embedding_dim=128, epochs=25, batch_size=1024)
        except Exception as e:
            print(f"  GMF failed: {e}, continuing...")
        
        # Step 3: NeuCF
        print("\n[3/3] Training NeuCF...")
        try:
            recommender.compute_neucf_factors(embedding_dim=128, epochs=30, batch_size=1024, mlp_layers=[128, 64, 32])
        except Exception as e:
            print(f"  NeuCF failed: {e}, continuing...")
        
        # Enable ensemble
        recommender.use_ensemble = True
        
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
        
        recommender.load_data('train-2.txt')
        recommender.build_user_item_matrix()
        
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Step 1: Cosine similarity
        print("\n[1/3] Computing Cosine Similarity...")
        recommender.compute_item_similarity(top_k_similar=200, use_gpu=None, chunk_size=2000)
        
        # Step 2: GMF
        print("\n[2/3] Training GMF...")
        try:
            recommender.compute_gmf_factors(embedding_dim=128, epochs=50)
        except Exception as e:
            print(f"  GMF failed: {e}, continuing...")
        
        # Step 3: NeuCF
        print("\n[3/3] Training NeuCF...")
        try:
            recommender.compute_neucf_factors(embedding_dim=128, epochs=50, mlp_layers=[128, 64, 32])
        except Exception as e:
            print(f"  NeuCF failed: {e}, continuing...")
        
        # Enable ensemble
        recommender.use_ensemble = True
        
        # Generate recommendations
        print("\n" + "="*60)
        print("GENERATING RECOMMENDATIONS")
        print("="*60)
        recommender.generate_all_recommendations(
            output_file='recommendations.txt',
            n_recommendations=20,
            use_ensemble=recommender.use_ensemble
        )

if __name__ == "__main__":
    main()
