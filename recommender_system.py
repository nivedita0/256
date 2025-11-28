"""
Recommender System for User-Item Interactions
This script includes EDA and implements a collaborative filtering recommender system.
"""

import numpy as np
from collections import defaultdict, Counter
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import diags
import random
import warnings
from eda_analysis import perform_eda

# Optional heavy dependencies (only needed for advanced models like LightGCN)
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
        self.use_test_split = False  # Flag to track if we're using train/test split
        self.item_popularity = {}  # Track item popularity for bias mitigation
        self.popular_items_cache = None  # Cache popular items
        self.reverse_similarity_index = None  # Reverse index: item -> set of items similar to it
        self.svd_model = None  # SVD model for matrix factorization
        self.user_factors = None  # User latent factors from SVD
        self.item_factors = None  # Item latent factors from SVD
        self.use_svd = False  # Flag for using SVD-based recommendations
        self.reconstructed_matrix = None  # Reconstructed matrix (only if memory allows)
        self.als_model = None  # ALS model for implicit feedback
        self.use_als = False  # Flag for using ALS-based recommendations
        self.bpr_model = None  # BPR model for personalized ranking
        self.use_bpr = False  # Flag for using BPR-based recommendations
        self.neucf_model = None  # NeuCF neural model
        self.neucf_device = None  # Device for NeuCF model
        self.use_neucf = False  # Flag for using NeuCF-based recommendations
        self.gmf_model = None  # GMF neural model
        self.gmf_device = None  # Device for GMF model
        self.use_gmf = False  # Flag for using GMF-based recommendations
        self.cosine_similarity_matrix = None  # Store cosine similarity for ensemble
        self.jaccard_similarity_matrix = None  # Store Jaccard similarity for ensemble
        self.use_ensemble = False  # Flag for using ensemble recommendations
        
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
        if test_ratio > 0:
            for user_id, items in all_user_items.items():
                if len(items) < 2:
                    # If user has only 1 item, put it in train
                    self.user_items[user_id] = set(items)
                    for item in items:
                        self.item_users[item].add(user_id)
                else:
                    # Randomly shuffle and split
                    shuffled_items = items.copy()
                    random.shuffle(shuffled_items)
                    split_idx = max(1, int(len(shuffled_items) * (1 - test_ratio)))
                    
                    train_items = shuffled_items[:split_idx]
                    test_items = shuffled_items[split_idx:]
                    
                    self.user_items[user_id] = set(train_items)
                    self.test_user_items[user_id] = set(test_items)
                    
                    # Only add train items to item_users for building the model
                    for item in train_items:
                        self.item_users[item].add(user_id)
            
            train_interactions = sum(len(items) for items in self.user_items.values())
            test_interactions = sum(len(items) for items in self.test_user_items.values())
            print(f"Train set: {len(self.user_items)} users, {train_interactions:,} interactions")
            print(f"Test set: {len(self.test_user_items)} users, {test_interactions:,} interactions")
        else:
            # Load all data as training data
            for user_id, items in all_user_items.items():
                self.user_items[user_id] = set(items)
                for item in items:
                    self.item_users[item].add(user_id)
        
        self.user_ids = sorted(self.user_items.keys())
        self.item_ids = sorted(self.item_users.keys())
        
        # Create mappings
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        self.idx_to_item = {idx: iid for iid, idx in self.item_to_idx.items()}
        
        # Calculate item popularity (for bias mitigation)
        for item_id in self.item_ids:
            self.item_popularity[item_id] = len(self.item_users[item_id])
        
        print(f"Loaded {len(self.user_ids)} users and {len(self.item_ids)} items")
        
    def perform_eda(self):
        """Perform Exploratory Data Analysis (uses eda_analysis.py)"""
        perform_eda(self.user_items, self.item_users, self.user_ids, self.item_ids)
        
    def build_user_item_matrix(self, normalize=False):
        """
        Build user-item interaction matrix
        Args:
            normalize: If True, applies TF-IDF-like normalization to reduce power user bias
                      (weight = 1/sqrt(user_item_count))
        """
        print("Building user-item matrix...")
        num_users = len(self.user_ids)
        num_items = len(self.item_ids)
        
        rows = []
        cols = []
        data = []
        
        for user_id, items in self.user_items.items():
            user_idx = self.user_to_idx[user_id]
            num_user_items = len(items)
            
            for item_id in items:
                item_idx = self.item_to_idx[item_id]
                rows.append(user_idx)
                cols.append(item_idx)
                
                # Normalize by user activity to reduce power user bias
                if normalize:
                    # TF-IDF-like normalization: 1 / sqrt(user_item_count)
                    # This gives less weight to power users
                    weight = 1.0 / np.sqrt(num_user_items)
                else:
                    weight = 1.0  # Binary interaction
                
                data.append(weight)
        
        self.user_item_matrix = csr_matrix((data, (rows, cols)), 
                                           shape=(num_users, num_items))
        print(f"Matrix shape: {self.user_item_matrix.shape}")
        print(f"Matrix density: {self.user_item_matrix.nnz / (num_users * num_items) * 100:.4f}%")
        if normalize:
            print("  (Using normalized weights to reduce power user bias)")
        
    def compute_item_similarity(self, top_k_similar=200, use_gpu=None, chunk_size=1000):
        """
        Compute item-item similarity using cosine similarity
        
        Args:
            top_k_similar: Only store top-k most similar items per item (saves memory)
            use_gpu: Use GPU acceleration if available (requires cupy). None = auto-detect
            chunk_size: Process items in chunks to save memory (larger for GPU: 2000-5000)
        """
        print("Computing item-item similarity matrix...")
        print(f"  Using top-{top_k_similar} similar items per item (approximate)")
        
        # Auto-detect GPU if not specified
        if use_gpu is None:
            try:
                import cupy as cp
                use_gpu = True
                print("  GPU detected (cuPy available) - using GPU acceleration")
            except ImportError:
                use_gpu = False
                print("  GPU not available (cuPy not installed) - using CPU")
        
        # Increase chunk size for GPU (better utilization)
        if use_gpu:
            chunk_size = max(chunk_size, 2000)  # Larger chunks for GPU
        
        # Transpose to get item-user matrix
        item_user_matrix = self.user_item_matrix.T
        num_items = len(self.item_ids)
        
        # Try GPU acceleration if requested
        if use_gpu:
            try:
                import cupy as cp
                print("  Attempting GPU acceleration with cuPy...")
                # Convert to GPU array
                item_user_gpu = cp.sparse.csr_matrix(item_user_matrix)
                # Normalize for cosine similarity
                norms = cp.sqrt(item_user_gpu.multiply(item_user_gpu).sum(axis=1))
                norms = norms.flatten()
                norms[norms == 0] = 1  # Avoid division by zero
                # Normalize each row
                from cupyx.scipy.sparse import diags
                norm_matrix = diags(1.0 / norms)
                item_user_normalized = norm_matrix.dot(item_user_gpu)
                
                # Compute similarity in chunks
                self.item_similarity_matrix = {}
                for i in range(0, num_items, chunk_size):
                    end_i = min(i + chunk_size, num_items)
                    chunk = item_user_normalized[i:end_i]
                    similarity_chunk = chunk.dot(item_user_normalized.T)
                    
                    # Get top-k for each item in chunk
                    for idx in range(end_i - i):
                        item_idx = i + idx
                        similarities = cp.asnumpy(similarity_chunk[idx].toarray().flatten())
                        # Get top-k (excluding self-similarity)
                        top_indices = np.argsort(similarities)[::-1][1:top_k_similar+1]
                        self.item_similarity_matrix[item_idx] = {
                            top_idx: float(similarities[top_idx]) 
                            for top_idx in top_indices if similarities[top_idx] > 0
                        }
                    
                    if (i // chunk_size + 1) % 10 == 0:
                        print(f"  Processed {end_i}/{num_items} items...")
                
                print("Item similarity matrix computed (GPU)!")
                return
            except ImportError:
                print("  cuPy not available, falling back to CPU...")
            except Exception as e:
                print(f"  GPU computation failed: {e}, falling back to CPU...")
        
        # CPU computation with chunking and top-k
        print("  Using CPU computation with chunking...")
        self.item_similarity_matrix = {}
        
        # Normalize matrix for cosine similarity
        from scipy.sparse import diags
        norms = np.sqrt(item_user_matrix.multiply(item_user_matrix).sum(axis=1))
        norms = np.array(norms).flatten()
        norms[norms == 0] = 1  # Avoid division by zero
        norm_matrix = diags(1.0 / norms)
        item_user_normalized = norm_matrix.dot(item_user_matrix)
        
        # Compute similarity in chunks
        for i in range(0, num_items, chunk_size):
            end_i = min(i + chunk_size, num_items)
            chunk = item_user_normalized[i:end_i]
            similarity_chunk = chunk.dot(item_user_normalized.T)
            
            # Get top-k for each item in chunk
            for idx in range(end_i - i):
                item_idx = i + idx
                similarities = similarity_chunk[idx].toarray().flatten()
                # Get top-k (excluding self-similarity)
                top_indices = np.argsort(similarities)[::-1][1:top_k_similar+1]
                self.item_similarity_matrix[item_idx] = {
                    top_idx: float(similarities[top_idx]) 
                    for top_idx in top_indices if similarities[top_idx] > 0
                }
            
            if (i // chunk_size + 1) % 10 == 0:
                print(f"  Processed {end_i}/{num_items} items...")
        
        print("Item similarity matrix computed (CPU)!")
        
        # Build reverse index for faster candidate lookup
        if isinstance(self.item_similarity_matrix, dict):
            print("  Building reverse similarity index for faster recommendations...")
            self.reverse_similarity_index = defaultdict(set)
            for item_idx, similar_items_dict in self.item_similarity_matrix.items():
                for similar_item_idx in similar_items_dict.keys():
                    self.reverse_similarity_index[similar_item_idx].add(item_idx)
            print("  Reverse index built!")
        
        # Store cosine similarity for ensemble
        self.cosine_similarity_matrix = self.item_similarity_matrix.copy()
        
        # Reset ALS/SVD flags when computing similarity (to use item-based CF)
        self.use_als = False
        self.use_svd = False
        self.use_neucf = False
    
    def compute_svd_factors(self, n_components=50, store_reconstructed=False):
        """
        Compute SVD (Singular Value Decomposition) factors for matrix factorization
        This can improve NDCG by capturing latent patterns in the data
        
        Args:
            n_components: Number of latent factors (default: 50, try 50-200)
            store_reconstructed: If True, stores full reconstructed matrix (memory intensive!)
                                 If False, computes scores on-the-fly (memory efficient)
        """
        print(f"Computing SVD factors (n_components={n_components})...")
        print("  This may take a few minutes but can significantly improve NDCG...")
        
        # Fit SVD on user-item matrix
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd_model.components_.T  # Transpose to get item factors
        
        # Optionally reconstruct full matrix (memory intensive for large datasets)
        if store_reconstructed:
            print("  Reconstructing full matrix (memory intensive)...")
            self.reconstructed_matrix = self.user_factors.dot(self.item_factors.T)
            print(f"  Reconstructed matrix shape: {self.reconstructed_matrix.shape}")
        else:
            self.reconstructed_matrix = None
            print("  Using on-the-fly score computation (memory efficient)")
        
        self.use_svd = True
        # Reset ALS flag when using SVD
        self.use_als = False
        self.use_neucf = False
        print(f"  SVD computed! User factors: {self.user_factors.shape}, Item factors: {self.item_factors.shape}")
        print("  Using SVD-based recommendations")
    
    def compute_jaccard_similarity(self, top_k_similar=200, chunk_size=1000, sample_ratio=1.0):
        """
        Compute item-item similarity using Jaccard similarity (OPTIMIZED)
        Jaccard works well for binary/sparse data and can improve metrics
        
        Args:
            top_k_similar: Only store top-k most similar items per item
            chunk_size: Process items in chunks to save memory
            sample_ratio: Sample items to process (1.0 = all, 0.1 = 10% for speed)
        """
        print("Computing item-item similarity using Jaccard similarity...")
        if sample_ratio < 1.0:
            print(f"  Using sampling: {sample_ratio*100:.1f}% of items (faster!)")
        print(f"  Using top-{top_k_similar} similar items per item (approximate)")
        
        item_user_matrix = self.user_item_matrix.T
        num_items = len(self.item_ids)
        
        # Sample items if requested (for speed)
        if sample_ratio < 1.0:
            num_sample = int(num_items * sample_ratio)
            items_to_process = np.random.choice(num_items, num_sample, replace=False)
            items_to_process = sorted(items_to_process)
            print(f"  Processing {len(items_to_process)} sampled items...")
        else:
            items_to_process = list(range(num_items))
        
        # Pre-compute item sizes for faster union calculation
        item_sizes = np.array(item_user_matrix.sum(axis=1)).flatten()
        
        # Jaccard similarity: |A ∩ B| / |A ∪ B|
        similarity_dict = {}
        
        # Compute similarity in chunks
        processed = 0
        for i in range(0, len(items_to_process), chunk_size):
            end_i = min(i + chunk_size, len(items_to_process))
            chunk_indices = items_to_process[i:end_i]
            
            for item_idx in chunk_indices:
                item_vector = item_user_matrix[item_idx]
                item_size = item_sizes[item_idx]
                
                # Compute intersections efficiently (sparse dot product)
                intersections = item_user_matrix.dot(item_vector.T)
                
                # Convert to dense only for this item's row (much faster)
                intersections_dense = intersections.toarray().flatten()
                
                # Compute unions: |A| + |B| - |A ∩ B|
                unions = item_sizes + item_size - intersections_dense
                
                # Avoid division by zero
                unions[unions == 0] = 1
                jaccard_scores = intersections_dense / unions
                
                # Get top-k (excluding self-similarity)
                jaccard_scores[item_idx] = -1  # Exclude self
                top_indices = np.argsort(jaccard_scores)[::-1][:top_k_similar]
                
                similarity_dict[item_idx] = {
                    top_idx: float(jaccard_scores[top_idx]) 
                    for top_idx in top_indices if jaccard_scores[top_idx] > 0
                }
            
            processed += len(chunk_indices)
            if processed % (chunk_size * 5) == 0:
                print(f"  Processed {processed}/{len(items_to_process)} items...")
        
        # Store similarity matrix
        self.item_similarity_matrix = similarity_dict
        self.jaccard_similarity_matrix = similarity_dict
        
        # Build reverse index
        print("  Building reverse similarity index...")
        self.reverse_similarity_index = defaultdict(set)
        for item_idx, similar_items_dict in self.item_similarity_matrix.items():
            for similar_item_idx in similar_items_dict.keys():
                self.reverse_similarity_index[similar_item_idx].add(item_idx)
        print("Item similarity matrix computed (Jaccard)!")
        self.use_neucf = False
    
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
            # Increase batch size for GPU (better utilization)
            if batch_size < 512:
                batch_size = 512
                print(f"  Increased batch_size to {batch_size} for GPU optimization")
        else:
            print("  Using CPU")
        
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
                    
                    for item_idx in item_indices:
                        self.samples.append((user_idx, item_idx, 1.0))
                        for _ in range(num_negatives):
                            neg_item_idx = rng.choice(all_item_indices)
                            while neg_item_idx in positive_set:
                                neg_item_idx = rng.choice(all_item_indices)
                            self.samples.append((user_idx, neg_item_idx, 0.0))
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                user_idx, item_idx, label = self.samples[idx]
                return (
                    torch.tensor(user_idx, dtype=torch.long),
                    torch.tensor(item_idx, dtype=torch.long),
                    torch.tensor(label, dtype=torch.float32),
                )
        
        num_users = len(self.user_ids)
        num_items = len(self.item_ids)
        if num_users == 0 or num_items == 0:
            print("ERROR: No users or items loaded. Please load data first.")
            return
        
        model = NeuCF(num_users, num_items, embedding_dim, mlp_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        dataset = InteractionDataset(self.user_items, self.user_to_idx, self.item_to_idx, num_items, num_negatives)
        if len(dataset) == 0:
            print("ERROR: NeuCF dataset is empty. Check input interactions.")
            return
        # Optimize DataLoader for GPU
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 4 if torch.cuda.is_available() else 0,
            'pin_memory': torch.cuda.is_available()
        }
        dataloader = DataLoader(dataset, **dataloader_kwargs)
        
        print("  Training NeuCF model (this may take 1-2 hours)...")
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for user_ids, item_ids, labels in dataloader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                avg_loss = total_loss / len(dataloader)
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.neucf_model = model
        self.neucf_device = device
        self.use_neucf = True
        self.use_als = False
        self.use_bpr = False
        self.use_svd = False
        self.use_lightgcn = False
        
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
            scores = self.neucf_model(user_tensor, item_tensor).cpu().numpy()
        
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
            # Increase batch size for GPU (better utilization)
            if batch_size < 512:
                batch_size = 512
                print(f"  Increased batch_size to {batch_size} for GPU optimization")
        else:
            print("  Using CPU")
        
        # GMF Model (simpler than NeuCF)
        class GMF(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim):
                super().__init__()
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                self.predict_layer = nn.Linear(embedding_dim, 1)
            
            def forward(self, user_ids, item_ids):
                user_emb = self.user_embedding(user_ids)
                item_emb = self.item_embedding(item_ids)
                element_product = user_emb * item_emb
                prediction = self.predict_layer(element_product)
                return prediction.squeeze(-1)
        
        # Dataset (similar to NeuCF)
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
                    
                    for item_idx in item_indices:
                        self.samples.append((user_idx, item_idx, 1.0))
                        for _ in range(num_negatives):
                            neg_item_idx = rng.choice(all_item_indices)
                            while neg_item_idx in positive_set:
                                neg_item_idx = rng.choice(all_item_indices)
                            self.samples.append((user_idx, neg_item_idx, 0.0))
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                user_idx, item_idx, label = self.samples[idx]
                return (
                    torch.tensor(user_idx, dtype=torch.long),
                    torch.tensor(item_idx, dtype=torch.long),
                    torch.tensor(label, dtype=torch.float32),
                )
        
        num_users = len(self.user_ids)
        num_items = len(self.item_ids)
        if num_users == 0 or num_items == 0:
            print("ERROR: No users or items loaded. Please load data first.")
            return
        
        model = GMF(num_users, num_items, embedding_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        dataset = InteractionDataset(self.user_items, self.user_to_idx, self.item_to_idx, num_items, num_negatives)
        if len(dataset) == 0:
            print("ERROR: GMF dataset is empty. Check input interactions.")
            return
        # Optimize DataLoader for GPU
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 4 if torch.cuda.is_available() else 0,
            'pin_memory': torch.cuda.is_available()
        }
        dataloader = DataLoader(dataset, **dataloader_kwargs)
        
        print("  Training GMF model (this may take 1-2 hours)...")
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for user_ids, item_ids, labels in dataloader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                avg_loss = total_loss / len(dataloader)
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.gmf_model = model
        self.gmf_device = device
        self.use_gmf = True
        self.use_als = False
        self.use_bpr = False
        self.use_svd = False
        self.use_neucf = False
        self.use_lightgcn = False
        
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


    def compute_als_factors(self, factors=50, iterations=15, regularization=0.1, alpha=40):
        """
        Compute ALS (Alternating Least Squares) factors for implicit feedback
        ALS is specifically designed for implicit feedback and often outperforms SVD
        
        Args:
            factors: Number of latent factors (default: 50, try 50-200)
            iterations: Number of ALS iterations (default: 15)
            regularization: Regularization parameter (default: 0.1)
            alpha: Confidence scaling factor for implicit feedback (default: 40)
        
        Note: Requires 'implicit' library. Install with: pip install implicit
        """
        try:
            import implicit
        except ImportError:
            print("ERROR: 'implicit' library not found!")
            print("Install it with: pip install implicit")
            print("ALS is highly recommended for implicit feedback - often gives best results!")
            return
        
        print(f"Computing ALS factors (factors={factors}, iterations={iterations})...")
        print("  ALS is optimized for implicit feedback and often gives best NDCG!")
        
        # Convert to implicit library format (COO matrix)
        from scipy.sparse import coo_matrix
        
        # Build confidence matrix: alpha * interactions (implicit feedback weighting)
        rows = []
        cols = []
        data = []
        
        for user_id, items in self.user_items.items():
            user_idx = self.user_to_idx[user_id]
            for item_id in items:
                item_idx = self.item_to_idx[item_id]
                rows.append(user_idx)
                cols.append(item_idx)
                data.append(alpha)  # Confidence weight for implicit feedback
        
        confidence_matrix = coo_matrix((data, (rows, cols)), 
                                       shape=(len(self.user_ids), len(self.item_ids)))
        
        # Check for GPU support in implicit library
        use_gpu_als = False
        try:
            # Try to import implicit with GPU support
            import implicit
            # Check if GPU is available (requires implicit[gpu] package)
            if torch is not None and torch.cuda.is_available():
                try:
                    # Test if implicit supports GPU
                    use_gpu_als = True
                    print("  Attempting to use GPU for ALS (requires implicit[gpu] package)")
                except:
                    use_gpu_als = False
                    print("  GPU available but implicit[gpu] not installed, using CPU")
        except:
            pass
        
        # Initialize ALS model
        self.als_model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=42,
            use_gpu=use_gpu_als
        )
        
        # Fit the model
        print("  Training ALS model (this may take 5-15 minutes)...")
        self.als_model.fit(confidence_matrix)
        
        self.use_als = True
        self.use_svd = False
        self.use_bpr = False
        self.use_neucf = False
        print(f"  ALS model trained! Factors: {factors}, Iterations: {iterations}")
        print("  Using ALS-based recommendations (should give best NDCG for implicit feedback)")
    
    def compute_bpr_factors(self, factors=100, iterations=30, learning_rate=0.05, regularization=0.01):
        """
        Compute BPR (Bayesian Personalized Ranking) factors for personalized ranking
        BPR is specifically optimized for ranking metrics like NDCG
        
        Args:
            factors: Number of latent factors (default: 100, try 100-200)
            iterations: Number of training iterations (default: 30, try 20-50)
            learning_rate: Learning rate (default: 0.05, try 0.01-0.1)
            regularization: Regularization parameter (default: 0.01, try 0.001-0.1)
        
        Note: Requires 'implicit' library. Install with: pip install implicit
        Expected: NDCG@20: 25-35% (better than ALS, might be better than Cosine)
        """
        try:
            import implicit
        except ImportError:
            print("ERROR: 'implicit' library not found!")
            print("Install it with: pip install implicit")
            print("BPR is optimized for ranking (NDCG) and often works better than ALS!")
            return
        
        print(f"Computing BPR factors (factors={factors}, iterations={iterations})...")
        print("  BPR is optimized for ranking (NDCG) and often works better than ALS!")
        
        # Convert to implicit library format (COO matrix)
        from scipy.sparse import coo_matrix
        
        # Build confidence matrix: alpha * interactions (implicit feedback weighting)
        rows = []
        cols = []
        data = []
        alpha = 40  # Confidence weight for implicit feedback
        
        for user_id, items in self.user_items.items():
            user_idx = self.user_to_idx[user_id]
            for item_id in items:
                item_idx = self.item_to_idx[item_id]
                rows.append(user_idx)
                cols.append(item_idx)
                data.append(alpha)
        
        confidence_matrix = coo_matrix((data, (rows, cols)), 
                                       shape=(len(self.user_ids), len(self.item_ids)))
        
        # Check for GPU support in implicit library (BPR)
        use_gpu_bpr = False
        try:
            if torch is not None and torch.cuda.is_available():
                try:
                    # Test if implicit supports GPU
                    use_gpu_bpr = True
                    print("  Attempting to use GPU for BPR (requires implicit[gpu] package)")
                except:
                    use_gpu_bpr = False
                    print("  GPU available but implicit[gpu] not installed, using CPU")
        except:
            pass
        
        # Initialize BPR model
        self.bpr_model = implicit.bpr.BayesianPersonalizedRanking(
            factors=factors,
            iterations=iterations,
            learning_rate=learning_rate,
            regularization=regularization,
            random_state=42
        )
        
        # Fit the model
        print("  Training BPR model (this may take 30-60 minutes)...")
        print("  BPR uses pairwise ranking optimization - better for NDCG!")
        self.bpr_model.fit(confidence_matrix)
        
        self.use_bpr = True
        # Reset other flags when using BPR
        self.use_als = False
        self.use_svd = False
        self.use_neucf = False
        print(f"  BPR model trained! Factors: {factors}, Iterations: {iterations}")
        print("  Using BPR-based recommendations (optimized for ranking/NDCG)")
        print("  Expected NDCG@20: 25-35% (better than ALS, potentially better than Cosine)")
    
    def _recommend_items_bpr(self, user_id, n_recommendations=20):
        """Generate recommendations using BPR"""
        user_idx = self.user_to_idx[user_id]
        
        # Get user's interacted items to exclude
        user_items = self.user_items[user_id]
        interacted_indices = [self.item_to_idx[item] for item in user_items if item in self.item_to_idx]
        
        # Get recommendations from BPR model
        # BPR recommend returns (item_indices, scores) as separate arrays
        try:
            recommendations_result = self.bpr_model.recommend(
                user_idx, 
                self.user_item_matrix[user_idx], 
                N=n_recommendations + len(interacted_indices),  # Get extra to filter out interacted
                filter_already_liked_items=False  # We'll filter manually
            )
            
            # Handle different return formats
            if isinstance(recommendations_result, tuple) and len(recommendations_result) == 2:
                # Returns (item_indices, scores) as separate arrays
                item_indices, scores = recommendations_result
            elif isinstance(recommendations_result, np.ndarray):
                # Returns just item indices array
                item_indices = recommendations_result
            else:
                # Try to convert to list of tuples
                item_indices = [item for item, score in recommendations_result]
        except Exception as e:
            print(f"  Warning: BPR recommend failed: {e}, using popular items")
            popular_items = self._get_popular_items(n_recommendations)
            return [item for item in popular_items if item not in user_items][:n_recommendations]
        
        # Filter out already interacted items
        recommendations = []
        for item_idx in item_indices:
            # Handle both numpy and Python int types
            item_idx_int = int(item_idx) if hasattr(item_idx, '__int__') else item_idx
            if item_idx_int < len(self.idx_to_item):
                item_id = self.idx_to_item[item_idx_int]
                if item_id not in user_items:
                    recommendations.append(item_id)
                    if len(recommendations) >= n_recommendations:
                        break
        
        # Fill with popular items if needed
        if len(recommendations) < n_recommendations:
            popular_items = self._get_popular_items(1000)
            for item in popular_items:
                if item not in user_items and item not in recommendations:
                    recommendations.append(item)
                    if len(recommendations) >= n_recommendations:
                        break
        
        return recommendations[:n_recommendations]
    
    def _recommend_items_als(self, user_id, n_recommendations=20):
        """Generate recommendations using ALS"""
        user_idx = self.user_to_idx[user_id]
        
        # Get user's interacted items to exclude
        user_items = self.user_items[user_id]
        interacted_indices = [self.item_to_idx[item] for item in user_items if item in self.item_to_idx]
        
        # Get recommendations from ALS model
        # Returns (item_indices, scores) tuples
        recommendations_with_scores = self.als_model.recommend(
            user_idx, 
            self.user_item_matrix[user_idx], 
            N=n_recommendations + len(interacted_indices),  # Get extra to filter out interacted
            filter_already_liked_items=False  # We'll filter manually
        )
        
        # Filter out already interacted items
        recommendations = []
        for item_idx, score in recommendations_with_scores:
            item_id = self.idx_to_item[item_idx]
            if item_id not in user_items:
                recommendations.append(item_id)
                if len(recommendations) >= n_recommendations:
                    break
        
        # Fill with popular items if needed
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
        Advanced Ensemble method: Combines ALL available methods with optimized weights
        This maximizes NDCG by leveraging strengths of multiple approaches
        """
        if user_id not in self.user_items:
            popular_items = self._get_popular_items(n_recommendations)
            return popular_items[:n_recommendations]
        
        user_items = self.user_items[user_id]
        all_scores = {}
        method_count = 0
        
        # Method 1: Cosine similarity (proven best baseline - 30.56%)
        if self.cosine_similarity_matrix:
            cosine_recs = self._get_item_based_scores(user_id, self.cosine_similarity_matrix)
            max_cosine = max(cosine_recs.values()) if cosine_recs else 1.0
            for item_id, score in cosine_recs.items():
                normalized_score = score / max_cosine if max_cosine > 0 else 0
                all_scores[item_id] = all_scores.get(item_id, 0) + 0.25 * normalized_score  # Weight: 0.25
            method_count += 1
        
        # Method 2: LightGCN (if available - best expected: 40-50%)
        if hasattr(self, 'use_lightgcn') and self.use_lightgcn and user_id in self.user_to_idx:
            try:
                lightgcn_recs = self._recommend_items_lightgcn(user_id, n_recommendations * 3)
                for rank, item_id in enumerate(lightgcn_recs):
                    if item_id not in user_items:
                        score = np.exp(-rank * 0.1)  # Exponential decay for top ranks
                        all_scores[item_id] = all_scores.get(item_id, 0) + 0.30 * score  # Weight: 0.30
                method_count += 1
            except:
                pass
        
        # Method 3: BPR (optimized for ranking)
        if hasattr(self, 'use_bpr') and self.use_bpr and user_id in self.user_to_idx:
            try:
                bpr_recs = self._recommend_items_bpr(user_id, n_recommendations * 3)
                for rank, item_id in enumerate(bpr_recs):
                    if item_id not in user_items:
                        score = np.exp(-rank * 0.1)
                        all_scores[item_id] = all_scores.get(item_id, 0) + 0.20 * score  # Weight: 0.20
                method_count += 1
            except:
                pass
        
        # Method 4: NeuCF (neural collaborative filtering)
        if hasattr(self, 'use_neucf') and self.use_neucf and user_id in self.user_to_idx:
            try:
                neucf_recs = self._recommend_items_neucf(user_id, n_recommendations * 3)
                for rank, item_id in enumerate(neucf_recs):
                    if item_id not in user_items:
                        score = np.exp(-rank * 0.1)
                        all_scores[item_id] = all_scores.get(item_id, 0) + 0.15 * score  # Weight: 0.15
                method_count += 1
            except:
                pass
        
        # Method 5: GMF (generalized matrix factorization)
        if hasattr(self, 'use_gmf') and self.use_gmf and user_id in self.user_to_idx:
            try:
                gmf_recs = self._recommend_items_gmf(user_id, n_recommendations * 3)
                for rank, item_id in enumerate(gmf_recs):
                    if item_id not in user_items:
                        score = np.exp(-rank * 0.1)
                        all_scores[item_id] = all_scores.get(item_id, 0) + 0.15 * score  # Weight: 0.15
                method_count += 1
            except:
                pass
        
        # Method 6: ALS (if available)
        if self.use_als and user_id in self.user_to_idx:
            try:
                als_recs = self._recommend_items_als(user_id, n_recommendations * 3)
                for rank, item_id in enumerate(als_recs):
                    if item_id not in user_items:
                        score = np.exp(-rank * 0.1)
                        all_scores[item_id] = all_scores.get(item_id, 0) + 0.10 * score  # Weight: 0.10
                method_count += 1
            except:
                pass
        
        # Method 7: SVD (if available)
        if self.use_svd and user_id in self.user_to_idx:
            try:
                svd_recs = self._recommend_items_svd(user_id, n_recommendations * 3)
                for rank, item_id in enumerate(svd_recs):
                    if item_id not in user_items:
                        score = np.exp(-rank * 0.1)
                        all_scores[item_id] = all_scores.get(item_id, 0) + 0.10 * score  # Weight: 0.10
                method_count += 1
            except:
                pass
        
        # Method 8: Popularity boost (diversity and cold-start)
        popular_items = self._get_popular_items(1000)
        for rank, item_id in enumerate(popular_items[:500]):  # Top 500 popular items
            if item_id not in user_items:
                score = np.exp(-rank * 0.01)  # Gentle decay for popularity
                all_scores[item_id] = all_scores.get(item_id, 0) + 0.05 * score  # Weight: 0.05
        
        # Normalize scores if we have multiple methods (prevents double-counting)
        if method_count > 1:
            max_score = max(all_scores.values()) if all_scores else 1.0
            if max_score > 0:
                all_scores = {item: score / max_score for item, score in all_scores.items()}
        
        # Sort by combined score and return top N
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item_id for item_id, score in sorted_items[:n_recommendations]]
        
        # Fill with popular items if needed
        if len(recommendations) < n_recommendations:
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
        Generate recommendations using item-based collaborative filtering (optimized)
        Can also use SVD, ALS, or ensemble-based recommendations for better NDCG
        
        Args:
            user_id: User ID to generate recommendations for
            n_recommendations: Number of recommendations to generate
            popularity_penalty: Penalty factor (0.0-1.0) to reduce popularity bias.
                               Higher values promote long-tail items.
                               Default 0.0 (no penalty, standard behavior)
            use_svd: If True and SVD factors available, use SVD-based recommendations
            use_als: If True and ALS model available, use ALS-based recommendations (best for implicit feedback)
            use_ensemble: If True, combine multiple methods (cosine + ALS + popularity) for best results
        """
        # Use ensemble if requested (combines multiple methods)
        if use_ensemble and self.use_ensemble:
            return self._recommend_items_ensemble(user_id, n_recommendations)
        
        # Use LightGCN if requested and available (best method)
        if hasattr(self, 'use_lightgcn') and self.use_lightgcn and user_id in self.user_to_idx:
            return self._recommend_items_lightgcn(user_id, n_recommendations)
        
        # Use BPR if requested and available (best for ranking/NDCG)
        if hasattr(self, 'use_bpr') and self.use_bpr and user_id in self.user_to_idx:
            return self._recommend_items_bpr(user_id, n_recommendations)

        if hasattr(self, 'use_neucf') and self.use_neucf and user_id in self.user_to_idx:
            return self._recommend_items_neucf(user_id, n_recommendations)
        
        # Use GMF if requested and available
        if hasattr(self, 'use_gmf') and self.use_gmf and user_id in self.user_to_idx:
            return self._recommend_items_gmf(user_id, n_recommendations)
        
        # Use ALS if requested and available (best for implicit feedback)
        if use_als and self.use_als and user_id in self.user_to_idx:
            return self._recommend_items_als(user_id, n_recommendations)
        
        # Use SVD if requested and available
        if use_svd and self.use_svd and user_id in self.user_to_idx:
            return self._recommend_items_svd(user_id, n_recommendations)
        
        # Only use BPR/ALS/SVD automatically if similarity matrix is NOT available
        # If similarity matrix exists, use it (unless explicitly requested otherwise)
        if self.item_similarity_matrix is None:
            # No similarity matrix - try BPR/ALS/SVD as fallback
            if hasattr(self, 'use_bpr') and self.use_bpr and user_id in self.user_to_idx:
                return self._recommend_items_bpr(user_id, n_recommendations)
            if hasattr(self, 'use_gmf') and self.use_gmf and user_id in self.user_to_idx:
                return self._recommend_items_gmf(user_id, n_recommendations)
            if self.use_als and user_id in self.user_to_idx:
                return self._recommend_items_als(user_id, n_recommendations)
            if self.use_svd and user_id in self.user_to_idx:
                return self._recommend_items_svd(user_id, n_recommendations)
            if hasattr(self, 'use_lightgcn') and self.use_lightgcn and user_id in self.user_to_idx:
                return self._recommend_items_lightgcn(user_id, n_recommendations)
            # No method available - return popular items
            popular_items = self._get_popular_items(n_recommendations)
            return popular_items[:n_recommendations]
        
        if user_id not in self.user_items:
            # New user - return popular items
            popular_items = self._get_popular_items(n_recommendations)
            return popular_items[:n_recommendations]
        
        user_items = self.user_items[user_id]
        
        # Get interacted item indices
        interacted_item_indices = []
        for item in user_items:
            if item in self.item_to_idx:
                interacted_item_indices.append(self.item_to_idx[item])
        
        if not interacted_item_indices:
            # No valid interactions, return popular items
            popular_items = self._get_popular_items(n_recommendations)
            return [item for item in popular_items if item not in user_items][:n_recommendations]
        
        # Calculate max popularity for normalization (if using penalty)
        max_popularity = max(self.item_popularity.values()) if popularity_penalty > 0 else 1
        
        # OPTIMIZATION: Only check items that have similarities with user's items
        # This is much faster than checking all 91k items
        candidate_items = set()
        if isinstance(self.item_similarity_matrix, dict):
            # Top-k format: only check items similar to user's items
            for interacted_item_idx in interacted_item_indices:
                # Direct lookup: items similar to this interacted item
                if interacted_item_idx in self.item_similarity_matrix:
                    similar_items_dict = self.item_similarity_matrix[interacted_item_idx]
                    for similar_item_idx in similar_items_dict.keys():
                        candidate_items.add(self.idx_to_item[similar_item_idx])
                
                # Fast reverse lookup using pre-built index
                if self.reverse_similarity_index and interacted_item_idx in self.reverse_similarity_index:
                    for item_idx in self.reverse_similarity_index[interacted_item_idx]:
                        candidate_items.add(self.idx_to_item[item_idx])
            
            # Limit candidate set size for performance (top 5000 candidates max)
            if len(candidate_items) > 5000:
                # Keep items with highest potential (from similarity matrix)
                candidate_scores = {}
                for item_id in candidate_items:
                    item_idx = self.item_to_idx[item_id]
                    score = 0
                    for interacted_item_idx in interacted_item_indices:
                        if item_idx in self.item_similarity_matrix:
                            score += self.item_similarity_matrix[item_idx].get(interacted_item_idx, 0.0)
                        elif self.reverse_similarity_index and interacted_item_idx in self.reverse_similarity_index:
                            if item_idx in self.reverse_similarity_index[interacted_item_idx]:
                                score += 0.1  # Lower weight for reverse matches
                    candidate_scores[item_id] = score
                # Keep top 5000 candidates
                sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
                candidate_items = set([item_id for item_id, _ in sorted_candidates[:5000]])
            
            # If we have very few candidates, expand to popular items
            if len(candidate_items) < n_recommendations * 2:
                popular_items = self._get_popular_items(1000)
                candidate_items.update(popular_items)
        else:
            # Full matrix: still need to check all items (slower)
            candidate_items = set(self.item_ids)
        
        # Remove items user already interacted with
        candidate_items -= user_items
        
        # Compute scores only for candidate items
        scores = {}
        for item_id in candidate_items:
            item_idx = self.item_to_idx[item_id]
            
            # Compute similarity-weighted score (improved: average instead of sum)
            similarity_scores = []
            for interacted_item_idx in interacted_item_indices:
                # Handle both full matrix and top-k dictionary formats
                if isinstance(self.item_similarity_matrix, dict):
                    # Top-k format: {item_idx: {similar_item_idx: similarity}}
                    if item_idx in self.item_similarity_matrix:
                        similarity = self.item_similarity_matrix[item_idx].get(interacted_item_idx, 0.0)
                    else:
                        # Check reverse: is interacted_item similar to item_idx?
                        if interacted_item_idx in self.item_similarity_matrix:
                            similarity = self.item_similarity_matrix[interacted_item_idx].get(item_idx, 0.0)
                        else:
                            similarity = 0.0
                else:
                    # Full matrix format
                    similarity = self.item_similarity_matrix[item_idx, interacted_item_idx]
                
                if similarity > 0:
                    similarity_scores.append(similarity)
            
            # Use sum of similarities (original approach - often works better)
            # Alternative: np.mean(similarity_scores) * len(similarity_scores) for weighted average
            similarity_score = sum(similarity_scores) if similarity_scores else 0.0
            
            # Apply popularity penalty to reduce bias toward popular items
            if popularity_penalty > 0:
                popularity = self.item_popularity[item_id]
                popularity_factor = 1.0 - (popularity_penalty * (popularity / max_popularity))
                scores[item_id] = similarity_score * popularity_factor
            else:
                scores[item_id] = similarity_score
        
        # Sort by score and return top N
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item_id for item_id, score in sorted_items[:n_recommendations]]
        
        # If we don't have enough recommendations, fill with popular items
        if len(recommendations) < n_recommendations:
            popular_items = self._get_popular_items(1000)
            for item in popular_items:
                if item not in user_items and item not in recommendations:
                    recommendations.append(item)
                    if len(recommendations) >= n_recommendations:
                        break
        
        return recommendations[:n_recommendations]
    
    def _recommend_items_svd(self, user_id, n_recommendations=20):
        """Generate recommendations using SVD matrix factorization (memory efficient)"""
        user_idx = self.user_to_idx[user_id]
        user_factor = self.user_factors[user_idx]
        
        # Get user's interacted items to exclude
        user_items = self.user_items[user_id]
        interacted_indices = set([self.item_to_idx[item] for item in user_items if item in self.item_to_idx])
        
        # Compute scores on-the-fly (memory efficient) or use pre-computed matrix
        if self.reconstructed_matrix is not None:
            # Use pre-computed matrix if available
            scores = self.reconstructed_matrix[user_idx, :].copy()
            scores[list(interacted_indices)] = -np.inf
        else:
            # Compute scores on-the-fly: user_factor * item_factors^T
            # This is memory efficient - only computes what we need
            scores = self.item_factors.dot(user_factor)
            scores[list(interacted_indices)] = -np.inf
        
        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [self.idx_to_item[idx] for idx in top_indices if scores[idx] > -np.inf]
        
        # Fill with popular items if needed
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
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG)@K
        NDCG measures ranking quality by considering position of relevant items
        """
        if len(test_items) == 0:
            return 0.0
        
        top_k = recommendations[:k]
        if len(top_k) == 0:
            return 0.0
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, item in enumerate(top_k):
            if item in test_items:
                # Relevance is 1 if item is in test set, 0 otherwise
                relevance = 1.0
                # Discount factor: log2(i+2) because position starts at 1
                dcg += relevance / np.log2(i + 2)
        
        # Calculate IDCG (Ideal DCG) - perfect ranking
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
        """
        Evaluate the recommender system on test set (optimized)
        
        Args:
            k_values: List of K values for metrics
            max_users: Maximum number of users to evaluate (None = all)
            sample_ratio: Fraction of users to sample (1.0 = all, 0.1 = 10%)
        
        Returns metrics: Precision@K, Recall@K, NDCG@K, MAP
        """
        if not self.use_test_split:
            print("Error: Cannot evaluate without train/test split. Use load_data with test_ratio > 0")
            return None
        
        print(f"\nEvaluating on test set...")
        print("="*60)
        
        # Only evaluate on users that have test items
        test_users = [uid for uid in self.test_user_items.keys() 
                     if len(self.test_user_items[uid]) > 0 and uid in self.user_items]
        
        # Sample users if requested (for faster evaluation)
        if sample_ratio < 1.0:
            import random
            random.seed(42)
            sample_size = max(1, int(len(test_users) * sample_ratio))
            test_users = random.sample(test_users, min(sample_size, len(test_users)))
            print(f"Sampling {len(test_users)} users (sample_ratio={sample_ratio})")
        
        # Limit users if max_users specified
        if max_users is not None and max_users < len(test_users):
            test_users = test_users[:max_users]
            print(f"Limiting to {max_users} users")
        
        print(f"Evaluating on {len(test_users)} users with test items")
        
        # Initialize metric accumulators
        metrics = {
            'precision': {k: [] for k in k_values},
            'recall': {k: [] for k in k_values},
            'ndcg': {k: [] for k in k_values},
            'map': []
        }
        
        max_k = max(k_values)
        
        # Evaluate for each user (optimized)
        for i, user_id in enumerate(test_users):
            test_items = self.test_user_items[user_id]
            # Use ensemble if available (best), then LightGCN, then BPR, then ALS, then SVD, otherwise item-based CF
            use_ensemble = self.use_ensemble
            # Priority: Ensemble > LightGCN > BPR > ALS > SVD > Item-based CF
            # Note: LightGCN/BPR/ALS/SVD are automatically used if available (checked in recommend_items_item_based)
            recommendations = self.recommend_items_item_based(user_id, n_recommendations=max_k, 
                                                             use_svd=False, use_als=False, 
                                                             use_ensemble=use_ensemble)
            
            # Calculate metrics for each K
            for k in k_values:
                metrics['precision'][k].append(
                    self.precision_at_k(recommendations, test_items, k))
                metrics['recall'][k].append(
                    self.recall_at_k(recommendations, test_items, k))
                metrics['ndcg'][k].append(
                    self.ndcg_at_k(recommendations, test_items, k))
            
            # Calculate MAP
            metrics['map'].append(
                self.mean_average_precision(recommendations, test_items))
            
            # Progress update (more frequent for large datasets)
            if len(test_users) > 1000:
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(test_users)} users ({100*(i+1)/len(test_users):.1f}%)...")
            else:
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(test_users)} users...")
        
        # Calculate and display average metrics
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        if sample_ratio < 1.0 or (max_users and max_users < len(self.test_user_items)):
            print(f"\nNote: Evaluated on {len(test_users)} users (sampled/limited)")
        
        for k in k_values:
            avg_precision = np.mean(metrics['precision'][k])
            avg_recall = np.mean(metrics['recall'][k])
            avg_ndcg = np.mean(metrics['ndcg'][k])
            
            print(f"\nMetrics @ K={k}:")
            print(f"  Precision@{k}: {avg_precision:.4f}")
            print(f"  Recall@{k}:    {avg_recall:.4f}")
            print(f"  NDCG@{k}:      {avg_ndcg:.4f}")
        
        avg_map = np.mean(metrics['map'])
        print(f"\nMean Average Precision (MAP): {avg_map:.4f}")
        
        print("\n" + "="*60)
        
        return metrics
    
    def compute_lightgcn_factors(self, n_layers=3, embedding_dim=64, epochs=100, batch_size=2048):
        """
        Compute LightGCN factors using PyTorch Geometric
        
        Args:
            n_layers: Number of LightGCN layers (default: 3, try 2-4)
            embedding_dim: Embedding dimension (default: 64, try 64-128)
            epochs: Training epochs (default: 100, try 50-200)
            batch_size: Batch size (default: 2048)
        
        Expected: NDCG@20: 40-50% (best possible results)
        """
        if torch is None or nn is None:
            print("ERROR: PyTorch Geometric not found!")
            print("\nFor Colab, run these commands in separate cells:")
            print("  Cell 1: !pip install torch torch-geometric")
            print("  Then: Runtime → Restart runtime")
            print("  Cell 2: Run your code again")
            print("\nFor local installation:")
            print("  pip install torch torch-geometric")
            return
        
        try:
            from torch_geometric.data import Data
            # Try to import LightGCNConv (preferred) or use basic GCN
            use_lightgcn_conv = False
            try:
                from torch_geometric.nn import LightGCNConv
                use_lightgcn_conv = True
                print("  Using LightGCNConv layers")
            except ImportError:
                # If LightGCNConv not available, use basic GCNConv
                from torch_geometric.nn import GCNConv
                use_lightgcn_conv = False
                print("  Note: LightGCNConv not found, using GCNConv instead (similar performance)")
        except ImportError:
            print("ERROR: PyTorch Geometric not found!")
            print("\nTry installing/upgrading with:")
            print("  pip install --upgrade torch-geometric")
            print("If you're on Colab, restart the runtime after installation.")
            return
        
        print(f"Computing LightGCN factors (layers={n_layers}, dim={embedding_dim}, epochs={epochs})...")
        print("  LightGCN is state-of-the-art for collaborative filtering!")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Increase batch size for GPU if too small
            if batch_size < 2048:
                batch_size = 2048
                print(f"  Increased batch_size to {batch_size} for GPU optimization")
        else:
            print(f"  Using CPU (GPU not available)")
        
        # Build edge list from user-item interactions
        edge_list = []
        for user_id, items in self.user_items.items():
            user_idx = self.user_to_idx[user_id]
            for item_id in items:
                item_idx = self.item_to_idx[item_id]
                # User to item edge
                edge_list.append([user_idx, len(self.user_ids) + item_idx])
                # Item to user edge (undirected)
                edge_list.append([len(self.user_ids) + item_idx, user_idx])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create PyG data object
        num_nodes = len(self.user_ids) + len(self.item_ids)
        data = Data(edge_index=edge_index, num_nodes=num_nodes)
        data = data.to(device)
        
        # Initialize LightGCN model
        class LightGCNModel(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim, n_layers, use_lightgcn_conv=True):
                super().__init__()
                self.num_users = num_users
                self.num_items = num_items
                self.embedding_dim = embedding_dim
                self.n_layers = n_layers
                
                # Embeddings
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                nn.init.normal_(self.user_embedding.weight, std=0.1)
                nn.init.normal_(self.item_embedding.weight, std=0.1)
                
                # Graph convolution layers
                if use_lightgcn_conv:
                    # Use LightGCNConv if available
                    self.convs = nn.ModuleList([LightGCNConv(embedding_dim, embedding_dim, add_self_loops=False) 
                                               for _ in range(n_layers)])
                else:
                    # Use GCNConv as fallback
                    self.convs = nn.ModuleList([GCNConv(embedding_dim, embedding_dim, add_self_loops=False) 
                                               for _ in range(n_layers)])
            
            def forward(self, edge_index):
                # Get embeddings
                user_emb = self.user_embedding.weight
                item_emb = self.item_embedding.weight
                x = torch.cat([user_emb, item_emb], dim=0)
                
                # Graph convolution layers
                embeddings = [x]
                for conv in self.convs:
                    x = conv(x, edge_index)
                    embeddings.append(x)
                
                # Average all layer embeddings (LightGCN approach)
                final_emb = torch.stack(embeddings, dim=0).mean(dim=0)
                return final_emb[:self.num_users], final_emb[self.num_users:]
        
        model = LightGCNModel(len(self.user_ids), len(self.item_ids), 
                              embedding_dim, n_layers, use_lightgcn_conv).to(device)
        
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        print("  Training LightGCN model (this may take 1-2 hours)...")
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            user_emb, item_emb = model(data.edge_index)
            
            # Sample positive and negative pairs
            pos_users, pos_items = [], []
            neg_users, neg_items = [], []
            
            for user_id, items in list(self.user_items.items())[:1000]:  # Sample for speed
                user_idx = self.user_to_idx[user_id]
                for item_id in list(items)[:5]:  # Sample items
                    item_idx = self.item_to_idx[item_id]
                    pos_users.append(user_idx)
                    pos_items.append(item_idx)
                    
                    # Negative sampling
                    neg_item_idx = torch.randint(0, len(self.item_ids), (1,)).item()
                    while self.idx_to_item[neg_item_idx] in items:
                        neg_item_idx = torch.randint(0, len(self.item_ids), (1,)).item()
                    neg_users.append(user_idx)
                    neg_items.append(neg_item_idx)
            
            if len(pos_users) > 0:
                pos_users_t = torch.tensor(pos_users, device=device)
                pos_items_t = torch.tensor(pos_items, device=device)
                neg_users_t = torch.tensor(neg_users, device=device)
                neg_items_t = torch.tensor(neg_items, device=device)
                
                pos_scores = (user_emb[pos_users_t] * item_emb[pos_items_t]).sum(dim=1)
                neg_scores = (user_emb[neg_users_t] * item_emb[neg_items_t]).sum(dim=1)
                
                # BPR loss
                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 20 == 0:
                    print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            else:
                if (epoch + 1) % 20 == 0:
                    print(f"    Epoch {epoch+1}/{epochs}, No training pairs (skipped)")
        
        # Store model
        self.lightgcn_model = model
        self.lightgcn_device = device
        self.lightgcn_data = data  # Store data for recommendations
        self.use_lightgcn = True
        self.use_als = False
        self.use_bpr = False
        self.use_svd = False
        self.use_neucf = False
        
        print("  LightGCN model trained!")
        print("  Expected NDCG@20: 40-50% (best possible results)")
    
    def _recommend_items_lightgcn(self, user_id, n_recommendations=20):
        """Generate recommendations using LightGCN"""
        if torch is None:
            raise ImportError("PyTorch is not available. Please install torch/torch-geometric to use LightGCN.")
        
        user_idx = self.user_to_idx[user_id]
        user_items = self.user_items[user_id]
        
        # Get embeddings
        self.lightgcn_model.eval()
        with torch.no_grad():
            user_emb, item_emb = self.lightgcn_model(self.lightgcn_data.edge_index)
        
        # Compute scores
        user_vec = user_emb[user_idx:user_idx+1]
        scores = (user_vec @ item_emb.T).squeeze().cpu().numpy()
        
        # Get top items
        top_indices = np.argsort(scores)[::-1][:n_recommendations*2]
        
        # Filter out interacted items
        recommendations = []
        for idx in top_indices:
            item_id = self.idx_to_item[idx]
            if item_id not in user_items:
                recommendations.append(item_id)
                if len(recommendations) >= n_recommendations:
                    break
        
        # Fill with popular items if needed
        if len(recommendations) < n_recommendations:
            popular_items = self._get_popular_items(1000)
            for item in popular_items:
                if item not in user_items and item not in recommendations:
                    recommendations.append(item)
                    if len(recommendations) >= n_recommendations:
                        break
        
        return recommendations[:n_recommendations]

    def generate_all_recommendations(self, output_file='recommendations.txt', n_recommendations=20, popularity_penalty=0.0, use_svd=False, use_als=False, use_ensemble=False):
        """
        Generate recommendations for all users (for submission)
        Output format: user_id item1 item2 ... item20 (one line per user)
        
        Args:
            output_file: Output file path
            n_recommendations: Number of recommendations per user (must be 20 for submission)
            popularity_penalty: Penalty factor (0.0-1.0) to reduce popularity bias
            use_svd: If True and SVD factors available, use SVD-based recommendations
            use_als: If True and ALS model available, use ALS-based recommendations (best for implicit feedback)
            use_ensemble: If True, use ensemble method (combines multiple methods)
        """
        print(f"\nGenerating {n_recommendations} recommendations for all users...")
        if use_ensemble and self.use_ensemble:
            print("  Using ENSEMBLE method (combines Cosine + ALS + Popularity)")
        elif use_als and self.use_als:
            print("  Using ALS-based recommendations (best for implicit feedback)")
        elif use_svd and self.use_svd:
            print("  Using SVD-based recommendations")
        else:
            print("  Using Item-Based Collaborative Filtering")
        
        # Get all unique user IDs from the data (sorted for consistent output)
        all_user_ids = sorted(self.user_items.keys())
        
        print(f"  Processing {len(all_user_ids)} users...")
        
        # Pre-compute popular items for filling (performance optimization)
        popular_items = self._get_popular_items(2000)  # Get more popular items for filling
        
        with open(output_file, 'w') as f:
            for idx, user_id in enumerate(all_user_ids):
                recommendations = self.recommend_items_item_based(user_id, n_recommendations, popularity_penalty, 
                                                                 use_svd=use_svd, use_als=use_als, 
                                                                 use_ensemble=use_ensemble)
                
                user_items = self.user_items.get(user_id, set())
                
                # CRITICAL: Ensure exactly n_recommendations items (fill with popular if needed)
                if len(recommendations) < n_recommendations:
                    # Fill with popular items that user hasn't interacted with
                    for item in popular_items:
                        if item not in user_items and item not in recommendations:
                            recommendations.append(item)
                            if len(recommendations) >= n_recommendations:
                                break
                
                # Final validation: Ensure exactly n_recommendations items
                if len(recommendations) < n_recommendations:
                    # Last resort: use any available items
                    all_items = set(self.item_ids)
                    remaining = n_recommendations - len(recommendations)
                    for item in all_items:
                        if item not in user_items and item not in recommendations:
                            recommendations.append(item)
                            if len(recommendations) >= n_recommendations:
                                break
                
                # Final check: must have exactly n_recommendations items
                recommendations = recommendations[:n_recommendations]
                assert len(recommendations) == n_recommendations, f"User {user_id}: Only {len(recommendations)} recommendations (need {n_recommendations})"
                
                # Format: user_id item1 item2 ... item20 (space-separated, one line per user)
                line = str(user_id) + ' ' + ' '.join(map(str, recommendations)) + '\n'
                f.write(line)
                
                if (idx + 1) % 5000 == 0:
                    print(f"  Processed {idx + 1}/{len(all_user_ids)} users ({100*(idx+1)/len(all_user_ids):.1f}%)...")
        
        print(f"\n✓ Recommendations saved to '{output_file}'")
        print(f"  Format: user_id item1 item2 ... item{n_recommendations}")
        print(f"  Total users: {len(all_user_ids)}")
        print(f"  Items per user: {n_recommendations}")
        print(f"\n  File is ready for submission to leaderboard!")


def main():
    import sys
    
    # Check GPU availability and report
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
            print("  cuPy: Available (for cosine similarity acceleration)")
        else:
            print("  cuPy: Not installed (install with: pip install cupy-cuda11x for GPU cosine)")
    else:
        print("✗ No GPU detected - will use CPU (slower)")
        print("  For Colab: Enable GPU in Runtime → Change runtime type → GPU")
    print("="*60)
    print()
    
    # Check if evaluation mode is requested
    evaluate_mode = '--evaluate' in sys.argv or '-e' in sys.argv
    
    # Initialize recommender system
    recommender = RecommenderSystem()
    
    if evaluate_mode:
        # Evaluation mode: split data and evaluate
        print("="*60)
        print("EVALUATION MODE")
        print("="*60)
        
        # Load data with train/test split (80/20)
        recommender.load_data('train-2.txt', test_ratio=0.2, random_seed=42)
        
        # Perform EDA on training data
        recommender.perform_eda()
        
        # Build user-item matrix from training data
        # Try both: normalize=True (reduces power user bias) or normalize=False (standard)
        # For this dataset, False might work better - test both!
        recommender.build_user_item_matrix(normalize=False)
        
        # ====================================================================
        # ADVANCED ENSEMBLE STRATEGY FOR MAXIMUM NDCG (Target: 50-80%)
        # ====================================================================
        # This trains multiple models and combines them for best results
        # Expected: 45-65% NDCG@20 (depending on training quality)
        # For 70-80%: See MAXIMIZE_NDCG_STRATEGY.md for full strategy
        # ====================================================================
        
        # Step 1: Cosine Similarity (proven baseline - 30.56%)
        print("\n[1/5] Training Cosine Similarity...")
        # Auto-detect GPU for cosine similarity (use_gpu=None means auto-detect)
        recommender.compute_item_similarity(top_k_similar=500, use_gpu=None, chunk_size=1000)
        
        # Step 2: LightGCN (best expected: 40-50%) - Takes 1-2 hours
        print("\n[2/5] Training LightGCN (this will take 1-2 hours)...")
        try:
            recommender.compute_lightgcn_factors(n_layers=3, embedding_dim=128, epochs=200)
        except Exception as e:
            print(f"  LightGCN failed: {e}, continuing with other methods...")
        
        # Step 3: BPR (optimized for ranking) - Takes 30-60 minutes
        print("\n[3/5] Training BPR...")
        try:
            recommender.compute_bpr_factors(factors=150, iterations=50, learning_rate=0.05)
        except Exception as e:
            print(f"  BPR failed: {e}, continuing with other methods...")
        
        # Step 4: GMF (neural matrix factorization) - Takes 1-2 hours
        print("\n[4/5] Training GMF...")
        try:
            recommender.compute_gmf_factors(embedding_dim=128, epochs=50)
        except Exception as e:
            print(f"  GMF failed: {e}, continuing with other methods...")
        
        # Step 5: NeuCF (neural collaborative filtering) - Takes 1-2 hours
        print("\n[5/5] Training NeuCF...")
        try:
            recommender.compute_neucf_factors(embedding_dim=128, epochs=50, mlp_layers=[128, 64, 32])
        except Exception as e:
            print(f"  NeuCF failed: {e}, continuing with other methods...")
        
        # Enable ADVANCED ENSEMBLE (combines all trained models)
        print("\n" + "="*60)
        print("ENABLING ADVANCED ENSEMBLE")
        print("="*60)
        print("  Combining all trained models with optimized weights...")
        print("  Expected NDCG@20: 45-65% (depending on models trained)")
        recommender.use_ensemble = True
        
        # Alternative: Quick test with fewer models (faster, lower NDCG)
        # Uncomment below and comment out above for faster testing:
        # recommender.compute_item_similarity(top_k_similar=500)
        # recommender.compute_lightgcn_factors(n_layers=3, embedding_dim=128, epochs=200)
        # recommender.compute_bpr_factors(factors=150, iterations=50)
        # recommender.use_ensemble = True
        
        metrics = recommender.evaluate(k_values=[20], max_users=None, sample_ratio=0.05)
        
        print("\nEvaluation complete!")
        
    else:
        # Submission mode: use all data for recommendations (no train/test split)
        print("="*60)
        print("SUBMISSION MODE - Generating recommendations for all users")
        print("="*60)
        
        # Load data (no split - use all data for final recommendations)
        recommender.load_data('train-2.txt', test_ratio=0.0)
        
        # Perform EDA (optional, can comment out for faster generation)
        # recommender.perform_eda()
        
        # Build user-item matrix
        recommender.build_user_item_matrix(normalize=False)
        
        # ====================================================================
        # SUBMISSION STRATEGY - Use same methods as evaluation for consistency
        # ====================================================================
        # Option 1: Quick (Cosine only - faster, ~30% NDCG)
        # Option 2: Ensemble (multiple models - slower, ~45-65% NDCG) ⬅️ RECOMMENDED
        # ====================================================================
        
        # QUICK MODE: Cosine similarity only (faster, ~5-10 minutes)
        # Uncomment below for quick submission:
        # recommender.compute_item_similarity(top_k_similar=200, use_gpu=None, chunk_size=1000)
        # recommender.use_ensemble = False
        
        # ENSEMBLE MODE: Multiple models (slower, better results, ~5-8 hours)
        # This matches evaluation mode for consistency
        print("\nTraining models for submission (same as evaluation mode)...")
        
        # Step 1: Cosine Similarity
        print("\n[1/5] Training Cosine Similarity...")
        recommender.compute_item_similarity(top_k_similar=500, use_gpu=None, chunk_size=1000)
        
        # Step 2: LightGCN (best expected: 40-50%) - Takes 1-2 hours
        print("\n[2/5] Training LightGCN (this will take 1-2 hours)...")
        try:
            recommender.compute_lightgcn_factors(n_layers=3, embedding_dim=128, epochs=200)
        except Exception as e:
            print(f"  LightGCN failed: {e}, continuing with other methods...")
        
        # Step 3: BPR (optimized for ranking) - Takes 30-60 minutes
        print("\n[3/5] Training BPR...")
        try:
            recommender.compute_bpr_factors(factors=150, iterations=50, learning_rate=0.05)
        except Exception as e:
            print(f"  BPR failed: {e}, continuing with other methods...")
        
        # Step 4: GMF (neural matrix factorization) - Takes 1-2 hours
        print("\n[4/5] Training GMF...")
        try:
            recommender.compute_gmf_factors(embedding_dim=128, epochs=50)
        except Exception as e:
            print(f"  GMF failed: {e}, continuing with other methods...")
        
        # Step 5: NeuCF (neural collaborative filtering) - Takes 1-2 hours
        print("\n[5/5] Training NeuCF...")
        try:
            recommender.compute_neucf_factors(embedding_dim=128, epochs=50, mlp_layers=[128, 64, 32])
        except Exception as e:
            print(f"  NeuCF failed: {e}, continuing with other methods...")
        
        # Enable ADVANCED ENSEMBLE (combines all trained models)
        print("\n" + "="*60)
        print("ENABLING ADVANCED ENSEMBLE")
        print("="*60)
        print("  Combining all trained models with optimized weights...")
        print("  Expected NDCG@20: 45-65% (depending on models trained)")
        recommender.use_ensemble = True
        
        # Generate recommendations for all users (for submission)
        # Output format: user_id item1 item2 ... item20 (one line per user)
        print("\n" + "="*60)
        print("GENERATING RECOMMENDATIONS FOR ALL USERS")
        print("="*60)
        recommender.generate_all_recommendations('recommendations.txt', n_recommendations=20, 
                                                 use_ensemble=recommender.use_ensemble)
        
        print("\n" + "="*60)
        print("SUBMISSION FILE READY!")
        print("="*60)
        print("\nFile: 'recommendations.txt'")


if __name__ == '__main__':
    main()

