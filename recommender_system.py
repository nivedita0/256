"""
Recommender System for User-Item Interactions
This script includes EDA and implements a collaborative filtering recommender system.
"""

import numpy as np
from collections import defaultdict, Counter
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import random
import warnings
warnings.filterwarnings('ignore')


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
        """Perform Exploratory Data Analysis"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic statistics
        num_users = len(self.user_ids)
        num_items = len(self.item_ids)
        total_interactions = sum(len(items) for items in self.user_items.values())
        
        print(f"\n1. Basic Statistics:")
        print(f"   - Number of users: {num_users:,}")
        print(f"   - Number of items: {num_items:,}")
        print(f"   - Total interactions: {total_interactions:,}")
        print(f"   - Average interactions per user: {total_interactions/num_users:.2f}")
        print(f"   - Average interactions per item: {total_interactions/num_items:.2f}")
        
        # Sparsity
        max_possible_interactions = num_users * num_items
        sparsity = 1 - (total_interactions / max_possible_interactions)
        print(f"   - Matrix sparsity: {sparsity*100:.4f}%")
        
        # User interaction distribution
        user_interaction_counts = [len(items) for items in self.user_items.values()]
        print(f"\n2. User Interaction Distribution:")
        print(f"   - Min interactions per user: {min(user_interaction_counts)}")
        print(f"   - Max interactions per user: {max(user_interaction_counts)}")
        print(f"   - Median interactions per user: {np.median(user_interaction_counts):.2f}")
        print(f"   - Mean interactions per user: {np.mean(user_interaction_counts):.2f}")
        print(f"   - Std dev: {np.std(user_interaction_counts):.2f}")
        
        # Item popularity distribution
        item_interaction_counts = [len(users) for users in self.item_users.values()]
        print(f"\n3. Item Popularity Distribution:")
        print(f"   - Min interactions per item: {min(item_interaction_counts)}")
        print(f"   - Max interactions per item: {max(item_interaction_counts)}")
        print(f"   - Median interactions per item: {np.median(item_interaction_counts):.2f}")
        print(f"   - Mean interactions per item: {np.mean(item_interaction_counts):.2f}")
        print(f"   - Std dev: {np.std(item_interaction_counts):.2f}")
        
        # Top items
        item_counts = Counter()
        for items in self.user_items.values():
            item_counts.update(items)
        top_items = item_counts.most_common(10)
        print(f"\n4. Top 10 Most Popular Items:")
        for item_id, count in top_items:
            print(f"   - Item {item_id}: {count} interactions")
        
        # Users with most interactions
        user_counts = [(uid, len(items)) for uid, items in self.user_items.items()]
        user_counts.sort(key=lambda x: x[1], reverse=True)
        print(f"\n5. Top 10 Most Active Users:")
        for user_id, count in user_counts[:10]:
            print(f"   - User {user_id}: {count} interactions")
        
        print("\n" + "="*60 + "\n")
        
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
        
    def compute_item_similarity(self, top_k_similar=200, use_gpu=False, chunk_size=1000):
        """
        Compute item-item similarity using cosine similarity
        
        Args:
            top_k_similar: Only store top-k most similar items per item (saves memory)
            use_gpu: Use GPU acceleration if available (requires cupy)
            chunk_size: Process items in chunks to save memory
        """
        print("Computing item-item similarity matrix...")
        print(f"  Using top-{top_k_similar} similar items per item (approximate)")
        
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
        
    def _get_popular_items(self, n=1000):
        """Cache popular items to avoid recomputing"""
        if self.popular_items_cache is None:
            item_counts = Counter()
            for items in self.user_items.values():
                item_counts.update(items)
            self.popular_items_cache = [item_id for item_id, _ in item_counts.most_common(n)]
        return self.popular_items_cache
    
    def recommend_items_item_based(self, user_id, n_recommendations=20, popularity_penalty=0.0):
        """
        Generate recommendations using item-based collaborative filtering (optimized)
        
        Args:
            user_id: User ID to generate recommendations for
            n_recommendations: Number of recommendations to generate
            popularity_penalty: Penalty factor (0.0-1.0) to reduce popularity bias.
                               Higher values promote long-tail items.
                               Default 0.0 (no penalty, standard behavior)
        """
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
            recommendations = self.recommend_items_item_based(user_id, n_recommendations=max_k)
            
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
    
    def generate_all_recommendations(self, output_file='recommendations.txt', n_recommendations=20, popularity_penalty=0.0):
        """
        Generate recommendations for all users
        
        Args:
            output_file: Output file path
            n_recommendations: Number of recommendations per user
            popularity_penalty: Penalty factor (0.0-1.0) to reduce popularity bias
        """
        print(f"\nGenerating {n_recommendations} recommendations for all users...")
        
        # Get all unique user IDs from the data
        all_user_ids = sorted(self.user_items.keys())
        
        with open(output_file, 'w') as f:
            for user_id in all_user_ids:
                recommendations = self.recommend_items_item_based(user_id, n_recommendations, popularity_penalty)
                # Format: user_id item1 item2 ... item20
                line = str(user_id) + ' ' + ' '.join(map(str, recommendations)) + '\n'
                f.write(line)
                
                if (user_id + 1) % 1000 == 0:
                    print(f"  Processed {user_id + 1}/{len(all_user_ids)} users...")
        
        print(f"\nRecommendations saved to {output_file}")
        print(f"Total users processed: {len(all_user_ids)}")


def main():
    import sys
    
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
        
        # Compute item similarity
        # Options: top_k_similar (default 200 - increased for better coverage), use_gpu (default False), chunk_size (default 1000)
        recommender.compute_item_similarity(top_k_similar=200, use_gpu=False, chunk_size=1000)
        
        # Evaluate on test set
        # Options: max_users (limit users), sample_ratio (sample fraction, e.g., 0.1 = 10%)
        # For faster evaluation: sample_ratio=0.05 (5%) or max_users=2000
        # Default: sample_ratio=0.05 (5% of users - fast, still statistically valid)
        metrics = recommender.evaluate(k_values=[5, 10, 20], max_users=None, sample_ratio=0.05)
        
        print("\nEvaluation complete!")
        print("\nNote: NDCG (Normalized Discounted Cumulative Gain) is a ranking metric")
        print("that gives higher weight to relevant items appearing earlier in the list.")
        
    else:
        # Normal mode: use all data for recommendations
        # Load data (no split)
        recommender.load_data('train-2.txt', test_ratio=0.0)
        
        # Perform EDA
        recommender.perform_eda()
        
        # Build user-item matrix
        # Try both: normalize=True (reduces power user bias) or normalize=False (standard)
        # For this dataset, False might work better - test both!
        # Optional: Set popularity_penalty in recommend_items_item_based to reduce popularity bias
        recommender.build_user_item_matrix(normalize=False)
        
        # Compute item similarity
        # Options: top_k_similar (default 200 - increased for better coverage), use_gpu (default False), chunk_size (default 1000)
        # Set use_gpu=True if you have cuPy installed and want GPU acceleration
        recommender.compute_item_similarity(top_k_similar=200, use_gpu=False, chunk_size=1000)
        
        # Generate recommendations for all users
        # Optional: Add popularity_penalty parameter to reduce popularity bias
        recommender.generate_all_recommendations('recommendations.txt', n_recommendations=20)
        
        print("\nDone! Check 'recommendations.txt' for the output.")
        print("\nTo evaluate the model, run: python recommender_system.py --evaluate")
        print("\nTo use bias mitigation features:")
        print("  - Set normalize=True in build_user_item_matrix() to reduce power user bias")
        print("  - Set popularity_penalty > 0 in recommend_items_item_based() to reduce popularity bias")


if __name__ == '__main__':
    main()

