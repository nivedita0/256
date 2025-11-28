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

warnings.filterwarnings('ignore')


def check_gpu_availability():
    """Check GPU availability for cuPy (cosine similarity acceleration)"""
    gpu_available = False
    device_info = {}
    
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
            row_sums = np.array(self.user_item_matrix.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1
            row_scaling = diags(1.0 / row_sums)
            self.user_item_matrix = row_scaling @ self.user_item_matrix
        
        # Compute item popularity
        for item_id, users in self.item_users.items():
            self.item_popularity[item_id] = len(users)
        
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
            try:
                import cupy as cp
                use_gpu = True
            except ImportError:
                use_gpu = False
        
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
        
        print("  Building reverse similarity index...")
        self.reverse_similarity_index = defaultdict(set)
        for item_idx, similar_items_dict in self.item_similarity_matrix.items():
            for similar_item_idx in similar_items_dict.keys():
                self.reverse_similarity_index[similar_item_idx].add(item_idx)
        print("Item similarity matrix computed!")
        
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
        Generate recommendations using item-based collaborative filtering
        
        Args:
            user_id: User ID to generate recommendations for
            n_recommendations: Number of recommendations to generate
            popularity_penalty: Penalty factor (0.0-1.0) to reduce popularity bias
        """
            # New user - return popular items
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
    
    def generate_all_recommendations(self, output_file='recommendations.txt', n_recommendations=20, popularity_penalty=0.0):
        """Generate recommendations for all users (for submission)"""
        print(f"\nGenerating {n_recommendations} recommendations for all users...")
        print("  Using Item-Based Collaborative Filtering with Cosine Similarity")
        
        all_user_ids = sorted(self.user_items.keys())
        print(f"  Processing {len(all_user_ids)} users...")
        
        popular_items = self._get_popular_items(2000)
        
        with open(output_file, 'w') as f:
            for idx, user_id in enumerate(all_user_ids):
                recommendations = self.recommend_items_item_based(user_id, n_recommendations, popularity_penalty)
                
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
        print("✓ GPU detected and will be used for cosine similarity acceleration")
        if 'cupy' in device_info and device_info['cupy']['available']:
            print("  cuPy: Available")
        else:
            print("  cuPy: Not installed (install with: pip install cupy-cuda11x for GPU acceleration)")
    else:
        print("✗ No GPU detected - will use CPU")
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
        print("COMPUTING SIMILARITY")
        print("="*60)
        
        # Compute cosine similarity
        print("\nComputing Cosine Similarity...")
        recommender.compute_item_similarity(top_k_similar=200, use_gpu=None, chunk_size=2000)
        
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
        print("COMPUTING SIMILARITY")
        print("="*60)
        
        # Compute cosine similarity
        print("\nComputing Cosine Similarity...")
        recommender.compute_item_similarity(top_k_similar=200, use_gpu=None, chunk_size=2000)
        
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
