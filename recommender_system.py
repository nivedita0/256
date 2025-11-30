"""
Recommender System for User-Item Interactions
Using Microsoft Recommenders library for LightGCN implementation.
"""

import numpy as np
import pandas as pd
import random
import warnings
from collections import defaultdict, Counter
from eda_analysis import perform_eda

warnings.filterwarnings('ignore')

# Microsoft Recommenders library
try:
    from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
    from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
    from recommenders.utils.constants import SEED as DEFAULT_SEED
    import tensorflow as tf
    RECOMMENDERS_AVAILABLE = True
except ImportError:
    print("Warning: Microsoft Recommenders library not found.")
    print("Install with: pip install recommenders[gpu] or recommenders[cpu]")
    RECOMMENDERS_AVAILABLE = False


def check_gpu_availability():
    """
    Check GPU availability for TensorFlow (LightGCN acceleration).
    Returns:
        gpu_available (bool), device_info (dict)
    """
    gpu_available = False
    device_info = {"tensorflow": {"available": False, "devices": []}}
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_available = True
            device_info["tensorflow"]["available"] = True
            device_info["tensorflow"]["devices"] = [gpu.name for gpu in gpus]
    except Exception as e:
        print(f"GPU check failed: {e}")
        gpu_available = False
    
    return gpu_available, device_info


class RecommenderSystem:
    def __init__(self):
        # Data containers
        self.user_items = defaultdict(set)       # user_id -> set of item_ids (train)
        self.test_user_items = defaultdict(set)  # user_id -> set of item_ids (test)
        self.item_users = defaultdict(set)       # item_id -> set of user_ids

        # DataFrame for recommenders library
        self.train_df = None
        self.test_df = None
        
        # Model
        self.model = None
        self.model_params = None
        
        # Popularity / fallback
        self.item_popularity = {}
        self.popular_items_cache = None
        
        # User/item mappings
        self.all_user_ids = []
        self.all_item_ids = []

    # ----------------- Data loading & EDA -----------------

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
                
                # Remove duplicates
                items = list(set(items))
                
                self.user_items[user_id] = set(items)
                for it in items:
                    self.item_users[it].add(user_id)

        print(f"Loaded {len(self.user_items)} users and {len(self.item_users)} items")
        
        # Data validation
        empty_users = [u for u, items in self.user_items.items() if len(items) == 0]
        if empty_users:
            print(f"Warning: Found {len(empty_users)} users with no interactions")
        
        empty_items = [i for i, users in self.item_users.items() if len(users) == 0]
        if empty_items:
            print(f"Warning: Found {len(empty_items)} items with no interactions")

    def perform_eda(self):
        """Perform exploratory data analysis."""
        perform_eda(self.user_items, self.item_users)

    def split_train_test(self, test_ratio=0.2, random_seed=42):
        """Create train/test split from current user_items."""
        random.seed(random_seed)
        np.random.seed(random_seed)

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

    def _build_dataframes(self):
        """Convert user_items to DataFrame format for recommenders library."""
        print("Building DataFrames for recommenders library...")
        
        # Build train DataFrame
        train_data = []
        for user_id, items in self.user_items.items():
            for item_id in items:
                train_data.append({
                    'userID': user_id,
                    'itemID': item_id,
                    'rating': 1.0  # Implicit feedback
                })
        
        self.train_df = pd.DataFrame(train_data)
        
        # Build test DataFrame if available
        if self.test_user_items:
            test_data = []
            for user_id, items in self.test_user_items.items():
                for item_id in items:
                    test_data.append({
                        'userID': user_id,
                        'itemID': item_id,
                        'rating': 1.0
                    })
            self.test_df = pd.DataFrame(test_data)
        else:
            self.test_df = None
        
        print(f"Train DataFrame: {len(self.train_df)} interactions")
        if self.test_df is not None:
            print(f"Test DataFrame: {len(self.test_df)} interactions")
        
        # Compute item popularity
        self.item_popularity = self.train_df['itemID'].value_counts().to_dict()
        
        # Store all unique users and items
        self.all_user_ids = sorted(self.user_items.keys())
        self.all_item_ids = sorted(self.item_users.keys())

    # ----------------- LightGCN training using Recommenders library -----------------

    def train_lightgcn(
        self,
        epochs=20,
        batch_size=4096,
        embedding_dim=64,
        n_layers=3,
        learning_rate=0.001,
        decay=0.0001,
        eval_epoch=5,
        top_k=20,
    ):
        """
        Train LightGCN using Microsoft Recommenders library.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            embedding_dim: Dimension of embeddings
            n_layers: Number of graph convolutional layers
            learning_rate: Learning rate
            decay: L2 regularization weight
            eval_epoch: Evaluate every N epochs
            top_k: Top K items to consider during evaluation
        """
        if not RECOMMENDERS_AVAILABLE:
            raise ImportError("Microsoft Recommenders library is required. "
                            "Install with: pip install recommenders[gpu]")
        
        assert self.train_df is not None, "Call _build_dataframes() first or load data."
        
        print("\nInitializing LightGCN model from Microsoft Recommenders...")
        
        # Check GPU
        gpu_available, device_info = check_gpu_availability()
        if gpu_available:
            print(f"✓ TensorFlow GPU detected - using GPU acceleration")
            print(f"  Devices: {device_info['tensorflow']['devices']}")
        else:
            print("✗ No GPU detected - using CPU")
        
        # Prepare data in ImplicitCF format
        data = ImplicitCF(
            train=self.train_df,
            test=self.test_df if self.test_df is not None else self.train_df,
            seed=DEFAULT_SEED
        )
        
        # Model hyperparameters
        self.model_params = {
            "n_layers": n_layers,
            "batch_size": batch_size,
            "decay": decay,
            "embed_size": embedding_dim,
            "learning_rate": learning_rate,
            "top_k": top_k,
            "epochs": epochs,
            "eval_epoch": eval_epoch,
            "save_model": False,
            "save_epoch": epochs,
        }
        
        print(f"\nModel Configuration:")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Number of layers: {n_layers}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  L2 regularization: {decay}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {epochs}")
        
        # Initialize model
        self.model = LightGCN(**self.model_params, data=data)
        
        # Train model
        print(f"\nTraining LightGCN on {len(self.train_df)} interactions...")
        print("=" * 60)
        
        self.model.fit()
        
        print("=" * 60)
        print("LightGCN training complete.\n")

    def _get_popular_items(self, n=1000):
        """Compute (and cache) globally popular items."""
        if self.popular_items_cache is None:
            if self.item_popularity:
                self.popular_items_cache = sorted(
                    self.item_popularity.keys(),
                    key=lambda x: self.item_popularity[x],
                    reverse=True
                )[:n]
            else:
                item_counts = Counter()
                for items in self.user_items.values():
                    item_counts.update(items)
                self.popular_items_cache = [
                    item_id for item_id, _ in item_counts.most_common(n)
                ]
        return self.popular_items_cache

    def recommend_items_lightgcn(self, user_id, n_recommendations=20, remove_seen=True):
        """
        Generate recommendations using trained LightGCN model.
        
        Args:
            user_id: User ID to generate recommendations for
            n_recommendations: Number of items to recommend
            remove_seen: Whether to remove items user has already interacted with
        
        Returns:
            List of recommended item IDs
        """
        if self.model is None:
            raise RuntimeError("LightGCN model is not trained. Call train_lightgcn() first.")
        
        # Cold-start user: return popular items
        if user_id not in self.user_items or len(self.user_items[user_id]) == 0:
            popular_items = self._get_popular_items(n_recommendations)
            return popular_items[:n_recommendations]
        
        # Get recommendations from model
        try:
            # Model expects user_id as is
            scores = self.model.recommend_k_items(
                user_id,
                top_k=n_recommendations * 3,  # Get more to filter
                remove_seen=remove_seen
            )
            
            # scores is a dictionary: {item_id: score, ...}
            recommendations = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            
        except Exception as e:
            print(f"Warning: Error getting recommendations for user {user_id}: {e}")
            # Fallback to popular items
            recommendations = []
        
        # Filter out already seen items if needed
        if remove_seen:
            interacted = self.user_items.get(user_id, set())
            recommendations = [item for item in recommendations if item not in interacted]
        
        # Take top N
        recommendations = recommendations[:n_recommendations]
        
        # Fallback: fill with popular items
        if len(recommendations) < n_recommendations:
            interacted = self.user_items.get(user_id, set())
            popular_items = self._get_popular_items(n_recommendations * 2)
            
            for item_id in popular_items:
                if item_id not in interacted and item_id not in recommendations:
                    recommendations.append(item_id)
                    if len(recommendations) >= n_recommendations:
                        break
        
        return recommendations[:n_recommendations]

    # ----------------- Evaluation methods -----------------

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
                dcg += 1.0 / np.log2(i + 2)

        ideal_relevance = sorted(
            [1.0 if item in test_items else 0.0 for item in top_k],
            reverse=True,
        )
        idcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(ideal_relevance)
            if rel > 0
        )

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
        """Evaluate the recommender system on test set."""
        if len(self.test_user_items) == 0:
            print("Error: No test split available. Run split_train_test() first.")
            return None

        print(f"\nEvaluating on test set...")
        print("=" * 60)

        test_users = [
            uid
            for uid in self.test_user_items.keys()
            if len(self.test_user_items[uid]) > 0 and uid in self.user_items
        ]

        if sample_ratio < 1.0:
            random.seed(42)
            num_sample = int(len(test_users) * sample_ratio)
            test_users = random.sample(test_users, num_sample)
            print(
                f"Sampling {len(test_users)} users ({sample_ratio*100:.1f}%) for evaluation"
            )

        if max_users:
            test_users = test_users[:max_users]

        metrics = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}
        map_scores = []

        print(f"Evaluating on {len(test_users)} users...")

        for idx, user_id in enumerate(test_users):
            test_items = self.test_user_items[user_id]
            recommendations = self.recommend_items_lightgcn(
                user_id, n_recommendations=max(k_values)
            )

            for k in k_values:
                precision = self.precision_at_k(recommendations, test_items, k)
                recall = self.recall_at_k(recommendations, test_items, k)
                ndcg = self.ndcg_at_k(recommendations, test_items, k)

                metrics[k]["precision"].append(precision)
                metrics[k]["recall"].append(recall)
                metrics[k]["ndcg"].append(ndcg)

            map_score = self.mean_average_precision(recommendations, test_items)
            map_scores.append(map_score)

            if (idx + 1) % 1000 == 0:
                print(f"  Evaluated {idx + 1}/{len(test_users)} users...")

        print("\nEvaluation Results:")
        print("=" * 60)
        for k in k_values:
            avg_precision = np.mean(metrics[k]["precision"])
            avg_recall = np.mean(metrics[k]["recall"])
            avg_ndcg = np.mean(metrics[k]["ndcg"])

            print(f"K={k}:")
            print(f"  Precision@{k}: {avg_precision:.4f}")
            print(f"  Recall@{k}: {avg_recall:.4f}")
            print(f"  NDCG@{k}: {avg_ndcg:.4f}")

        avg_map = np.mean(map_scores)
        print(f"\nMAP: {avg_map:.4f}")
        print("=" * 60)

        return metrics

    def generate_all_recommendations(
        self,
        output_file="recommendations.txt",
        n_recommendations=20,
    ):
        """Generate recommendations for all users (for submission)."""
        print(
            f"\nGenerating {n_recommendations} recommendations for all users using LightGCN..."
        )

        all_user_ids = sorted(self.user_items.keys())
        print(f"  Processing {len(all_user_ids)} users...")

        popular_items = self._get_popular_items(2000)

        with open(output_file, "w") as f:
            for idx, user_id in enumerate(all_user_ids):
                recommendations = self.recommend_items_lightgcn(
                    user_id, n_recommendations
                )

                user_items = self.user_items.get(user_id, set())

                # Ensure exactly n_recommendations items
                if len(recommendations) < n_recommendations:
                    for item in popular_items:
                        if item not in user_items and item not in recommendations:
                            recommendations.append(item)
                            if len(recommendations) >= n_recommendations:
                                break

                if len(recommendations) < n_recommendations:
                    for item_id in self.all_item_ids:
                        if item_id not in user_items and item_id not in recommendations:
                            recommendations.append(item_id)
                            if len(recommendations) >= n_recommendations:
                                break

                recommendations = recommendations[:n_recommendations]
                assert (
                    len(recommendations) == n_recommendations
                ), f"User {user_id}: Only {len(recommendations)} recommendations (need {n_recommendations})"

                line = str(user_id) + " " + " ".join(map(str, recommendations)) + "\n"
                f.write(line)

                if (idx + 1) % 5000 == 0:
                    print(
                        f"  Processed {idx + 1}/{len(all_user_ids)} users "
                        f"({100*(idx+1)/len(all_user_ids):.1f}%)..."
                    )

        print(f"\n✓ Recommendations saved to '{output_file}'")
        print(f"  Format: user_id item1 item2 ... item{n_recommendations}")
        print(f"  Total users: {len(all_user_ids)}")
        print(f"  Items per user: {n_recommendations}")


def main():
    import sys

    # Check dependencies
    if not RECOMMENDERS_AVAILABLE:
        print("=" * 60)
        print("ERROR: Microsoft Recommenders library not installed")
        print("=" * 60)
        print("Please install with one of the following:")
        print("  pip install recommenders[gpu]  # For GPU support")
        print("  pip install recommenders[cpu]  # For CPU only")
        print("=" * 60)
        return

    # Check GPU availability
    print("=" * 60)
    print("TENSORFLOW GPU DETECTION")
    print("=" * 60)
    gpu_available, device_info = check_gpu_availability()
    if gpu_available:
        print("✓ GPU detected and will be used for LightGCN acceleration")
        print(f"  TensorFlow devices: {device_info['tensorflow']['devices']}")
    else:
        print("✗ No GPU detected - will use CPU")
    print("=" * 60)
    print()

    recommender = RecommenderSystem()

    if len(sys.argv) > 1 and sys.argv[1] == "--evaluate":
        # EVALUATION MODE
        print("\n" + "=" * 60)
        print("EVALUATION MODE (LightGCN - Microsoft Recommenders)")
        print("=" * 60)

        recommender.load_raw_data("train-2.txt")
        recommender.perform_eda()
        recommender.split_train_test(test_ratio=0.2, random_seed=42)
        recommender._build_dataframes()

        # Train LightGCN
        recommender.train_lightgcn(
            epochs=20,
            batch_size=4096,
            embedding_dim=64,
            n_layers=3,
            learning_rate=0.005,
            decay=0.0001,
            eval_epoch=5,
            top_k=20,
        )

        # Evaluate
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        recommender.evaluate(k_values=[5, 10, 20], sample_ratio=1.0)

    else:
        # SUBMISSION MODE
        print("\n" + "=" * 60)
        print("SUBMISSION MODE (LightGCN - Microsoft Recommenders)")
        print("=" * 60)

        recommender.load_raw_data("train-2.txt")
        recommender.perform_eda()
        recommender._build_dataframes()

        # Train LightGCN on all data
        recommender.train_lightgcn(
            epochs=20,
            batch_size=4096,
            embedding_dim=64,
            n_layers=3,
            learning_rate=0.005,
            decay=0.0001,
            eval_epoch=5,
            top_k=20,
        )

        # Generate recommendations for leaderboard submission
        print("\n" + "=" * 60)
        print("GENERATING RECOMMENDATIONS")
        print("=" * 60)
        recommender.generate_all_recommendations(
            output_file="recommendations.txt",
            n_recommendations=20,
        )


if __name__ == "__main__":
    main()