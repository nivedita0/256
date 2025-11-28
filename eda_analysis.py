"""
Exploratory Data Analysis (EDA) for Recommender System
Performs comprehensive analysis of user-item interaction data
"""

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def perform_eda(user_items, item_users, out_dir="eda_plots"):
    """
    Perform Exploratory Data Analysis on user-item interaction data
    
    Args:
        user_items: Dictionary mapping user_id to set of item_ids
        item_users: Dictionary mapping item_id to set of user_ids
        user_ids: List of all user IDs
        item_ids: List of all item IDs
        out_dir: Folder to save histogram plots
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def save_and_show(fig_name):
        save_path = out_path / fig_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"   - Saved plot to: {save_path}")
        plt.show()  
        plt.close()  


    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Basic statistics
    num_users = len(user_items)
    num_items = len(item_users)
    total_interactions = sum(len(items) for items in user_items.values())
    
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
    user_interaction_counts = [len(items) for items in user_items.values()]
    print(f"\n2. User Interaction Distribution:")
    print(f"   - Min interactions per user: {min(user_interaction_counts)}")
    print(f"   - Max interactions per user: {max(user_interaction_counts)}")
    print(f"   - Median interactions per user: {np.median(user_interaction_counts):.2f}")
    print(f"   - Mean interactions per user: {np.mean(user_interaction_counts):.2f}")
    print(f"   - Std dev: {np.std(user_interaction_counts):.2f}")
    sns.histplot(user_interaction_counts, bins=50, log_scale=(False, True))
    plt.title("Distribution of Interactions per User (log scale)")
    plt.xlabel("Interactions per User")
    plt.ylabel("Count")
    save_and_show("user_interaction_distribution.png")
    
    # Item popularity distribution
    item_interaction_counts = [len(users) for users in item_users.values()]
    print(f"\n3. Item Popularity Distribution:")
    print(f"   - Min interactions per item: {min(item_interaction_counts)}")
    print(f"   - Max interactions per item: {max(item_interaction_counts)}")
    print(f"   - Median interactions per item: {np.median(item_interaction_counts):.2f}")
    print(f"   - Mean interactions per item: {np.mean(item_interaction_counts):.2f}")
    print(f"   - Std dev: {np.std(item_interaction_counts):.2f}")
    sns.histplot(item_interaction_counts, bins=50, log_scale=(False, True))
    plt.title("Item Popularity Distribution (log scale)")
    plt.xlabel("Interactions per Item")
    plt.ylabel("Count")
    save_and_show("item_popularity_distribution.png")
    
    # Top items
    item_counts = Counter()
    for items in user_items.values():
        item_counts.update(items)
    top_items = item_counts.most_common(10)
    print(f"\n4. Top 10 Most Popular Items:")
    for item_id, count in top_items:
        print(f"   - Item {item_id}: {count} interactions")
    
    # Users with most interactions
    user_counts = [(uid, len(items)) for uid, items in user_items.items()]
    user_counts.sort(key=lambda x: x[1], reverse=True)
    print(f"\n5. Top 10 Most Active Users:")
    for user_id, count in user_counts[:10]:
        print(f"   - User {user_id}: {count} interactions")

    # Cold start analysis
    cold_users = sum(1 for c in user_interaction_counts if c < 3)
    cold_items = sum(1 for c in item_interaction_counts if c < 3)
    print(f"\n6. Cold-Start Analysis:")
    print(f"   - Users with <3 interactions: {cold_users} ({cold_users/num_users*100:.2f}%)")
    print(f"   - Items with <3 interactions: {cold_items} ({cold_items/num_items*100:.2f}%)")

    # Gini Coefficient
    def gini(array):
        array = np.sort(np.array(array))
        n = len(array)
        cum = np.cumsum(array)
        return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    print(f"\n7. Gini Coefficient:")
    print(f"   - Gini (user activity inequality): {gini(user_interaction_counts):.3f}")
    print(f"   - Gini (item popularity inequality): {gini(item_interaction_counts):.3f}")

    
    print("\n" + "="*60 + "\n")

