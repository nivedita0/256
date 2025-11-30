"""
Exploratory Data Analysis (EDA) for Recommender System
Performs comprehensive analysis of user-item interaction data
"""

import numpy as np
from collections import Counter


def perform_eda(user_items, item_users):
    """
    Perform Exploratory Data Analysis on user-item interaction data
    
    Args:
        user_items: Dictionary mapping user_id to set of item_ids
        item_users: Dictionary mapping item_id to set of user_ids
        user_ids: List of all user IDs
        item_ids: List of all item IDs
    """
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
    
    # Item popularity distribution
    item_interaction_counts = [len(users) for users in item_users.values()]
    print(f"\n3. Item Popularity Distribution:")
    print(f"   - Min interactions per item: {min(item_interaction_counts)}")
    print(f"   - Max interactions per item: {max(item_interaction_counts)}")
    print(f"   - Median interactions per item: {np.median(item_interaction_counts):.2f}")
    print(f"   - Mean interactions per item: {np.mean(item_interaction_counts):.2f}")
    print(f"   - Std dev: {np.std(item_interaction_counts):.2f}")
    
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
    
    print("\n" + "="*60 + "\n")

