"""
Simple NDCG Tuning - Baby Steps
Just try a few different settings and see what works best

NOTE: Works with BOTH versions:
- PyTorch from-scratch implementation
- Microsoft Recommenders library
"""

from recommender_system import RecommenderSystem

def test_configuration(config_name, embedding_dim, n_layers, learning_rate):
    """
    Test one configuration and print results.
    """
    print("\n" + "=" * 60)
    print(f"TESTING: {config_name}")
    print("=" * 60)
    print(f"Embedding size: {embedding_dim}")
    print(f"Number of layers: {n_layers}")
    print(f"Learning rate: {learning_rate}")
    print()
    
    # Create and train model
    recommender = RecommenderSystem()
    recommender.load_raw_data("train-2.txt")
    recommender.split_train_test(test_ratio=0.2, random_seed=42)
    
    # Build matrix (for PyTorch version) or dataframes (for Recommenders library)
    try:
        recommender.build_user_item_matrix()  # PyTorch version
        is_pytorch = True
    except AttributeError:
        recommender._build_dataframes()  # Recommenders library version
        is_pytorch = False
    
    # Train with appropriate method
    if is_pytorch:
        # PyTorch from-scratch version
        recommender.train_lightgcn(
            embedding_dim=embedding_dim,
            num_layers=n_layers,
            reg=0.0001,
            lr=learning_rate,
            epochs=20,
            batch_size=4096,
        )
    else:
        # Microsoft Recommenders library version
        recommender.train_lightgcn(
            epochs=20,
            batch_size=4096,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            learning_rate=learning_rate,
            decay=0.0001,
        )
    
    # Evaluate (only sample 30% of users to be faster)
    metrics = recommender.evaluate(k_values=[5, 10, 20], sample_ratio=0.3)
    
    # Get NDCG scores
    ndcg_5 = sum(metrics[5]['ndcg']) / len(metrics[5]['ndcg'])
    ndcg_10 = sum(metrics[10]['ndcg']) / len(metrics[10]['ndcg'])
    ndcg_20 = sum(metrics[20]['ndcg']) / len(metrics[20]['ndcg'])
    
    return {
        'config': config_name,
        'ndcg@5': ndcg_5,
        'ndcg@10': ndcg_10,
        'ndcg@20': ndcg_20,
        'avg_ndcg': (ndcg_5 + ndcg_10 + ndcg_20) / 3
    }


def main():
    """
    Test just 5 configurations - this will take about 2-3 hours total.
    """
    
    print("\n" + "=" * 60)
    print("SIMPLE NDCG OPTIMIZATION")
    print("Testing 5 configurations")
    print("=" * 60)
    
    # Define 5 configurations to test
    configs = [
        # (name, embedding_dim, n_layers, learning_rate)
        ("Baseline (Current)", 64, 3, 0.001),
        ("Bigger Embeddings", 128, 3, 0.001),
        ("More Layers", 64, 4, 0.001),
        ("Faster Learning", 64, 3, 0.005),
        ("Best Combo", 128, 4, 0.001),
    ]
    
    results = []
    
    # Test each configuration
    for i, (name, emb_dim, layers, lr) in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"Progress: {i}/5 configurations")
        print(f"{'='*60}")
        
        result = test_configuration(name, emb_dim, layers, lr)
        results.append(result)
        
        print(f"\n✓ {name} Results:")
        print(f"  NDCG@5:  {result['ndcg@5']:.4f}")
        print(f"  NDCG@10: {result['ndcg@10']:.4f}")
        print(f"  NDCG@20: {result['ndcg@20']:.4f}")
        print(f"  Average: {result['avg_ndcg']:.4f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY - ALL CONFIGURATIONS")
    print("=" * 60)
    
    # Sort by average NDCG
    results.sort(key=lambda x: x['avg_ndcg'], reverse=True)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['config']}")
        print(f"   Avg NDCG: {result['avg_ndcg']:.4f}")
        print(f"   NDCG@5={result['ndcg@5']:.4f}, "
              f"NDCG@10={result['ndcg@10']:.4f}, "
              f"NDCG@20={result['ndcg@20']:.4f}")
    
    # Save best config
    best = results[0]
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
