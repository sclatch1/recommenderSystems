"""
Enhanced RecPack pipeline with comprehensive analysis and visualization.
"""

import argparse
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from recpack.datasets import MovieLens1M, DummyDataset
from recpack.scenarios import WeakGeneralization, StrongGeneralization
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem, MinRating, Filter
from recpack.util import get_top_K_ranks
from recpack.matrix import InteractionMatrix
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.pipelines import PipelineBuilder
from src.my_bprmf import MyBPRMF, PPAC_BPRMF
from recpack.pipelines import ALGORITHM_REGISTRY, METRIC_REGISTRY
from recpack.algorithms import Popularity, ItemKNN

from recpack.scenarios.splitters import FractionInteractionSplitter

from src.preprocess_block import PreprocessBlock
from src.utils import scores2recommendations, save_metrics_incremental, save_recommendations_incremental
from src.splitting_block import SplittingBlock
from src.new_metrics import ItemGiniK, MeanItemPopularityK
from src.splitter import ItemInterventionSplitter, EqualExposureSplitter, ItemInterventionSplitter1, ItemInterventionSplitter2

from hyperopt import hp
from recpack.pipelines.hyperparameter_optimisation import HyperoptInfo

from src.metrics import calculate_ndcg, calculate_calibrated_recall, calculate_item_gini, calculate_publisher_gini, calculate_mean_popularity

import warnings
warnings.simplefilter("ignore", sp.sparse.SparseEfficiencyWarning)

HOUR = 3600
DAY = 24*HOUR
HALF_HOUR = HOUR/2
MINUTE = 60

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_coverage(recommendations_df, total_items):
    """Calculate catalog coverage"""
    unique_items = recommendations_df['item_id'].nunique()
    return unique_items / total_items * 100

def calculate_novelty(recommendations_df, train_df):
    """Calculate average novelty (inverse popularity)"""
    item_popularity = train_df.groupby('item_id').size()
    total_interactions = len(train_df)
    
    novelties = []
    for item_id in recommendations_df['item_id']:
        if item_id in item_popularity.index:
            pop = item_popularity[item_id] / total_interactions
            novelty = -np.log2(pop)
            novelties.append(novelty)
    
    return np.mean(novelties) if novelties else 0

def get_popularity_bins(train_df, n_bins=3):
    """Divide items into popularity bins (head, torso, tail)"""
    item_popularity = train_df.groupby('item_id').size().sort_values(ascending=False)
    n_items = len(item_popularity)
    
    bins = {
        'head': set(item_popularity.index[:int(n_items * 0.2)]),
        'torso': set(item_popularity.index[int(n_items * 0.2):int(n_items * 0.5)]),
        'tail': set(item_popularity.index[int(n_items * 0.5):])
    }
    return bins, item_popularity

def analyze_popularity_distribution(recommendations_df, popularity_bins):
    """Analyze distribution of recommendations across popularity bins"""
    recommended_items = set(recommendations_df['item_id'].unique())
    
    results = {
        'head': len(recommended_items & popularity_bins['head']),
        'torso': len(recommended_items & popularity_bins['torso']),
        'tail': len(recommended_items & popularity_bins['tail'])
    }
    
    total = sum(results.values())
    if total > 0:
        results = {k: v/total * 100 for k, v in results.items()}
    
    return results

def create_comprehensive_metrics_table(all_recommendations, train_df, test_df, k=10):
    """Create comprehensive metrics comparison table"""
    results = []
    total_items = train_df['item_id'].nunique()
    
    for algo_name, recs_df in all_recommendations.items():
        # Basic metrics
        gini = calculate_item_gini(recs_df, k=k)
        coverage = calculate_coverage(recs_df, total_items)
        novelty = calculate_novelty(recs_df, train_df)
        
        # Popularity distribution
        item_counts = recs_df['item_id'].value_counts()
        avg_recs_per_item = item_counts.mean()
        std_recs_per_item = item_counts.std()
        
        results.append({
            'Algorithm': algo_name,
            'Gini Index': gini,
            'Coverage (%)': coverage,
            'Novelty': novelty,
            'Avg Recs/Item': avg_recs_per_item,
            'Std Recs/Item': std_recs_per_item,
            'Unique Items': len(item_counts)
        })
    
    return pd.DataFrame(results)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_metrics_comparison(metrics_df, output_dir):
    """Create bar chart comparing key metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Multi-Metric Comparison Across Recommender Systems', fontsize=16, y=1.02)
    
    metrics_to_plot = [
        ('NDCGK_10', 'NDCG@10', 'Higher is better'),
        ('RecallK_10', 'Recall@10', 'Higher is better'),
        ('gini_item_10', 'Gini Index@10', 'Lower is better'),
        ('NDCGK_20', 'NDCG@20', 'Higher is better'),
        ('RecallK_20', 'Recall@20', 'Higher is better'),
        ('gini_item_20', 'Gini Index@20', 'Lower is better')
    ]
    
    for idx, (metric, title, direction) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        if metric in metrics_df.columns:
            data = metrics_df[['algorithm', metric]].copy()
            
            # Sort by metric value
            data = data.sort_values(metric, ascending=(direction == 'Lower is better'))
            
            bars = ax.bar(range(len(data)), data[metric])
            
            # Color bars
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data['algorithm'], rotation=45, ha='right')
            ax.set_ylabel(title)
            ax.set_title(f'{title}\n({direction})')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, data[metric])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_item_distribution(all_recommendations, k, output_dir):
    """Plot item distribution histograms for each algorithm"""
    n_algos = len(all_recommendations)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (algo_name, recs_df) in enumerate(all_recommendations.items()):
        if idx >= 4:
            break
            
        ax = axes[idx]
        
        # Get item counts
        item_counts = recs_df['item_id'].value_counts()
        gini = calculate_item_gini(recs_df, k=k)
        
        # Plot histogram
        ax.hist(item_counts.values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(item_counts.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {item_counts.mean():.1f}')
        ax.axvline(item_counts.median(), color='orange', linestyle='--',
                   linewidth=2, label=f'Median: {item_counts.median():.1f}')
        
        ax.set_xlabel('Number of Recommendations per Item', fontsize=10)
        ax.set_ylabel('Number of Items', fontsize=10)
        ax.set_title(f'{algo_name}\nGini: {gini:.4f}', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(all_recommendations), 4):
        axes[idx].axis('off')
    
    plt.suptitle(f'Item Distribution in Top-{k} Recommendations', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'item_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_long_tail_coverage(all_recommendations, train_df, k, output_dir):
    """Plot coverage across popularity bins"""
    popularity_bins, _ = get_popularity_bins(train_df)
    
    results = []
    for algo_name, recs_df in all_recommendations.items():
        dist = analyze_popularity_distribution(recs_df, popularity_bins)
        results.append({
            'Algorithm': algo_name,
            'Head (%)': dist.get('head', 0),
            'Torso (%)': dist.get('torso', 0),
            'Tail (%)': dist.get('tail', 0)
        })
    
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df.plot(x='Algorithm', y=['Head (%)', 'Torso (%)', 'Tail (%)'],
            kind='bar', stacked=True, ax=ax,
            color=['#d62728', '#ff7f0e', '#2ca02c'])
    
    ax.set_ylabel('Percentage of Recommended Items', fontsize=12)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_title(f'Coverage Across Popularity Segments (Top-{k})', fontsize=14, fontweight='bold')
    ax.legend(title='Popularity Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'long_tail_coverage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def plot_accuracy_diversity_tradeoff(metrics_df, comprehensive_metrics, output_dir):
    """Scatter plot showing accuracy-diversity trade-off"""
    # Merge dataframes
    merged = metrics_df.merge(
        comprehensive_metrics[['Algorithm', 'Coverage (%)']],
        left_on='algorithm',
        right_on='Algorithm'
    )
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create scatter plot
    scatter = ax.scatter(
        merged['gini_item_10'],
        merged['NDCGK_10'],
        s=merged['Coverage (%)'] * 20,  # Size by coverage
        alpha=0.6,
        c=range(len(merged)),
        cmap='viridis',
        edgecolors='black',
        linewidth=1.5
    )
    
    # Annotate points
    for idx, row in merged.iterrows():
        ax.annotate(
            row['algorithm'],
            (row['gini_item_10'], row['NDCGK_10']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold'
        )
    
    # Add reference lines
    ax.axhline(merged['NDCGK_10'].mean(), color='gray', 
               linestyle='--', alpha=0.5, linewidth=1, label='Mean NDCG')
    ax.axvline(merged['gini_item_10'].mean(), color='gray',
               linestyle='--', alpha=0.5, linewidth=1, label='Mean Gini')
    
    ax.set_xlabel('Item Gini Index (Concentration) →', fontsize=12)
    ax.set_ylabel('NDCG@10 (Accuracy) →', fontsize=12)
    ax.set_title('Accuracy-Diversity Trade-off\n(Bubble size = Catalog Coverage %)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_diversity_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_hyperparameter_sensitivity(optimization_results, output_dir):
    """Plot hyperparameter sensitivity if optimization was run"""
    if optimization_results is None or len(optimization_results) == 0:
        print("No optimization results to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract relevant columns
    if 'gamma' in optimization_results.columns and 'loss' in optimization_results.columns:
        # Gamma sensitivity
        ax = axes[0]
        scatter = ax.scatter(
            optimization_results['gamma'],
            -optimization_results['loss'],  # Negative because loss is minimized
            alpha=0.6,
            s=100,
            c=range(len(optimization_results)),
            cmap='viridis'
        )
        ax.set_xlabel('Gamma (Local Novelty Weight)', fontsize=12)
        ax.set_ylabel('NDCG@10', fontsize=12)
        ax.set_title('Gamma Hyperparameter Sensitivity', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Beta sensitivity
        if 'beta' in optimization_results.columns:
            ax = axes[1]
            scatter = ax.scatter(
                optimization_results['beta'],
                -optimization_results['loss'],
                alpha=0.6,
                s=100,
                c=range(len(optimization_results)),
                cmap='viridis'
            )
            ax.set_xlabel('Beta (Global Novelty Weight)', fontsize=12)
            ax.set_ylabel('NDCG@10', fontsize=12)
            ax.set_title('Beta Hyperparameter Sensitivity', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_analysis_report(metrics_df, comprehensive_metrics, popularity_distribution, output_dir):
    """Generate a text report summarizing findings"""
    report_path = output_dir / 'analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RECOMMENDATION SYSTEM ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. ACCURACY METRICS (NDCG@10)\n")
        f.write("-" * 40 + "\n")
        for _, row in metrics_df.iterrows():
            f.write(f"{row['algorithm']:20s}: {row['NDCGK_10']:.4f}\n")
        f.write("\n")
        
        f.write("2. FAIRNESS METRICS (Gini Index@10 - Lower is Better)\n")
        f.write("-" * 40 + "\n")
        for _, row in metrics_df.iterrows():
            f.write(f"{row['algorithm']:20s}: {row['gini_item_10']:.4f}\n")
        f.write("\n")
        
        f.write("3. DIVERSITY METRICS\n")
        f.write("-" * 40 + "\n")
        for _, row in comprehensive_metrics.iterrows():
            f.write(f"\n{row['Algorithm']}:\n")
            f.write(f"  Coverage: {row['Coverage (%)']:.2f}%\n")
            f.write(f"  Novelty:  {row['Novelty']:.4f}\n")
            f.write(f"  Unique Items: {row['Unique Items']}\n")
        f.write("\n")
        
        f.write("4. LONG-TAIL COVERAGE\n")
        f.write("-" * 40 + "\n")
        for _, row in popularity_distribution.iterrows():
            f.write(f"\n{row['Algorithm']}:\n")
            f.write(f"  Head:  {row['Head (%)']:.2f}%\n")
            f.write(f"  Torso: {row['Torso (%)']:.2f}%\n")
            f.write(f"  Tail:  {row['Tail (%)']:.2f}%\n")
        f.write("\n")
        
        f.write("5. KEY INSIGHTS\n")
        f.write("-" * 40 + "\n")
        
        # Best accuracy
        best_acc = metrics_df.loc[metrics_df['NDCGK_10'].idxmax()]
        f.write(f"• Best Accuracy: {best_acc['algorithm']} (NDCG@10: {best_acc['NDCGK_10']:.4f})\n")
        
        # Best fairness
        best_fair = metrics_df.loc[metrics_df['gini_item_10'].idxmin()]
        f.write(f"• Best Fairness: {best_fair['algorithm']} (Gini: {best_fair['gini_item_10']:.4f})\n")
        
        # Best coverage
        best_cov = comprehensive_metrics.loc[comprehensive_metrics['Coverage (%)'].idxmax()]
        f.write(f"• Best Coverage: {best_cov['Algorithm']} ({best_cov['Coverage (%)']:.2f}%)\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Analysis report saved to: {report_path}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--algo", type=str, default="bprmf", help="Algorithm to use")
    parser.add_argument("--optimize", action="store_true", help="Enable hyperparameter optimization")
    parser.add_argument("--analyze", action="store_true", help="Generate comprehensive analysis")
    return parser.parse_args()


def hyperparam_tuning():
    optimisation_space = HyperoptInfo(
    {
        "num_components": hp.choice("num_components", [68, 16, 32, 64, 128, 256]),
        "learning_rate": hp.uniform("learning_rate", 0.001, 0.5),
        "lambda_h": hp.uniform("lambda_h", 0.001, 0.1),
        "lambda_w": hp.uniform("lambda_w", 0.001, 0.1),
        "max_epochs": hp.choice("max_epochs", [ 4, 8]),
        "batch_size": hp.choice("batch_size", [128, 256, 512])
    },
    max_evals=50,
    )

    return optimisation_space


def run_pipeline(args) -> pd.DataFrame:
    pb = PreprocessBlock()

    # df_user_review = pd.read_csv("data/user_reviews.csv")

    # df_user_review.drop(columns=["funny", "posted", "last_edited", "helpful", "review"], inplace=True)

    # print(df_user_review.head())



    df_train = pd.concat([
        pd.read_csv("data/train_interactions.csv"), pd.read_csv("data/test_interactions_in.csv")
    ])

    df_test = pd.read_csv("data/test_interactions_in.csv")



    df_train = pb.to_binary(df_train, playtime="playtime")
    df_train = pb.to_positive(df_train, playtime="playtime")
    (train_interaction, test_interaction) , user_mapping, item_mapping = pb.apply_filter(df_train, df_test, user_col="user_id", item_col="item_id", is_test=False)



    print("Train interaction shape:", train_interaction.shape)
    print("Test interaction shape:", test_interaction.shape)

    scenario = WeakGeneralization(0.7, validation=True, seed=42)
    scenario.split(train_interaction)

    # scenario = StrongGeneralization(0.7, validation=True, seed=42)
    # scenario.split(train_interaction)

    # optimisation_space = hyperparam_tuning()
    # params = {
    #     "validation_sample_size": 200,
    #     "predict_topK": 20,
    #     "seed": 42,
    #     "save_best_to_file": False,
    #     "keep_last": True
    # }

    # splitter = FractionInteractionSplitter(in_frac=0.7,    
    #     seed=42
    # )


    # X_train_fit, X_val = splitter.split(train_interaction)

    # print(f"shape of X_VAL ", X_val.shape )
    # print(f"shape of X_train_fit", X_train_fit.shape)

    # X_val = splitter.split(X_val)



    # test = splitter.split(test_interaction)


    # Train / Validation (intervened)
    # splitter_val = ItemInterventionSplitter2(out_frac=0.1, seed=42)
    # X_train, X_val = splitter_val.split(train_interaction)

    # # Validation in / out (optional)
    # splitter_val_io = ItemInterventionSplitter2(out_frac=0.1, seed=43)
    # X_val = splitter_val_io.split(X_val)

    # # Test in / out (intervened)
    # splitter_test = ItemInterventionSplitter2(out_frac=0.1, seed=44)
    # test = splitter_test.split(test_interaction)

    # splitter = EqualExposureSplitter(
    # test_frac=0.1,
    # validation_frac=0.1,
    # seed=42
    # )
    # train, val, test = splitter.split(train_interaction)

    # check_item_balance(test[0])
    

 
    params, hyper_params = bprmf(include_hyperparams=args.optimize)

    params_p, hyper_params_p = ppac(include_hyperparams=args.optimize)

    

    ALGORITHM_REGISTRY.register("PPAC_BPRMF", PPAC_BPRMF)
    ALGORITHM_REGISTRY.register("MyBPRMF", MyBPRMF)

    METRIC_REGISTRY.register("gini_item", ItemGiniK)
    # METRIC_REGISTRY.register("gini_pub", PublisherGiniK)
    item_popularity = np.array(train_interaction.values.sum(axis=0)).ravel()

    METRIC_REGISTRY.register(
        "MeanPopK",
        lambda K: MeanItemPopularityK(K, item_popularity)
    )


    pb = PipelineBuilder("results")
    
    pb.add_algorithm("PPAC_BPRMF", params=params_p, optimisation_info=hyper_params_p)
    pb.add_algorithm("MyBPRMF", params=params, optimisation_info=hyper_params)
    # pb.add_algorithm("ItemKNN", params={"K":200, "similarity":"cosine"})
    # pb.add_algorithm("EASE", params={"density": 0.8})
    pb.add_algorithm("Popularity", params={"K": 10})
    pb.set_data_from_scenario(scenario)
    # pb.set_full_training_data(train_interaction)
    # pb.set_validation_data(X_val)
    # pb.set_validation_training_data(X_val[0])
    # pb.set_test_data(test)
    if args.optimize:
        pb.set_optimisation_metric("NDCGK", 10)
    pb.add_metric("NDCGK", [10, 20])
    pb.add_metric("gini_item", [10,20])
    pb.add_metric("RecallK", [10, 20])
    pb.add_metric("MeanPopK", [10, 20])
    # pb.add_metric("gini_pub", [10,20])
    # pb.add_metric("gini", [])
    pipe = pb.build()
    scores = pipe.run() 
    df_metrics = pipe.get_metrics()
    df_optimization = None
    if args.optimize:
        df_optimization = pipe.optimisation_results
    pipe.save_metrics()


    print("Metrics:")
    print(df_metrics)


    save_metrics_incremental("metrics/", df_metrics, df_optimization,prefix=f"{args.algo}")


    # # model = MyBPRMF(**bpr_params)
    # # model.fit(X_train_fit, validation_data=X_val)
    # # scores = model.predict(test_interaction)
    all_recommendations = {}
    for algorithm, score in scores:
        df_recos = scores2recommendations(
                        score,
                        test_interaction.binary_values,
                        recommendation_count=10,
                        user_id_mapping=user_mapping,
                        item_id_mapping=item_mapping,
                        prevent_history_recos=True
                    )
    
        print(f"item_gini_{algorithm}", calculate_item_gini(df_recos,k=10))
        print(f"calculate mean pop {algorithm}", calculate_mean_popularity(df_recos,df_train,10))

        all_recommendations[algorithm] = df_recos
        df_recos.drop(columns=["rank"], inplace=True)

        save_recommendations_incremental("output", df_recos, algorithm=algorithm)
    
    
    plot_exposure(
        all_recommendations,
        title="Item Exposure Distribution (Top-10)"
    )


    # return df_recos


    if args.analyze:
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE ANALYSIS")
        print("="*80)
        
        output_dir = Path("analysis_output")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Create comprehensive metrics table
        print("\n1. Creating comprehensive metrics table...")
        comprehensive_metrics = create_comprehensive_metrics_table(
            all_recommendations, df_train, df_test, k=10
        )
        print(comprehensive_metrics)
        comprehensive_metrics.to_csv(output_dir / 'comprehensive_metrics.csv', index=False)
        
        # 2. Plot metrics comparison
        print("2. Plotting metrics comparison...")
        plot_metrics_comparison(df_metrics, output_dir)
        
        # 3. Plot item distribution
        print("3. Plotting item distribution...")
        plot_item_distribution(all_recommendations, k=10, output_dir=output_dir)
        
        # 4. Plot long-tail coverage
        print("4. Plotting long-tail coverage...")
        popularity_dist = plot_long_tail_coverage(
            all_recommendations, df_train, k=10, output_dir=output_dir
        )
        popularity_dist.to_csv(output_dir / 'popularity_distribution.csv', index=False)
        
        # 5. Plot accuracy-diversity trade-off
        print("5. Plotting accuracy-diversity trade-off...")
        plot_accuracy_diversity_tradeoff(df_metrics, comprehensive_metrics, output_dir)
        
        # 6. Plot hyperparameter sensitivity (if optimization was run)
        if args.optimize and df_optimization is not None:
            print("6. Plotting hyperparameter sensitivity...")
            plot_hyperparameter_sensitivity(df_optimization, output_dir)
        
        # 7. Generate text report
        print("7. Generating analysis report...")
        generate_analysis_report(
            df_metrics, comprehensive_metrics, popularity_dist, output_dir
        )
        
        print("\n" + "="*80)
        print(f"✓ All analysis outputs saved to: {output_dir}/")
        print("="*80)
        print("\nGenerated files:")
        print("  • metrics_comparison.png")
        print("  • item_distribution.png")
        print("  • long_tail_coverage.png")
        print("  • accuracy_diversity_tradeoff.png")
        if args.optimize:
            print("  • hyperparameter_sensitivity.png")
        print("  • comprehensive_metrics.csv")
        print("  • popularity_distribution.csv")
        print("  • analysis_report.txt")
        print()


def check_item_balance(X):
    counts = np.array(X.values.sum(axis=0)).ravel()
    print("min / max / std:", counts.min(), counts.max(), counts.std())


def compute_item_exposure(df_recos: pd.DataFrame):
    # df_recos must have column "item_id"
    return df_recos["item_id"].value_counts().sort_values(ascending=False)


def plot_exposure(models_recos: dict, title="Item Exposure Distribution"):
    plt.figure(figsize=(7,5))

    for name, df in models_recos.items():
        exposure = compute_item_exposure(df)
        exposure = exposure.values / exposure.sum()  # normalize
        plt.plot(np.cumsum(exposure), label=name)

    plt.xlabel("Fraction of Items")
    plt.ylabel("Cumulative Exposure")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



def bprmf(include_hyperparams=True):
    """
    BPRMF with partially fixed hyperparameters and a constrained
    Hyperopt search space to keep optimization time reasonable.

    Args:
        include_hyperparams (bool): 
            - If True, returns (fixed_params, hyperopt_space) for hyperparameter tuning.
            - If False, returns default training parameters ready for training.

    Returns:
        dict or tuple: 
            - If include_hyperparams=True: (fixed_params, hyperopt_space)
            - If include_hyperparams=False: default training parameters (bpr_params)
    """

    if include_hyperparams:
        # ------------------------------------------------------------------
        # Fixed BPRMF parameters
        # ------------------------------------------------------------------
        fixed_params = {
            "max_epochs": 5,
            "lambda_h": 0.0000455332,           
            "lambda_w": 0.0001938869,
            "save_best_to_file": False,
            "keep_last": False,
            "seed": 42,
            "validation_sample_size": 200,
            "predict_topK": 20,
        }

        # ------------------------------------------------------------------
        # Hyperopt search space (only for tunable parameters)
        # ------------------------------------------------------------------
        hyperopt_space = HyperoptInfo(
            {
                "num_components": hp.choice(
                    "num_components",
                    [64, 128, 256]
                ),
                "learning_rate": hp.choice(
                    "learning_rate", [0.001, 0.005, 0.01]
                ),
                "batch_size": hp.choice(
                    "batch_size",
                    [256, 512]
                ),
            },
            timeout=MINUTE*10
        )

        return fixed_params, hyperopt_space

    else:
        # ------------------------------------------------------------------
        # Default parameters for training (no hyperparameter search)
        # ------------------------------------------------------------------
        bpr_params = {
            "num_components": 128,       # Reduce - easier to learn
            "learning_rate": 0.005,      # Higher LR
            "lambda_h": 0.0000455332,           
            "lambda_w": 0.0001938869,
            "max_epochs": 8,           
            "batch_size": 512,           # Larger batches
            "save_best_to_file": False,         
            "keep_last": False,         # Don't keep last model
            "seed": 42,
            "validation_sample_size": 200,
            "predict_topK": 100,
        }
        return bpr_params, None



def ppac(include_hyperparams=True):
    """
    PPAC-BPRMF with partially fixed hyperparameters and a constrained
    Hyperopt search space for reasonable optimization time.

    Args:
        include_hyperparams (bool): 
            - If True, returns (fixed_params, hyperopt_space) for hyperparameter tuning.
            - If False, returns default training parameters ready for training.

    Returns:
        dict or tuple: 
            - If include_hyperparams=True: (fixed_params, hyperopt_space)
            - If include_hyperparams=False: default training parameters (ppac_params)
    """

    if include_hyperparams:
        # ------------------------------------------------------------------
        # Fixed parameters
        # ------------------------------------------------------------------
        fixed_params = {
            "max_epochs": 4,
            "batch_size":256,
            "learning_rate":0.001,
            "num_components":256,
            "max_epochs":5,
            "lambda_h":0.0000455332,
            "lambda_w":0.0001938869,
            "reg_coe": 1e-3,
            "l2_coe":1e-4,
            "save_best_to_file": False,
            "keep_last": False,
            "seed": 42,
            "validation_sample_size": 200,
            "predict_topK": 100,
        }

        # ------------------------------------------------------------------
        # Hyperopt search space (tunable parameters)
        # ------------------------------------------------------------------
        hyperopt_space = HyperoptInfo(
            {
                "gamma": hp.choice("gamma", [128, 256, 512]),
                "beta": hp.choice("beta", [-128, -512, -1024]),
            },
            timeout=MINUTE*10
        )

        return fixed_params, hyperopt_space

    else:
        # ------------------------------------------------------------------
        # Default parameters for training (no hyperparameter search)
        # ------------------------------------------------------------------
        ppac_params = {
            "gamma": 1024,                 # default novelty weight
            "beta": -514,                  # default novelty weight             
            "reg_coe": 1e-3,
            "l2_coe":1e-4,             
            "save_best_to_file": False,         
            "keep_last": False,            
            "seed": 42,
            "validation_sample_size": 200,
            "predict_topK": 100,
            "num_components": 128,       # Reduce - easier to learn
            "learning_rate": 0.005,       # Higher LR
            "lambda_h": 0.0,           
            "lambda_w": 0.0,
            "max_epochs": 8,           
            "batch_size": 512,           # Larger batches
            "save_best_to_file": False,         
            "keep_last": False,         # Don't keep last model
            "seed": 42,
            "validation_sample_size": 200,
            "predict_topK": 100,

        }
        return ppac_params, None


    


def run_mostpop_pipeline():
    pb = PreprocessBlock()

    df_train = pd.read_csv("data/train_interactions.csv")
    df_train = pb.to_binary(df_train, "playtime")
    df_train = pb.to_positive(df_train, "playtime")

    
    
    train_interaction = pb.apply_filter(
        df_train, 
        user_col="user_id", 
        item_col="item_id", 
        is_test=False
    )

    df_test_in = pd.read_csv("data/test_interactions_in.csv")
    df_test_in = pb.to_binary(df_test_in, "playtime")
    df_test_in = pb.to_positive(df_test_in, "playtime")

    

    test_interaction = pb.apply_filter(
        df_test_in,
        user_col="user_id",
        item_col="item_id",
        is_test=True
    )

    model = Popularity()
    
    model.fit(train_interaction)
    
    scores = model.predict(test_interaction)

    df_recos = scores2recommendations(
        scores,
        test_interaction.binary_values,
        recommendation_count=20,
        prevent_history_recos=True
    )

    df_recos.drop(columns=["rank"], inplace=True)

    return df_recos

def main():
    args = parse_args()


    run_pipeline(args)




if __name__ == "__main__":
    main()
