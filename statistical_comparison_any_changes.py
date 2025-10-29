import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(ai_file: str, human_file: str):
    """Load AI and Human PR data."""
    print("Loading data...")
    ai_df = pd.read_csv(ai_file)
    human_df = pd.read_csv(human_file)
    
    print(f"AI PRs: {len(ai_df):,}")
    print(f"Human PRs: {len(human_df):,}")
    
    return ai_df, human_df

def calculate_any_metrics(df, label: str):
    """Calculate various any-related metrics for each PR."""
    if 'any_additions' not in df.columns or 'any_removals' not in df.columns:
        print(f"Warning: Missing any_additions or any_removals columns in {label}")
        return None
    
    metrics = {}
    
    # Net any changes (additions - removals)
    metrics['net_any_changes'] = df['any_additions'] - df['any_removals']
    
    # Total any usage (additions + removals)
    metrics['total_any_usage'] = df['any_additions'] + df['any_removals']
    
    # Just additions
    metrics['any_additions'] = df['any_additions']
    
    # Just removals
    metrics['any_removals'] = df['any_removals']
    
    print(f"\n{label} Statistics:")
    print(f"  [Net Any Changes] Mean: {metrics['net_any_changes'].mean():.4f}, Median: {metrics['net_any_changes'].median():.4f}")
    print(f"  [Total Any Usage] Mean: {metrics['total_any_usage'].mean():.4f}, Median: {metrics['total_any_usage'].median():.4f}")
    print(f"  [Any Additions] Mean: {metrics['any_additions'].mean():.4f}, Median: {metrics['any_additions'].median():.4f}")
    print(f"  [Any Removals] Mean: {metrics['any_removals'].mean():.4f}, Median: {metrics['any_removals'].median():.4f}")
    
    return metrics

def perform_statistical_tests(ai_data, human_data):
    """Perform t-test and Mann-Whitney U test."""
    print("\n" + "="*80)
    print("Statistical Tests")
    print("="*80)
    
    # Remove NaN values
    ai_data = ai_data.dropna()
    human_data = human_data.dropna()
    
    print(f"\nAI sample size: {len(ai_data):,}")
    print(f"Human sample size: {len(human_data):,}")
    
    # T-test (assumes normal distribution)
    print("\n[1] Independent Samples T-test:")
    t_stat, t_pvalue = stats.ttest_ind(ai_data, human_data)
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {t_pvalue:.6f}")
    print(f"   Significant at α=0.05: {'Yes' if t_pvalue < 0.05 else 'No'}")
    
    # Mann-Whitney U test (non-parametric, doesn't assume normal distribution)
    print("\n[2] Mann-Whitney U test (Wilcoxon rank-sum test):")
    u_stat, u_pvalue = stats.mannwhitneyu(ai_data, human_data, alternative='two-sided')
    print(f"   U-statistic: {u_stat:.4f}")
    print(f"   p-value: {u_pvalue:.6f}")
    print(f"   Significant at α=0.05: {'Yes' if u_pvalue < 0.05 else 'No'}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(ai_data) - 1) * ai_data.var() + 
                          (len(human_data) - 1) * human_data.var()) / 
                         (len(ai_data) + len(human_data) - 2))
    cohens_d = (ai_data.mean() - human_data.mean()) / pooled_std
    print(f"\n   Cohen's d (effect size): {cohens_d:.4f}")
    
    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    print(f"   Effect size interpretation: {effect_size}")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"   AI mean: {ai_data.mean():.4f}")
    print(f"   Human mean: {human_data.mean():.4f}")
    print(f"   Difference: {ai_data.mean() - human_data.mean():.4f}")
    
    results = {
        't_test': {'statistic': t_stat, 'pvalue': t_pvalue},
        'mann_whitney': {'statistic': u_stat, 'pvalue': u_pvalue},
        'cohens_d': cohens_d,
        'ai_mean': ai_data.mean(),
        'human_mean': human_data.mean(),
        'ai_median': ai_data.median(),
        'human_median': human_data.median(),
        'ai_count': len(ai_data),
        'human_count': len(human_data),
    }
    
    return results

def create_plots(ai_data, human_data, save_path: str = 'any_changes_comparison.png'):
    """Create visualization plots."""
    print("\nCreating visualizations...")
    
    # Prepare data for plotting
    plot_data = pd.DataFrame({
        'Group': ['AI'] * len(ai_data) + ['Human'] * len(human_data),
        'Net Any Changes': pd.concat([ai_data, human_data], ignore_index=True)
    })
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparison of Net Any Changes (any_additions - any_removals)\nAI vs Human', 
                 fontsize=16, fontweight='bold')
    
    # 1. Box plot
    ax1 = axes[0, 0]
    sns.boxplot(data=plot_data, x='Group', y='Net Any Changes', ax=ax1)
    ax1.set_title('Box Plot', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Violin plot
    ax2 = axes[0, 1]
    sns.violinplot(data=plot_data, x='Group', y='Net Any Changes', ax=ax2)
    ax2.set_title('Violin Plot (Distribution)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram overlay
    ax3 = axes[1, 0]
    ax3.hist(ai_data, bins=50, alpha=0.6, label='AI', color='skyblue', density=True)
    ax3.hist(human_data, bins=50, alpha=0.6, label='Human', color='salmon', density=True)
    ax3.set_xlabel('Net Any Changes')
    ax3.set_ylabel('Density')
    ax3.set_title('Histogram Overlay', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Bar plot with means and error bars
    ax4 = axes[1, 1]
    means = [ai_data.mean(), human_data.mean()]
    stds = [ai_data.std(), human_data.std()]
    groups = ['AI', 'Human']
    colors = ['skyblue', 'salmon']
    
    bars = ax4.bar(groups, means, yerr=stds, capsize=10, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Mean Net Any Changes')
    ax4.set_title('Mean with Standard Deviation', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Saved plot to {save_path}")
    plt.close()

def save_results(results, save_path: str = 'statistical_comparison_results.csv'):
    """Save statistical test results to CSV."""
    results_df = pd.DataFrame([{
        'Test': 'T-test',
        'Statistic': results['t_test']['statistic'],
        'P-value': results['t_test']['pvalue'],
        'Significant (α=0.05)': 'Yes' if results['t_test']['pvalue'] < 0.05 else 'No'
    }, {
        'Test': 'Mann-Whitney U',
        'Statistic': results['mann_whitney']['statistic'],
        'P-value': results['mann_whitney']['pvalue'],
        'Significant (α=0.05)': 'Yes' if results['mann_whitney']['pvalue'] < 0.05 else 'No'
    }])
    
    # Add summary statistics
    summary_df = pd.DataFrame([{
        'Metric': 'Cohen\'s d',
        'Value': results['cohens_d'],
        'Interpretation': 'negligible' if abs(results['cohens_d']) < 0.2 else 
                         'small' if abs(results['cohens_d']) < 0.5 else
                         'medium' if abs(results['cohens_d']) < 0.8 else 'large'
    }, {
        'Metric': 'AI Mean',
        'Value': results['ai_mean'],
        'Interpretation': ''
    }, {
        'Metric': 'Human Mean',
        'Value': results['human_mean'],
        'Interpretation': ''
    }, {
        'Metric': 'AI Median',
        'Value': results['ai_median'],
        'Interpretation': ''
    }, {
        'Metric': 'Human Median',
        'Value': results['human_median'],
        'Interpretation': ''
    }, {
        'Metric': 'AI Count',
        'Value': results['ai_count'],
        'Interpretation': ''
    }, {
        'Metric': 'Human Count',
        'Value': results['human_count'],
        'Interpretation': ''
    }])
    
    # Combine and save
    final_df = pd.concat([results_df, summary_df], ignore_index=True)
    final_df.to_csv(save_path, index=False)
    print(f"   Saved results to {save_path}")


def main():
    print("="*80)
    print("Statistical Comparison: Any Usage Metrics (AI vs Human)")
    print("="*80)
    
    # Load data
    ai_df, human_df = load_data('ai_baseline_results.csv', 'human_baseline_results.csv')
    
    # Calculate all metrics
    ai_metrics = calculate_any_metrics(ai_df, 'AI')
    human_metrics = calculate_any_metrics(human_df, 'Human')
    
    if ai_metrics is None or human_metrics is None:
        print("Error: Could not calculate metrics. Please check column names.")
        return
    
    # Define which metrics to test
    metrics_to_test = {
        'net_any_changes': 'Net Any Changes (Additions - Removals)',
        'total_any_usage': 'Total Any Usage (Additions + Removals)',
        'any_additions': 'Any Additions',
        'any_removals': 'Any Removals'
    }
    
    all_results = []
    
    # Perform tests for each metric
    for metric_name, metric_label in metrics_to_test.items():
        print(f"\n{'='*80}")
        print(f"Testing: {metric_label}")
        print(f"{'='*80}")
        
        ai_data = ai_metrics[metric_name]
        human_data = human_metrics[metric_name]
        
        # Perform statistical tests
        results = perform_statistical_tests(ai_data, human_data)
        results['metric'] = metric_name
        results['metric_label'] = metric_label
        all_results.append(results)
        
        # Create plots for net_any_changes (main metric)
        if metric_name == 'net_any_changes':
            create_plots(ai_data, human_data, f'any_changes_comparison_{metric_name}.png')
        else:
            create_plots(ai_data, human_data, f'any_changes_comparison_{metric_name}.png')
    
    # Save all results
    save_all_results(all_results)
    
    print("\n" + "="*80)
    print("Analysis completed!")
    print("="*80)


def save_all_results(all_results, save_path: str = 'statistical_comparison_results.csv'):
    """Save statistical test results for all metrics to CSV."""
    rows = []
    
    for result in all_results:
        metric = result['metric_label']
        
        # T-test row
        rows.append({
            'Metric': metric,
            'Test': 'T-test',
            'Statistic': result['t_test']['statistic'],
            'P-value': result['t_test']['pvalue'],
            'Significant (α=0.05)': 'Yes' if result['t_test']['pvalue'] < 0.05 else 'No',
            "Cohen's d": result['cohens_d'],
            'Effect Size': 'negligible' if abs(result['cohens_d']) < 0.2 else 
                          'small' if abs(result['cohens_d']) < 0.5 else
                          'medium' if abs(result['cohens_d']) < 0.8 else 'large',
            'AI Mean': result['ai_mean'],
            'Human Mean': result['human_mean'],
            'AI Median': result['ai_median'],
            'Human Median': result['human_median']
        })
        
        # Mann-Whitney U row
        rows.append({
            'Metric': metric,
            'Test': 'Mann-Whitney U',
            'Statistic': result['mann_whitney']['statistic'],
            'P-value': result['mann_whitney']['pvalue'],
            'Significant (α=0.05)': 'Yes' if result['mann_whitney']['pvalue'] < 0.05 else 'No',
            "Cohen's d": result['cohens_d'],
            'Effect Size': 'negligible' if abs(result['cohens_d']) < 0.2 else 
                          'small' if abs(result['cohens_d']) < 0.5 else
                          'medium' if abs(result['cohens_d']) < 0.8 else 'large',
            'AI Mean': result['ai_mean'],
            'Human Mean': result['human_mean'],
            'AI Median': result['ai_median'],
            'Human Median': result['human_median']
        })
    
    results_df = pd.DataFrame(rows)
    results_df.to_csv(save_path, index=False)
    print(f"\n   Saved all results to {save_path}")


if __name__ == '__main__':
    main()

