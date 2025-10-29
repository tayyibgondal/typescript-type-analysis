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

def get_advanced_feature_columns(df):
    """Get all advanced feature count columns."""
    # List of advanced feature columns based on ADVANCED_TYPE_PATTERNS_REFINED
    advanced_features = [
        'generics_count',
        'conditional_type_count',
        'infer_count',
        'mapped_type_count',
        'key_remap_count',
        'template_literal_type_count',
        'satisfies_count',
        'as_const_count',
        'non_null_assertion_count',
        'type_guard_is_count',
        'keyof_count',
        'typeof_count',
        'in_operator_count',
        'intersection_type_count',
        'union_type_count',
        'indexed_access_count',
        'utility_type_usage_count',
        'recursive_type_count',
        'const_assertion_count',
        'discriminated_union_count',
        'unique_symbol_count',
        'readonly_modifier_count',
        'optional_property_count'
    ]
    
    # Filter to only columns that exist in the dataframe
    existing_features = [col for col in advanced_features if col in df.columns]
    
    return existing_features

def test_normality(data, feature_name: str, group: str):
    """
    Test for normality using Shapiro-Wilk test.
    Returns (statistic, pvalue, is_normal)
    """
    # Remove zeros and NaN values for meaningful normality test
    non_zero_data = data[data > 0].dropna()
    
    if len(non_zero_data) < 3:
        # Too few non-zero values for meaningful test
        return None, None, None
    
    # Limit sample size for Shapiro-Wilk (max 5000)
    if len(non_zero_data) > 5000:
        sample_data = non_zero_data.sample(n=5000, random_state=42)
    else:
        sample_data = non_zero_data
    
    try:
        stat, pvalue = stats.shapiro(sample_data)
        is_normal = pvalue > 0.05  # Typically use α=0.05 for normality
        return stat, pvalue, is_normal
    except Exception as e:
        print(f"    Warning: Normality test failed for {group} {feature_name}: {e}")
        return None, None, None

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    # Remove NaN values
    group1 = group1.dropna()
    group2 = group2.dropna()
    
    if len(group1) == 0 or len(group2) == 0:
        return None
    
    # Calculate pooled standard deviation
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    
    # Handle edge cases
    if n1 < 2 or n2 < 2:
        return None
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return None
    
    cohens_d = (group1.mean() - group2.mean()) / pooled_std
    return cohens_d

def interpret_effect_size(d):
    """Interpret Cohen's d effect size."""
    if d is None:
        return None
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def perform_statistical_test(ai_data, human_data, feature_name: str, use_parametric: bool):
    """
    Perform appropriate statistical test based on normality assumption.
    
    Args:
        ai_data: AI group data
        human_data: Human group data
        feature_name: Name of the feature being tested
        use_parametric: Whether to use parametric (t-test) or non-parametric (Mann-Whitney U) test
    
    Returns:
        Dictionary with test results
    """
    # Remove NaN values
    ai_data = ai_data.dropna()
    human_data = human_data.dropna()
    
    if len(ai_data) == 0 or len(human_data) == 0:
        return None
    
    # Perform appropriate test
    if use_parametric:
        # Independent samples t-test
        try:
            t_stat, pvalue = stats.ttest_ind(ai_data, human_data)
            test_name = "T-test"
            test_stat = t_stat
        except Exception as e:
            print(f"    Warning: T-test failed: {e}")
            return None
    else:
        # Mann-Whitney U test (Wilcoxon rank-sum test)
        try:
            u_stat, pvalue = stats.mannwhitneyu(ai_data, human_data, alternative='two-sided')
            test_name = "Mann-Whitney U"
            test_stat = u_stat
        except Exception as e:
            print(f"    Warning: Mann-Whitney U test failed: {e}")
            return None
    
    # Calculate Cohen's d
    cohens_d = calculate_cohens_d(ai_data, human_data)
    effect_size = interpret_effect_size(cohens_d)
    
    results = {
        'feature': feature_name,
        'test_used': test_name,
        'test_statistic': test_stat,
        'pvalue': pvalue,
        'significant': pvalue < 0.05,
        'cohens_d': cohens_d,
        'effect_size': effect_size,
        'ai_mean': ai_data.mean(),
        'human_mean': human_data.mean(),
        'ai_median': ai_data.median(),
        'human_median': human_data.median(),
        'ai_std': ai_data.std(),
        'human_std': human_data.std(),
        'ai_count': len(ai_data),
        'human_count': len(human_data),
        'use_parametric': use_parametric
    }
    
    return results

def analyze_feature(ai_df, human_df, feature_name: str):
    """Analyze a single advanced feature."""
    if feature_name not in ai_df.columns or feature_name not in human_df.columns:
        return None
    
    ai_data = ai_df[feature_name]
    human_data = human_df[feature_name]
    
    # Test normality for both groups
    ai_stat, ai_p, ai_normal = test_normality(ai_data, feature_name, 'AI')
    human_stat, human_p, human_normal = test_normality(human_data, feature_name, 'Human')
    
    # Decision: Use parametric if both groups are normal, else use non-parametric
    use_parametric = False
    if ai_normal is not None and human_normal is not None:
        use_parametric = ai_normal and human_normal
        normality_decision = "Both groups normal → T-test" if use_parametric else "Not both normal → Mann-Whitney U"
    else:
        normality_decision = "Normality test inconclusive → Mann-Whitney U"
        use_parametric = False
    
    # Perform statistical test
    results = perform_statistical_test(ai_data, human_data, feature_name, use_parametric)
    
    if results:
        results['ai_normality_p'] = ai_p
        results['human_normality_p'] = human_p
        results['ai_normal'] = ai_normal
        results['human_normal'] = human_normal
        results['normality_decision'] = normality_decision
    
    return results

def main():
    print("="*80)
    print("Statistical Comparison: Advanced Type Features (AI vs Human)")
    print("="*80)
    
    # Load data
    ai_df, human_df = load_data('ai_baseline_results.csv', 'human_baseline_results.csv')
    
    # Get advanced feature columns
    features = get_advanced_feature_columns(ai_df)
    print(f"\nFound {len(features)} advanced features to analyze")
    
    # Analyze each feature
    all_results = []
    
    for i, feature in enumerate(features, 1):
        print(f"\n[{i}/{len(features)}] Analyzing: {feature}")
        print("-" * 60)
        
        results = analyze_feature(ai_df, human_df, feature)
        
        if results:
            all_results.append(results)
            
            # Print summary
            print(f"  Normality: {results['normality_decision']}")
            print(f"  Test: {results['test_used']}")
            print(f"  P-value: {results['pvalue']:.6f}")
            print(f"  Significant: {'Yes' if results['significant'] else 'No'}")
            print(f"  Cohen's d: {results['cohens_d']:.4f} ({results['effect_size']})")
            print(f"  AI mean: {results['ai_mean']:.4f}, Human mean: {results['human_mean']:.4f}")
        else:
            print(f"  Warning: Could not analyze {feature}")
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Reorder columns for better readability
        column_order = [
            'feature', 'test_used', 'test_statistic', 'pvalue', 'significant',
            'cohens_d', 'effect_size',
            'ai_mean', 'human_mean', 'ai_median', 'human_median',
            'ai_std', 'human_std',
            'ai_count', 'human_count',
            'ai_normal', 'human_normal', 'ai_normality_p', 'human_normality_p',
            'normality_decision', 'use_parametric'
        ]
        
        results_df = results_df[[col for col in column_order if col in results_df.columns]]
        results_df = results_df.sort_values('pvalue')
        
        results_df.to_csv('advanced_features_statistical_comparison.csv', index=False)
        print(f"\n{'='*80}")
        print(f"Results saved to advanced_features_statistical_comparison.csv")
        print(f"Total features analyzed: {len(all_results)}")
        
        # Summary statistics
        significant_count = results_df['significant'].sum()
        print(f"Significant differences (p < 0.05): {significant_count}/{len(all_results)}")
        print(f"Features with medium/large effect size: {len(results_df[results_df['effect_size'].isin(['medium', 'large'])])}")
        
    print("\n" + "="*80)
    print("Analysis completed!")
    print("="*80)

if __name__ == '__main__':
    main()

