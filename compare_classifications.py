"""
Compare Regex vs OpenAI Classification Results

This script compares the results from:
1. Regex-based classification (extract_typescript_type_prs.py)
2. OpenAI-powered classification (openai_type_classifier.py)

Shows agreement rates, disagreements, and confidence analysis.
"""

import pandas as pd
import json

def load_datasets():
    """Load both classification results"""
    print("="*80)
    print("Comparing Regex vs OpenAI Classifications")
    print("="*80)
    
    print("\n📂 Loading datasets...")
    
    # Load regex results
    regex_df = pd.read_csv('typescript_type_related_agentic_prs.csv')
    print(f"   ✅ Regex results: {len(regex_df):,} PRs")
    
    # Load OpenAI results
    try:
        openai_df = pd.read_csv('ai_classified_type_prs.csv')
        print(f"   ✅ OpenAI results: {len(openai_df):,} PRs")
    except FileNotFoundError:
        print("   ❌ OpenAI results not found. Run openai_type_classifier.py first!")
        return None, None
    
    return regex_df, openai_df

def compare_results(regex_df, openai_df):
    """Compare classification results"""
    
    # Merge datasets on PR ID
    merged = regex_df.merge(
        openai_df[['id', 'final_is_type_related', 'classifier_confidence', 'validation_result']],
        on='id',
        how='inner',
        suffixes=('_regex', '_openai')
    )
    
    print(f"\n📊 Comparison Statistics")
    print("="*80)
    print(f"Total PRs compared: {len(merged):,}")
    
    # Agreement analysis
    regex_positive = merged['is_type_related'] if 'is_type_related' in merged else merged['is_type_in_title'] | merged['is_type_in_body']
    openai_positive = merged['final_is_type_related']
    
    both_positive = (regex_positive) & (openai_positive)
    both_negative = (~regex_positive) & (~openai_positive)
    regex_only = (regex_positive) & (~openai_positive)
    openai_only = (~regex_positive) & (openai_positive)
    
    agreement = both_positive.sum() + both_negative.sum()
    
    print(f"\n🤝 Agreement:")
    print(f"   Both say TYPE-RELATED: {both_positive.sum():,} ({both_positive.sum()/len(merged)*100:.1f}%)")
    print(f"   Both say NOT type-related: {both_negative.sum():,} ({both_negative.sum()/len(merged)*100:.1f}%)")
    print(f"   Total agreement: {agreement:,} ({agreement/len(merged)*100:.1f}%)")
    
    print(f"\n❌ Disagreements:")
    print(f"   Regex YES, OpenAI NO: {regex_only.sum():,} ({regex_only.sum()/len(merged)*100:.1f}%)")
    print(f"   Regex NO, OpenAI YES: {openai_only.sum():,} ({openai_only.sum()/len(merged)*100:.1f}%)")
    
    # Confidence analysis for disagreements
    if 'classifier_confidence' in merged.columns:
        print(f"\n📈 Confidence Analysis (Disagreements):")
        
        if regex_only.sum() > 0:
            avg_conf_regex_only = merged[regex_only]['classifier_confidence'].mean()
            print(f"   Regex YES, OpenAI NO - Avg OpenAI confidence: {avg_conf_regex_only:.2f}")
            print(f"   (Low confidence suggests regex false positives)")
        
        if openai_only.sum() > 0:
            avg_conf_openai_only = merged[openai_only]['classifier_confidence'].mean()
            print(f"   Regex NO, OpenAI YES - Avg OpenAI confidence: {avg_conf_openai_only:.2f}")
            print(f"   (High confidence suggests regex false negatives)")
    
    # Show examples
    print(f"\n" + "="*80)
    print("📋 Example Disagreements")
    print("="*80)
    
    # Regex YES, OpenAI NO (potential false positives)
    if regex_only.sum() > 0:
        print(f"\n🔴 Regex YES, OpenAI NO (Potential False Positives):")
        examples = merged[regex_only].head(3)
        for idx, row in examples.iterrows():
            print(f"\n  PR #{row['number']}: {row['title'][:70]}")
            print(f"  Agent: {row['agent']} | OpenAI confidence: {row.get('classifier_confidence', 0):.2f}")
            print(f"  URL: {row['html_url']}")
    
    # Regex NO, OpenAI YES (potential false negatives)
    if openai_only.sum() > 0:
        print(f"\n🟢 Regex NO, OpenAI YES (Potential False Negatives):")
        examples = merged[openai_only].head(3)
        for idx, row in examples.iterrows():
            print(f"\n  PR #{row['number']}: {row['title'][:70]}")
            print(f"  Agent: {row['agent']} | OpenAI confidence: {row.get('classifier_confidence', 0):.2f}")
            print(f"  URL: {row['html_url']}")
    
    return merged

def create_comparison_report(merged):
    """Create a detailed comparison report"""
    
    report_file = 'classification_comparison_report.json'
    
    regex_positive = merged['is_type_related'] if 'is_type_related' in merged else merged['is_type_in_title'] | merged['is_type_in_body']
    openai_positive = merged['final_is_type_related']
    
    report = {
        'total_prs': int(len(merged)),
        'agreement': {
            'both_positive': int(((regex_positive) & (openai_positive)).sum()),
            'both_negative': int(((~regex_positive) & (~openai_positive)).sum()),
            'total': int(((regex_positive == openai_positive)).sum()),
            'percentage': float(((regex_positive == openai_positive)).sum() / len(merged) * 100)
        },
        'disagreements': {
            'regex_only': int(((regex_positive) & (~openai_positive)).sum()),
            'openai_only': int(((~regex_positive) & (openai_positive)).sum()),
            'total': int(((regex_positive != openai_positive)).sum()),
            'percentage': float(((regex_positive != openai_positive)).sum() / len(merged) * 100)
        },
        'regex_stats': {
            'total_positive': int(regex_positive.sum()),
            'percentage': float(regex_positive.sum() / len(merged) * 100)
        },
        'openai_stats': {
            'total_positive': int(openai_positive.sum()),
            'percentage': float(openai_positive.sum() / len(merged) * 100),
            'avg_confidence': float(merged['classifier_confidence'].mean()) if 'classifier_confidence' in merged else 0
        }
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Saved comparison report to {report_file}")

def main():
    """Main entry point"""
    regex_df, openai_df = load_datasets()
    
    if regex_df is None or openai_df is None:
        return
    
    merged = compare_results(regex_df, openai_df)
    create_comparison_report(merged)
    
    print("\n" + "="*80)
    print("✅ Comparison Complete!")
    print("="*80)
    
    print("\n💡 Insights:")
    print("  - High agreement rate = Both methods are reliable")
    print("  - Regex YES, OpenAI NO + Low confidence = Regex false positives")
    print("  - Regex NO, OpenAI YES + High confidence = Regex false negatives")
    print("  - Review disagreements to improve classification rules")

if __name__ == '__main__':
    main()

