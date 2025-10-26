"""
Analysis script for TypeScript type-related PRs
Shows insights and examples from the extracted data
"""

import pandas as pd
import json

print("="*80)
print("TypeScript Type-Related PR Analysis")
print("="*80)

# Load the extracted data
print("\n📂 Loading extracted data...")
df = pd.read_csv('typescript_type_related_agentic_prs.csv')
with open('typescript_type_related_agentic_prs_summary.json', 'r') as f:
    summary = json.load(f)

print(f"✅ Loaded {len(df):,} type-related PRs")

# Summary statistics
print("\n" + "="*80)
print("📊 Summary Statistics")
print("="*80)
print(f"Extraction Date: {summary['extraction_date']}")
print(f"Total PRs: {summary['total_type_prs']:,}")
print(f"Merged PRs: {summary['merged_prs']:,} ({summary['merged_prs']/summary['total_type_prs']*100:.1f}%)")
print(f"\nAverage per PR:")
print(f"  - Lines added: {summary['avg_additions']:.1f}")
print(f"  - Lines deleted: {summary['avg_deletions']:.1f}")
print(f"  - TypeScript files changed: {summary['avg_files_changed']:.1f}")

# Agent distribution
print("\n" + "="*80)
print("🤖 Distribution by AI Agent")
print("="*80)
for agent, count in sorted(summary['by_agent'].items(), key=lambda x: x[1], reverse=True):
    pct = count / summary['total_type_prs'] * 100
    print(f"  {agent:15s}: {count:,} ({pct:.1f}%)")

# Type detection methods
print("\n" + "="*80)
print("🔍 Type Detection Methods")
print("="*80)
for method, count in summary['type_detection_methods'].items():
    pct = count / summary['total_type_prs'] * 100
    print(f"  {method:10s}: {count:,} ({pct:.1f}%)")

# Top 10 PRs by lines added
print("\n" + "="*80)
print("📈 Top 10 PRs by Lines Added")
print("="*80)
top_additions = df.nlargest(10, 'additions')[['title', 'agent', 'additions', 'deletions', 'ts_files_changed', 'html_url']]
for idx, row in top_additions.iterrows():
    print(f"\n  {row['agent']} - +{int(row['additions'])}/-{int(row['deletions'])} lines, {int(row['ts_files_changed'])} files")
    print(f"  Title: {row['title'][:80]}")
    print(f"  URL: {row['html_url']}")

# Type-focused PRs (type in title)
print("\n" + "="*80)
print("🎯 PRs with 'Type' in Title (Sample)")
print("="*80)
type_in_title = df[df['is_type_in_title'] == True].head(10)
for idx, row in type_in_title.iterrows():
    print(f"\n  {row['agent']} - {row['state']}")
    print(f"  Title: {row['title'][:100]}")
    print(f"  Stats: +{int(row['additions'])}/-{int(row['deletions'])} lines, {int(row['ts_files_changed'])} TS files")
    print(f"  URL: {row['html_url']}")

# Analysis by state
print("\n" + "="*80)
print("📦 Analysis by PR State")
print("="*80)
state_stats = df.groupby('state').agg({
    'additions': 'mean',
    'deletions': 'mean',
    'ts_files_changed': 'mean',
    'id': 'count'
}).rename(columns={'id': 'count'})
print(state_stats)

# Type detection overlap
print("\n" + "="*80)
print("🔀 Type Detection Method Overlap")
print("="*80)
print(f"PRs with type in multiple locations:")
print(f"  - Title + Body: {((df['is_type_in_title']) & (df['is_type_in_body'])).sum():,}")
print(f"  - Title + Commits: {((df['is_type_in_title']) & (df['is_type_in_commits'])).sum():,}")
print(f"  - Body + Patches: {((df['is_type_in_body']) & (df['is_type_in_patches'])).sum():,}")
print(f"  - All 4 methods: {((df['is_type_in_title']) & (df['is_type_in_body']) & (df['is_type_in_commits']) & (df['is_type_in_patches'])).sum():,}")

# Agent-specific statistics
print("\n" + "="*80)
print("📊 Agent-Specific Statistics")
print("="*80)
agent_stats = df.groupby('agent').agg({
    'additions': 'mean',
    'deletions': 'mean',
    'ts_files_changed': 'mean',
    'state': lambda x: (x == 'closed').sum() / len(x) * 100
}).rename(columns={'state': 'merge_rate_%'})
print(agent_stats.round(2))

print("\n" + "="*80)
print("✅ Analysis Complete!")
print("="*80)
print(f"\n💡 Key Insights:")
print(f"  - {summary['total_type_prs']:,} type-related PRs extracted from {len(df['repo_id'].unique())} repositories")
print(f"  - Devin is the most active agent with {summary['by_agent']['Devin']:,} PRs ({summary['by_agent']['Devin']/summary['total_type_prs']*100:.1f}%)")
print(f"  - Most type changes detected in PR body ({summary['type_detection_methods']['body']:,} PRs)")
print(f"  - Average change size: +{summary['avg_additions']:.0f}/-{summary['avg_deletions']:.0f} lines")
print(f"  - {summary['merged_prs']:,} PRs merged ({summary['merged_prs']/summary['total_type_prs']*100:.1f}% merge rate)")

