import pandas as pd
import re

# Load the CSV file with any_replacements column
df = pd.read_csv('ts_type_prs_with_any_replacement.csv')

print("="*80)
print("Malicious Any Replacement Analysis")
print("="*80)

# Filter PRs with any_replacements > 0
bad_prs = df[df['any_replacements'] > 0]
print(f"\n Malicious PRs: {len(bad_prs)}")

# Agent breakdown
print("\n Agent-wise any_replacements total:")
print(bad_prs.groupby('agent')['any_replacements'].sum())

# Detailed breakdown
print("\n Agent-wise PR count:")
print(bad_prs['agent'].value_counts())

print("\n" + "="*80)
print("Actual changes (top 3 PRs)")
print("="*80)

# Show top 3 PRs with actual changes
for idx, (_, row) in enumerate(bad_prs.head(3).iterrows(), 1):
    print(f"\n[{idx}] PR #{row['number']} ({row['id']})")
    print(f"Agent: {row['agent']}")
    print(f"Title: {row['title']}")
    print(f"any_replacements: {row['any_replacements']}")
    
    # Extract actual replacement patterns from patch_text
    patch_text = row['patch_text']
    
    # Find lines where we remove a concrete type and add any
    lines = patch_text.split('\n')
    for i in range(len(lines) - 1):
        if lines[i].startswith('-') and ':' in lines[i]:
            # Check if next line adds 'any'
            if i + 1 < len(lines) and lines[i + 1].startswith('+'):
                if ':' in lines[i + 1] and ': any' in lines[i + 1]:
                    # Print the context
                    start_idx = max(0, i - 2)
                    end_idx = min(len(lines), i + 3)
                    
                    # Check if it's part of a code block (not just diff headers)
                    context_lines = lines[start_idx:end_idx]
                    has_removal = any('-' in line and ':' in line for line in context_lines)
                    has_addition = any('+' in line and ': any' in line for line in context_lines)
                    
                    if has_removal and has_addition:
                        print("\n  Changes:")
                        for line in context_lines:
                            if line.strip():  # Skip empty lines
                                print(f"  {line[:100]}")  # First 100 chars
                        break
    
    print("-" * 80)

print("\n" + "="*80)
print(f"Total {len(bad_prs)} PRs with {bad_prs['any_replacements'].sum()} malicious any replacements")
print("="*80)

