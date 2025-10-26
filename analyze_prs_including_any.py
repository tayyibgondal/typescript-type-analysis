import pandas as pd

# Load CSV file
CSV_FILE = 'ts_type_prs_including_any.csv'

print("Loading CSV file...")
df = pd.read_csv(CSV_FILE)

print(f"Total {len(df):,} type-related PRs loaded!\n")
print("="*60)

# =============================================================================
# Analysis 1: Average any additions per agent (any_additions.mean())
# =============================================================================
print("1. Average any additions per agent (any_additions.mean())")
print("-" * 50)
agent_any_mean = df.groupby('agent')['any_additions'].mean().round(2).sort_values(ascending=False)
print(agent_any_mean)

# Visualization (optional)
print("\n[Visualization] Average any additions per agent")
agent_any_mean.plot(kind='barh', color='skyblue', title='Average any_additions per Agent')
import matplotlib.pyplot as plt
plt.xlabel('Average any_additions')
plt.tight_layout()
plt.show()

print("\n" + "="*60)

# =============================================================================
# Analysis 2: Extract PRs with type improvements (any_additions == 0 AND any_removals == 0)
# =============================================================================
print("2. PRs with type improvements (any_additions == 0 AND any_removals == 0)")
print("-" * 50)
clean = df[(df['any_additions'] == 0) & (df['any_removals'] == 0)]

print(f"   → Total {len(clean):,} PRs (of {len(clean)/len(df):.1%})")
print(f"   → Average patch_score: {clean['patch_score'].mean():.1f}")
print(f"   → Mainly detected methods: {clean['detection_method'].value_counts().to_dict()}")

# Save to CSV (optional)
clean_output = 'clean_type_improvement_no_any.csv'
clean[['id', 'title', 'agent', 'patch_score', 'detection_method']].to_csv(clean_output, index=False)
print(f"   → Saved: {clean_output}")

print("\n" + "="*60)

# =============================================================================
# Analysis 3: PRs with 10 or more any additions (serious any introduction)
# =============================================================================
print("3. PR that adds any 10 or more times (any_additions >= 10)")
print("-" * 50)
heavy_any = df[df['any_additions'] >= 10]

print(f"   → Total {len(heavy_any):,} PRs")
if len(heavy_any) > 0:
    print("\n   Top 5 PR examples:")
    print(heavy_any[['id', 'title', 'agent', 'any_additions', 'any_removals']].head())
    
    # any addition-heavy agents
    print("\n   Agent distribution:")
    print(heavy_any['agent'].value_counts())
else:
    print("   → No PRs with 10 or more any additions")

# Save to CSV (optional)
heavy_output = 'heavy_any_additions.csv'
heavy_any[['id', 'title', 'agent', 'any_additions', 'any_removals', 'patch_text']].to_csv(heavy_output, index=False)
print(f"   → Saved: {heavy_output}")

print("\n" + "="*60)
print("All analyses completed!")