"""
Pipeline to extract TypeScript type-related PRs from AI coding agents
in the AIDev dataset (100+ stars subset)

This pipeline:
1. Loads the AIDev-pop dataset (repos with 100+ stars)
2. Filters for TypeScript repositories
3. Identifies type-related PRs based on multiple signals
4. Enriches with commit statistics (lines added/deleted)
5. Exports results to CSV
"""

import pandas as pd
import re
from typing import List, Set
import json
from datetime import datetime
import os

class TypeScriptTypePRExtractor:
    """Extract TypeScript type-related PRs from AIDev dataset"""
    
    # Keywords that indicate type-related changes
    TYPE_KEYWORDS = [
        # Type annotations and definitions
        r'\btype\s+\w+',
        r'interface\s+\w+',
        r':\s*\w+\s*[;,\)]',
        r'as\s+\w+',
        
        # Type fixes and changes
        r'type\s+fix',
        r'fix.*type',
        r'type.*error',
        r'type.*issue',
        r'typing',
        r'typescript',
        
        # Type-related refactoring
        r'add.*type',
        r'update.*type',
        r'change.*type',
        r'improve.*type',
        r'refactor.*type',
        r'type.*annotation',
        r'type.*definition',
        r'type.*safety',
        r'type.*checking',
        
        # Generic and utility types
        r'generic.*type',
        r'utility.*type',
        r'type.*parameter',
        
        # Type inference and casting
        r'type.*infer',
        r'type.*cast',
        r'type.*assertion',
        
        # Common TS type issues
        r'any.*type',
        r'unknown.*type',
        r'never.*type',
        r'void.*type',
        r'null.*type',
        r'undefined.*type',
        
        # Type strictness
        r'strict.*type',
        r'type.*strict',
        r'noImplicitAny',
        r'strictNullChecks',
    ]
    
    # Patch patterns that indicate type changes
    PATCH_TYPE_PATTERNS = [
        r'^\+.*:\s*\w+',  # Added type annotation
        r'^\-.*:\s*any',  # Removed 'any' type
        r'^\+.*interface\s+\w+',  # Added interface
        r'^\+.*type\s+\w+',  # Added type alias
        r'^\+.*<.*>',  # Added generics
        r'^\+.*as\s+\w+',  # Added type assertion
    ]
    
    def __init__(self):
        self.pr_df = None
        self.repo_df = None
        self.pr_commits_df = None
        self.pr_commit_details_df = None
        self.typescript_type_prs = None
        
    def load_datasets(self):
        """Load all required datasets from HuggingFace"""
        print("📂 Loading datasets from HuggingFace...")
        
        # Core datasets
        print("  - Loading pull_request.parquet...")
        self.pr_df = pd.read_parquet('hf://datasets/hao-li/AIDev/pull_request.parquet')
        
        print("  - Loading repository.parquet...")
        self.repo_df = pd.read_parquet('hf://datasets/hao-li/AIDev/repository.parquet')
        
        print("  - Loading pr_commits.parquet...")
        self.pr_commits_df = pd.read_parquet('hf://datasets/hao-li/AIDev/pr_commits.parquet')
        
        print("  - Loading pr_commit_details.parquet...")
        self.pr_commit_details_df = pd.read_parquet('hf://datasets/hao-li/AIDev/pr_commit_details.parquet')
        
        print(f"\n✅ Loaded datasets:")
        print(f"   PRs: {len(self.pr_df):,}")
        print(f"   Repositories: {len(self.repo_df):,}")
        print(f"   PR Commits: {len(self.pr_commits_df):,}")
        print(f"   Commit Details: {len(self.pr_commit_details_df):,}")
        
    def filter_typescript_repos(self) -> Set[int]:
        """Get repository IDs that are TypeScript-based"""
        print("\n🔍 Filtering TypeScript repositories...")
        
        # Filter repos where language contains TypeScript
        ts_repos = self.repo_df[
            self.repo_df['language'].str.contains('TypeScript', case=False, na=False)
        ]
        
        repo_ids = set(ts_repos['id'].tolist())
        print(f"   Found {len(repo_ids):,} TypeScript repositories")
        
        return repo_ids
    
    def filter_ai_agent_prs(self, ts_repo_ids: Set[int]) -> pd.DataFrame:
        """Filter PRs from AI agents in TypeScript repos"""
        print("\n🤖 Filtering AI agent PRs in TypeScript repositories...")
        
        # Filter for AI agents only (exclude human PRs)
        ai_agents = ['OpenAI_Codex', 'Devin', 'Copilot', 'Cursor', 'Claude_Code']
        
        agent_prs = self.pr_df[
            (self.pr_df['agent'].isin(ai_agents)) &
            (self.pr_df['repo_id'].isin(ts_repo_ids))
        ].copy()
        
        print(f"   Found {len(agent_prs):,} AI agent PRs in TypeScript repos")
        print(f"\n   Distribution by agent:")
        for agent, count in agent_prs['agent'].value_counts().items():
            print(f"     - {agent}: {count:,}")
            
        return agent_prs
    
    def is_type_related_text(self, text: str) -> bool:
        """Check if text contains type-related keywords"""
        if pd.isna(text):
            return False
            
        text_lower = text.lower()
        
        for pattern in self.TYPE_KEYWORDS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False
    
    def is_type_related_patch(self, patch: str) -> bool:
        """Check if patch contains type-related changes"""
        if pd.isna(patch):
            return False
            
        for pattern in self.PATCH_TYPE_PATTERNS:
            if re.search(pattern, patch, re.MULTILINE):
                return True
        return False
    
    def identify_type_related_prs(self, agent_prs: pd.DataFrame) -> pd.DataFrame:
        """Identify PRs that are related to type changes"""
        print("\n🎯 Identifying type-related PRs...")
        
        # Initialize type-related flags
        agent_prs['is_type_in_title'] = agent_prs['title'].apply(self.is_type_related_text)
        agent_prs['is_type_in_body'] = agent_prs['body'].apply(self.is_type_related_text)
        
        # Check commit messages
        print("   Analyzing commit messages...")
        pr_commit_types = self.pr_commits_df.groupby('pr_id', group_keys=False).apply(
            lambda x: any(self.is_type_related_text(msg) for msg in x['message']),
            include_groups=False
        ).to_dict()
        agent_prs['is_type_in_commits'] = agent_prs['id'].map(pr_commit_types).fillna(False).astype(bool)
        
        # Check patches (TypeScript files only)
        print("   Analyzing code patches...")
        ts_commit_details = self.pr_commit_details_df[
            self.pr_commit_details_df['filename'].str.endswith(('.ts', '.tsx'), na=False)
        ]
        
        pr_patch_types = ts_commit_details.groupby('pr_id', group_keys=False).apply(
            lambda x: any(self.is_type_related_patch(patch) for patch in x['patch']),
            include_groups=False
        ).to_dict()
        agent_prs['is_type_in_patches'] = agent_prs['id'].map(pr_patch_types).fillna(False).astype(bool)
        
        # Check if TypeScript files were modified
        pr_has_ts_files = ts_commit_details.groupby('pr_id').size().to_dict()
        agent_prs['has_typescript_files'] = agent_prs['id'].map(pr_has_ts_files).fillna(0) > 0
        
        # Combined type-related flag
        agent_prs['is_type_related'] = (
            agent_prs['is_type_in_title'] |
            agent_prs['is_type_in_body'] |
            agent_prs['is_type_in_commits'] |
            agent_prs['is_type_in_patches']
        )
        
        type_prs = agent_prs[agent_prs['is_type_related']].copy()
        
        print(f"\n   ✅ Found {len(type_prs):,} type-related PRs")
        print(f"   Breakdown:")
        print(f"     - Type in title: {agent_prs['is_type_in_title'].sum():,}")
        print(f"     - Type in body: {agent_prs['is_type_in_body'].sum():,}")
        print(f"     - Type in commits: {agent_prs['is_type_in_commits'].sum():,}")
        print(f"     - Type in patches: {agent_prs['is_type_in_patches'].sum():,}")
        
        return type_prs
    
    def enrich_with_commit_stats(self, type_prs: pd.DataFrame) -> pd.DataFrame:
        """Add commit statistics (lines added/deleted) and patch text to PRs"""
        print("\n📊 Enriching with commit statistics and patch text...")
        
        # Get TypeScript file changes only
        ts_commits = self.pr_commit_details_df[
            self.pr_commit_details_df['filename'].str.endswith(('.ts', '.tsx'), na=False)
        ]
        
        # Aggregate statistics per PR
        commit_stats = ts_commits.groupby('pr_id').agg({
            'additions': 'sum',
            'deletions': 'sum',
            'changes': 'sum',
            'filename': 'count'
        }).rename(columns={'filename': 'ts_files_changed'})
        
        # Aggregate patches per PR with file information
        print("   Collecting patch text...")
        def aggregate_patches(group):
            """Combine all patches with file information"""
            patches = []
            for _, row in group.iterrows():
                if pd.notna(row['patch']) and row['patch'].strip():
                    file_header = f"=== {row['filename']} (+{int(row['additions'])}/-{int(row['deletions'])}) ==="
                    patches.append(f"{file_header}\n{row['patch']}")
            return "\n\n".join(patches) if patches else ""
        
        patch_text = ts_commits.groupby('pr_id').apply(
            aggregate_patches,
            include_groups=False
        ).rename('patch_text')
        
        # Merge statistics and patches with type PRs
        enriched_prs = type_prs.merge(
            commit_stats,
            left_on='id',
            right_index=True,
            how='left'
        ).merge(
            patch_text,
            left_on='id',
            right_index=True,
            how='left'
        )
        
        # Fill NaN values
        enriched_prs['additions'] = enriched_prs['additions'].fillna(0).astype(int)
        enriched_prs['deletions'] = enriched_prs['deletions'].fillna(0).astype(int)
        enriched_prs['changes'] = enriched_prs['changes'].fillna(0).astype(int)
        enriched_prs['ts_files_changed'] = enriched_prs['ts_files_changed'].fillna(0).astype(int)
        enriched_prs['patch_text'] = enriched_prs['patch_text'].fillna('')
        
        print(f"   ✅ Added commit statistics and patch text")
        print(f"   Average per PR:")
        print(f"     - Lines added: {enriched_prs['additions'].mean():.1f}")
        print(f"     - Lines deleted: {enriched_prs['deletions'].mean():.1f}")
        print(f"     - Files changed: {enriched_prs['ts_files_changed'].mean():.1f}")
        print(f"     - PRs with patches: {(enriched_prs['patch_text'] != '').sum():,}")
        
        return enriched_prs
    
    def export_results(self, enriched_prs: pd.DataFrame, output_file: str):
        """Export results to CSV"""
        print(f"\n💾 Exporting results to {output_file}...")
        
        # Select columns for export
        export_columns = [
            'id', 'number', 'title', 'body', 'agent', 'state',
            'created_at', 'closed_at', 'merged_at',
            'repo_id', 'repo_url', 'html_url',
            'additions', 'deletions', 'changes', 'ts_files_changed',
            'is_type_in_title', 'is_type_in_body', 
            'is_type_in_commits', 'is_type_in_patches',
            'has_typescript_files',
            'patch_text'  # Include the actual patch/diff text
        ]
        
        enriched_prs[export_columns].to_csv(output_file, index=False)
        print(f"   ✅ Exported {len(enriched_prs):,} PRs to {output_file}")
        
        # Calculate file size
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"   📦 File size: {file_size_mb:.1f} MB")
        
        # Also export summary statistics
        summary_file = output_file.replace('.csv', '_summary.json')
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'total_type_prs': int(len(enriched_prs)),
            'by_agent': {k: int(v) for k, v in enriched_prs['agent'].value_counts().to_dict().items()},
            'by_state': {k: int(v) for k, v in enriched_prs['state'].value_counts().to_dict().items()},
            'merged_prs': int((enriched_prs['state'] == 'closed').sum()),
            'avg_additions': float(enriched_prs['additions'].mean()),
            'avg_deletions': float(enriched_prs['deletions'].mean()),
            'avg_files_changed': float(enriched_prs['ts_files_changed'].mean()),
            'type_detection_methods': {
                'title': int(enriched_prs['is_type_in_title'].sum()),
                'body': int(enriched_prs['is_type_in_body'].sum()),
                'commits': int(enriched_prs['is_type_in_commits'].sum()),
                'patches': int(enriched_prs['is_type_in_patches'].sum()),
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   ✅ Exported summary to {summary_file}")
    
    def run_pipeline(self, output_file: str = 'typescript_type_related_agentic_prs.csv'):
        """Run the complete extraction pipeline"""
        print("="*80)
        print("TypeScript Type-Related PR Extraction Pipeline")
        print("="*80)
        
        # Step 1: Load datasets
        self.load_datasets()
        
        # Step 2: Filter TypeScript repositories
        ts_repo_ids = self.filter_typescript_repos()
        
        # Step 3: Filter AI agent PRs in TypeScript repos
        agent_prs = self.filter_ai_agent_prs(ts_repo_ids)
        
        # Step 4: Identify type-related PRs
        type_prs = self.identify_type_related_prs(agent_prs)
        
        # Step 5: Enrich with commit statistics
        enriched_prs = self.enrich_with_commit_stats(type_prs)
        
        # Step 6: Export results
        self.export_results(enriched_prs, output_file)
        
        # Store for later access
        self.typescript_type_prs = enriched_prs
        
        print("\n" + "="*80)
        print("✅ Pipeline completed successfully!")
        print("="*80)
        
        return enriched_prs


def main():
    """Main entry point"""
    extractor = TypeScriptTypePRExtractor()
    results = extractor.run_pipeline()
    
    print("\n📈 Final Statistics:")
    print(f"   Total type-related PRs: {len(results):,}")
    print(f"   Merged PRs: {(results['state'] == 'closed').sum():,}")
    print(f"   Open PRs: {(results['state'] == 'open').sum():,}")
    print(f"\n   By Agent:")
    for agent, count in results['agent'].value_counts().items():
        print(f"     - {agent}: {count:,}")
    
    return results


if __name__ == '__main__':
    main()

