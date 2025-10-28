import pandas as pd
import re
from typing import List, Set, Dict, Any
import json
from datetime import datetime
from pathlib import Path

class TypeScriptTypePRExtractorV4:
    """
    V4: Extractor focused on detecting and counting advanced TypeScript type features
    and type safety patterns (generics, conditional types, satisfies, etc.)
    in PR patch additions, comparing AI vs. Human problem-solving styles.
    """
    
    # AI Agents filter
    AI_AGENTS = ['OpenAI_Codex', 'Devin', 'Copilot', 'Cursor', 'Claude_Code']
    
    # File extensions
    TS_EXTENSIONS = {'.ts', '.tsx'}
    
    # --- V4 NEW PATTERNS for Advanced Type Feature Counting (Refined for Rigor) ---
    ADVANCED_TYPE_PATTERNS = [
        # 1. Generics: Focused on declaration/definition to avoid JSX FPs.
        (r'(interface|type|class|function)\s+\w*\s*<[^<>]+>', 'generics_count'),
        
        # 2. Conditional Types: Matches T extends U ? X : Y structure.
        (r'\bextends\b[^?]+\?.+:.+', 'conditional_type_count'),
        
        # 3. satisfies operator (TS 4.9+):
        (r'\s+satisfies\s+', 'satisfies_count'),
        
        # 4. as const: Enforcing literal types
        (r'\bas\s+const\b', 'as_const_count'),
        
        # 5. Non-null Assertion Operator: Matches the ! used for assertion (e.g., foo!.bar, foo![0], foo!())
        (r'!\s*([\.\[(])', 'non_null_assertion_count'),
        
        # 6. Type Guard ('is'): Matching 'is' in return types (e.g., arg is T)
        (r':\s*\(?\s*\w+\s+is\s+\w+', 'type_guard_is_count'), 
        
        # 7. keyof typeof: Advanced utility type usage
        (r'\bkeyof\s+typeof\b', 'keyof_typeof_count'),
    ]
    
    # V3 Legacy Patterns (Scores and Filters)
    TYPE_KEYWORDS_SCORED = [
        (r'\btype\s+\w+\s*=', 12), (r'\binterface\s+\w+', 12),
        (r':\s*[A-Z][a-zA-Z]*\s*[;\)\}]', 10), (r'as\s+[A-Z][a-zA-Z]*', 8),
        (r'<[A-Z][a-zA-Z]*>', 7), (r'\btype\s+fix\b', 8), (r'\bfix.*type error\b', 9),
        (r'\bnoImplicitAny\b', 11), (r'\bstrictNullChecks\b', 11),
        (r'\badd.*\btype\b', 6), (r'\bimprove.*\btyping\b', 7), (r'\brefactor.*\btype\b', 6),
    ]
    PATCH_ADDITION_PATTERNS = [
        (r'.*:\s*[a-zA-Z_][\w]*\s*[;\)\}]', 15), (r'.*:\s*[A-Z][a-zA-Z]*\s*[;\)\}]', 18),
        (r'(interface|type)\s+\w+', 25), (r'.*\s+as\s+[A-Z][a-zA-Z]*', 12),
        (r'.*<[^<>]*[A-Z][a-zA-Z][^<>]*>', 10), (r'.*:\s*any\b', 18),
    ]
    FP_EXCLUDE_PATTERNS = [
        r'type\s*=\s*["\']', r'typeof\s+\w+', r'@type\s+{', r'console\.log.*type',
        r'button.*type', r'input.*type', r'\.d\.ts\b',
    ]
    ANY_REPLACEMENT_RAW_PATTERN = r'^\-\s*.*:\s*([a-zA-Z_][\w\[\]<>]*)\s*[;\)\}]?\s*$\n^\+\s*.*:\s*any\b'
    MIN_TITLE_SCORE, MIN_PATCH_SCORE, MIN_TOTAL_SCORE = 10, 18, 28

    def __init__(self):
        self.pr_df = None
        self.human_pr_df = None
        self.repo_df = None
        self.pr_commits_df = None
        self.pr_commit_details_df = None
        self.typescript_type_prs = None
        
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile all regular expressions."""
        self.compiled_type_keywords = [
            (re.compile(p, re.IGNORECASE), w) for p, w in self.TYPE_KEYWORDS_SCORED
        ]
        self.compiled_patch_add_patterns = [
            (re.compile(p), w) for p, w in self.PATCH_ADDITION_PATTERNS
        ]
        self.compiled_fp_patterns = [re.compile(p, re.IGNORECASE) for p in self.FP_EXCLUDE_PATTERNS]
        
        # Specific patterns for 'any' counting (findall)
        self.compiled_any_add = re.compile(r'^\+\s*.*:\s*any\b', re.MULTILINE)
        self.compiled_any_rem = re.compile(r'^\-\s*.*:\s*any\b', re.MULTILINE)
        self.compiled_any_replacement = re.compile(self.ANY_REPLACEMENT_RAW_PATTERN, re.MULTILINE)

        # V4 NEW: Compiled Advanced Type Patterns
        self.compiled_advanced_patterns = [
            (re.compile(p), col_name) for p, col_name in self.ADVANCED_TYPE_PATTERNS
        ]
        self.ADVANCED_COUNT_COLS = [col_name for _, col_name in self.ADVANCED_TYPE_PATTERNS]

    def load_datasets(self):
        print("Loading datasets from HuggingFace...")
        try:
             self.pr_df = pd.read_parquet('hf://datasets/hao-li/AIDev/pull_request.parquet')
             self.repo_df = pd.read_parquet('hf://datasets/hao-li/AIDev/repository.parquet')
             self.pr_commits_df = pd.read_parquet('hf://datasets/hao-li/AIDev/pr_commits.parquet')
             self.pr_commit_details_df = pd.read_parquet('hf://datasets/hao-li/AIDev/pr_commit_details.parquet')
             
             print(f"Loaded: {len(self.pr_df):,} PRs, {len(self.repo_df):,} repos")
        except Exception as e:
             print(f"Error loading data from HuggingFace/Parquet: {e}")
             raise

    def filter_prs_by_agent_status(self) -> pd.DataFrame:
        """Filters all PRs in TS repos and marks them as 'AI' or 'Human'."""
        if self.pr_df.empty or self.repo_df.empty:
            return pd.DataFrame()
            
        print("\nFiltering TypeScript PRs and classifying by agent status...")
        
        ts_repos = self.repo_df[
            self.repo_df['language'].str.contains('TypeScript', case=False, na=False)
        ]
        ts_repo_ids = set(ts_repos['id'].tolist())

        # Filter all PRs in TS repos
        all_ts_prs = self.pr_df[
            self.pr_df['repo_id'].isin(ts_repo_ids)
        ].copy()
        
        # Classify PRs
        all_ts_prs['group'] = all_ts_prs['agent'].apply(
            lambda x: 'AI' if x in self.AI_AGENTS else 'Human'
        )

        print(f"   Found {len(all_ts_prs):,} total PRs in TS repos ({len(all_ts_prs[all_ts_prs['group'] == 'AI']):,} AI, {len(all_ts_prs[all_ts_prs['group'] == 'Human']):,} Human)")
        return all_ts_prs

    def _has_fp(self, text: str) -> bool:
        if pd.isna(text): return False
        return any(p.search(text) for p in self.compiled_fp_patterns)

    def _score_text(self, text: str) -> int:
        if pd.isna(text): return 0
        score = 0
        for pattern, weight in self.compiled_type_keywords:
            if pattern.search(text):
                score += weight
        return score

    def _score_patch(self, patch: str) -> int:
        if pd.isna(patch): return 0
        score = 0
        for line in patch.splitlines():
            # Only score additions (+)
            if line.startswith('+') and not line.startswith('+++'):
                line_content = line[1:].strip() 
                for pattern, weight in self.compiled_patch_add_patterns:
                    if pattern.search(line_content):
                        score += weight
                        break
        return score

    def _is_valid_ts_file(self, filename: str) -> bool:
        if pd.isna(filename): return False
        path = Path(filename)
        # Exclude declaration files (.d.ts) as they often only contain type definitions
        return path.suffix.lower() in self.TS_EXTENSIONS and not path.name.endswith('.d.ts')

    def identify_type_related_prs(self, all_ts_prs: pd.DataFrame) -> pd.DataFrame:
        if all_ts_prs.empty:
            print("No PRs to analyze.")
            return pd.DataFrame()

        print("\nIdentifying type-related PRs with advanced feature detection...")

        # 1. False Positive removal
        print("   Applying FP filters...")
        all_ts_prs['has_fp'] = (
            all_ts_prs['title'].apply(self._has_fp) |
            all_ts_prs['body'].apply(self._has_fp)
        )
        filtered_prs = all_ts_prs[~all_ts_prs['has_fp']].copy()

        # 2. Text/Commit Scoring
        print("   Scoring PR titles and bodies...")
        filtered_prs['text_score'] = filtered_prs['title'].apply(self._score_text) + filtered_prs['body'].apply(self._score_text)
        
        commit_scores = self.pr_commits_df.groupby('pr_id').apply(
            lambda g: max(self._score_text(msg) for msg in g['message']),
            include_groups=False
        ).to_dict()
        filtered_prs['commit_score'] = filtered_prs['id'].map(commit_scores).fillna(0).astype(int)

        # 3. Patch analysis with type feature counting
        print("   Analyzing patches + counting all features...")
        ts_details = self.pr_commit_details_df[
            self.pr_commit_details_df['filename'].apply(self._is_valid_ts_file)
        ].copy()

        def analyze_patch_group(group):
            max_score = 0
            any_add = 0
            any_rem = 0
            any_replacements = 0
            
            advanced_counts = {col: 0 for col in self.ADVANCED_COUNT_COLS}

            # Concatenate all valid TS patches for this PR
            full_patch_text = "\n".join(group['patch'].dropna().tolist())
            
            if full_patch_text:
                # Score (needs to be calculated on a per-patch basis to find max_score)
                max_score = max(self._score_patch(patch) for patch in group['patch'].dropna())
                
                # any counts (on the combined text)
                any_add = len(self.compiled_any_add.findall(full_patch_text))
                any_rem = len(self.compiled_any_rem.findall(full_patch_text))
                any_replacements = len(self.compiled_any_replacement.findall(full_patch_text))

                # Count advanced features on ADDED lines only
                added_lines = "\n".join([line for line in full_patch_text.splitlines() if line.startswith('+') and not line.startswith('+++')])
                for pattern, col_name in self.compiled_advanced_patterns:
                    advanced_counts[col_name] += len(pattern.findall(added_lines))
            
            results = {
                'patch_score': max_score,
                'any_additions': any_add,
                'any_removals': any_rem,
                'any_replacements': any_replacements,
                **advanced_counts
            }
            return pd.Series(results)

        patch_stats = ts_details.groupby('pr_id').apply(analyze_patch_group, include_groups=False)
        
        # Merge all patch stats back into the PR dataframe
        filtered_prs = filtered_prs.merge(patch_stats, left_on='id', right_index=True, how='left')

        # Fill NaNs for all count columns
        count_cols = ['patch_score', 'any_additions', 'any_removals', 'any_replacements'] + self.ADVANCED_COUNT_COLS
        for col in count_cols:
            filtered_prs[col] = filtered_prs[col].fillna(0).astype(int)

        # 4. TS file count and Total score
        ts_file_count = ts_details.groupby('pr_id').size().to_dict()
        filtered_prs['ts_file_count'] = filtered_prs['id'].map(ts_file_count).fillna(0).astype(int)
        filtered_prs['has_ts_files'] = filtered_prs['ts_file_count'] > 0

        filtered_prs['total_score'] = (
            filtered_prs['text_score'] +
            filtered_prs['commit_score'] +
            filtered_prs['patch_score'] +
            (filtered_prs['ts_file_count'] * 2)
        )

        # 5. Final filtering
        type_prs = filtered_prs[
            filtered_prs['has_ts_files'] &
            (
                (filtered_prs['text_score'] >= self.MIN_TITLE_SCORE) |
                (filtered_prs['patch_score'] >= self.MIN_PATCH_SCORE)
            ) &
            (filtered_prs['total_score'] >= self.MIN_TOTAL_SCORE)
        ].copy()

        # Add detection method for filtering validation
        type_prs['detection_method'] = ''
        type_prs.loc[type_prs['text_score'] >= self.MIN_TITLE_SCORE, 'detection_method'] += 'text|'
        type_prs.loc[type_prs['patch_score'] >= self.MIN_PATCH_SCORE, 'detection_method'] += 'patch|'
        type_prs['detection_method'] = type_prs['detection_method'].str.rstrip('|')

        print(f"\n   Found {len(type_prs):,} high-confidence type-related PRs.")
        return type_prs

    def enrich_with_commit_stats(self, type_prs: pd.DataFrame) -> pd.DataFrame:
        if type_prs.empty:
            return pd.DataFrame()
        
        print("\nEnriching with commit stats and collecting full patches...")
        ts_details = self.pr_commit_details_df[
            self.pr_commit_details_df['pr_id'].isin(type_prs['id']) &
            self.pr_commit_details_df['filename'].apply(self._is_valid_ts_file)
        ]

        stats = ts_details.groupby('pr_id').agg(
            additions=('additions', 'sum'),
            deletions=('deletions', 'sum'),
            changes=('changes', 'sum'),
            ts_files_changed=('filename', 'nunique')
        )

        def collect_patches(group):
            patches = []
            for _, row in group.iterrows():
                if pd.notna(row['patch']) and row['patch'].strip():
                    header = f"=== {row['filename']} (+{row['additions']}/-{row['deletions']}) ==="
                    patches.append(f"{header}\n{row['patch']}")
            return "\n\n".join(patches) if patches else ""

        patch_text = ts_details.groupby('pr_id').apply(collect_patches, include_groups=False).rename('patch_text')

        enriched = type_prs.merge(stats, left_on='id', right_index=True, how='left') \
                          .merge(patch_text, left_on='id', right_index=True, how='left')

        for col in ['additions', 'deletions', 'changes', 'ts_files_changed']:
            enriched[col] = enriched[col].fillna(0).astype(int)
        enriched['patch_text'] = enriched['patch_text'].fillna('')

        return enriched

    def export_results(self, enriched_prs: pd.DataFrame, output_file: str):
        if enriched_prs.empty:
            print("No results to export.")
            return

        print(f"\nExporting to {output_file}...")
        export_cols = [
            'id', 'number', 'title', 'body', 'agent', 'group', 'state', 'created_at', 'merged_at',
            'repo_id', 'html_url', 'additions', 'deletions', 'changes', 'ts_files_changed',
            'any_additions', 'any_removals', 'any_replacements',
        ] + self.ADVANCED_COUNT_COLS + [
            'text_score', 'patch_score', 'total_score', 'detection_method', 'patch_text'
        ]
        
        export_cols = [col for col in export_cols if col in enriched_prs.columns]
            
        enriched_prs[export_cols].to_csv(output_file, index=False)
        print(f"   Exported {len(enriched_prs):,} PRs")

        # Calculate agent-specific feature density (per PR average)
        agent_feature_density = {}
        for agent in enriched_prs['agent'].unique():
            agent_data = enriched_prs[enriched_prs['agent'] == agent]
            agent_feature_density[agent] = {}
            for col in self.ADVANCED_COUNT_COLS:
                total = float(agent_data[col].sum())
                count = len(agent_data)
                agent_feature_density[agent][col] = round(total / count if count > 0 else 0, 4)
        
        # Calculate group-specific feature density
        group_feature_density = {}
        for group in enriched_prs['group'].unique():
            group_data = enriched_prs[enriched_prs['group'] == group]
            group_feature_density[group] = {}
            for col in self.ADVANCED_COUNT_COLS:
                total = float(group_data[col].sum())
                count = len(group_data)
                group_feature_density[group][col] = round(total / count if count > 0 else 0, 4)
        
        # Calculate agent-specific any statistics
        agent_any_stats = {}
        for agent in enriched_prs['agent'].unique():
            agent_data = enriched_prs[enriched_prs['agent'] == agent]
            agent_any_stats[agent] = {
                'any_additions': int(agent_data['any_additions'].sum()),
                'any_removals': int(agent_data['any_removals'].sum()),
                'any_replacements': int(agent_data['any_replacements'].sum()),
            }
        
        # Calculate group-specific any statistics
        group_any_stats = {}
        for group in enriched_prs['group'].unique():
            group_data = enriched_prs[enriched_prs['group'] == group]
            group_any_stats[group] = {
                'any_additions': int(group_data['any_additions'].sum()),
                'any_removals': int(group_data['any_removals'].sum()),
                'any_replacements': int(group_data['any_replacements'].sum()),
            }
        
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'total_type_prs': len(enriched_prs),
            'by_group': enriched_prs['group'].value_counts().to_dict(),
            'by_agent': enriched_prs['agent'].value_counts().to_dict(),
            'any_additions_total': int(enriched_prs['any_additions'].sum()),
            'any_removals_total': int(enriched_prs['any_removals'].sum()),
            'any_replacements_total': int(enriched_prs['any_replacements'].sum()),
            'any_replacements_pr': int((enriched_prs['any_replacements'] > 0).sum()),
            'agent_any_stats': agent_any_stats,
            'group_any_stats': group_any_stats,
            'advanced_feature_totals': {col: int(enriched_prs[col].sum()) for col in self.ADVANCED_COUNT_COLS},
            'agent_feature_density': agent_feature_density,
            'group_feature_density': group_feature_density,
        }
        with open(output_file.replace('.csv', '_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   Summary saved to {output_file.replace('.csv', '_summary.json')}")

    def run_pipeline(self, output_file: str = 'ts_type_prs_all_groups_final.csv'):
        print("="*80)
        print("TypeScript Type-Related PR Extraction (V4 - Advanced Feature Analysis)")
        print("="*80)

        self.load_datasets()
        all_ts_prs = self.filter_prs_by_agent_status() 
        type_prs = self.identify_type_related_prs(all_ts_prs)
        enriched = self.enrich_with_commit_stats(type_prs)
        self.export_results(enriched, output_file)
        self.typescript_type_prs = enriched

        print("\n" + "="*80)
        print("Pipeline completed!")
        print("="*80)
        return enriched


def main():
    extractor = TypeScriptTypePRExtractorV4()
    results = extractor.run_pipeline()
    return results


if __name__ == '__main__':
    main()
