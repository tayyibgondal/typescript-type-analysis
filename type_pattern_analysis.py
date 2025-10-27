"""
TypeScript Type Pattern Analysis
- Infer type pattern analysis from AIDev dataset
- Quantitative analysis of type-related patterns (any, infer, satisfies, etc.)
- Compare with human PRs (GitHub API)
- Include merge rate
"""

import pandas as pd
import re
import requests
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import time
import os
from dotenv import load_dotenv
load_dotenv()

# GitHub token (required)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Set in environment variable or directly

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable must be set! (export GITHUB_TOKEN=ghp_...)")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}


class TypePatternAnalyzer:
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        print(f"Created temp directory: {self.temp_dir}")

    def __del__(self):
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up: {self.temp_dir}")
            except:
                pass

    def load_dataset(self):
        print("Loading AIDev dataset via hf:// URLs...")
        self.pr_df = pd.read_parquet('hf://datasets/hao-li/AIDev/pull_request.parquet')
        self.pr_commit_details_df = pd.read_parquet('hf://datasets/hao-li/AIDev/pr_commit_details.parquet')
        self.repo_df = pd.read_parquet('hf://datasets/hao-li/AIDev/repository.parquet')
        
        print(f"Loaded: {len(self.pr_df):,} PRs, {len(self.repo_df):,} repos, {len(self.pr_commit_details_df):,} commit details")
        return self.pr_df

    def is_typescript_pr(self, row) -> bool:
        patch_text = row.get('patch_text', '')
        if pd.isna(patch_text):
            return False
        # Look for TypeScript files in the patch (check for file names with .ts or .tsx extension)
        # Check lines starting with +++ (old file) or +++ (new file) or ===
        lines = patch_text.split('\n')
        ts_patterns = [
            r'^--- a.*\.(ts|tsx)\s*$',
            r'^\+\+\+ b.*\.(ts|tsx)\s*$',
            r'^===.*\.(ts|tsx)\s*$',
            r'diff --git.*\.(ts|tsx)',
        ]
        for line in lines:
            for pattern in ts_patterns:
                if re.search(pattern, line):
                    return True
        return False

    def is_type_relevant(self, patch_text: str) -> bool:
        if pd.isna(patch_text):
            return False
        keywords = ['type ', 'interface ', ' as ', ' infer ', ' extends ', ' satisfies ', ': ']
        return any(k in patch_text for k in keywords)

    def extract_added_code(self, patch_text: str) -> str:
        if pd.isna(patch_text):
            return ""
        lines = []
        for line in patch_text.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                code = line[1:].rstrip()
                if code and not code.startswith('//'):
                    lines.append(code)
        return '\n'.join(lines)

    def analyze_type_patterns(self, patch_text: str) -> Dict[str, Any]:
        code = self.extract_added_code(patch_text)
        if not code.strip():
            return {}

        lines = [l for l in code.split('\n') if l.strip()]
        total_lines = len(lines)

        patterns = {
            'infer': len(re.findall(r'\binfer\b', code)),
            'extends': len(re.findall(r'\bextends\b', code)),
            'keyof': len(re.findall(r'\bkeyof\b', code)),
            'satisfies': len(re.findall(r'\bsatisfies\b', code)),
            'as_const': len(re.findall(r'\bas\s+const\b', code)),
            'non_null_assertion': len(re.findall(r'!', code)),
            'any': len(re.findall(r'\bany\b', code)),
            'unknown': len(re.findall(r'\bunknown\b', code)),
            'type_annotations': len(re.findall(r':\s*[A-Za-z_<>\[\]{}|]', code)),
            'total_lines': total_lines
        }

        # Calculate rates
        if total_lines > 0:
            pattern_keys = list(patterns.keys())  # Create a copy of keys to avoid modification during iteration
            for k in pattern_keys:
                if k != 'total_lines':
                    patterns[f'{k}_rate'] = patterns[k] / total_lines

        return patterns

    def filter_typescript_prs(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Filtering TypeScript + Type-relevant PRs...")
        
        # 1. Filter TypeScript repositories
        ts_repos = self.repo_df[
            self.repo_df['language'].str.contains('TypeScript', case=False, na=False)
        ]
        ts_repo_ids = set(ts_repos['id'].tolist())
        print(f"  Found {len(ts_repo_ids):,} TypeScript repos")
        
        # 2. Filter PRs in TS repos
        ts_prs = df[df['repo_id'].isin(ts_repo_ids)].copy()
        print(f"  Found {len(ts_prs):,} PRs in TS repos")
        
        # 3. Get TS file details and aggregate patches
        def is_ts_file(filename):
            if pd.isna(filename): return False
            return filename.endswith(('.ts', '.tsx')) and not filename.endswith('.d.ts')
        
        ts_details = self.pr_commit_details_df[
            self.pr_commit_details_df['filename'].apply(is_ts_file)
        ]
        print(f"  Found {len(ts_details):,} TS file changes")
        
        # Aggregate patches by PR
        def collect_patches(group):
            patches = []
            for _, row in group.iterrows():
                if pd.notna(row['patch']) and row['patch'].strip():
                    header = f"=== {row['filename']} (+{row['additions']}/-{row['deletions']}) ==="
                    patches.append(f"{header}\n{row['patch']}")
            return "\n\n".join(patches) if patches else ""
        
        patch_text = ts_details.groupby('pr_id').apply(collect_patches, include_groups=False).rename('patch_text')
        
        # Merge with PRs
        ts_prs = ts_prs.merge(patch_text, left_on='id', right_index=True, how='inner')
        ts_prs['patch_text'] = ts_prs['patch_text'].fillna('')
        
        # 4. Filter type-relevant PRs
        ts_prs['is_type_relevant'] = ts_prs['patch_text'].apply(self.is_type_relevant)
        ts_prs_filtered = ts_prs[ts_prs['is_type_relevant']].copy()
        
        print(f"  After type-relevant filter: {len(ts_prs_filtered):,} PRs")
        return ts_prs_filtered

    def fetch_human_prs(self, repo_full_name: str, sample_size: int = 20) -> List[Dict]:
        owner, name = repo_full_name.split('/')
        url = f"https://api.github.com/repos/{owner}/{name}/pulls"
        params = {'state': 'closed', 'per_page': 100, 'page': 1}
        prs = []

        while len(prs) < sample_size:
            response = requests.get(url, headers=HEADERS, params=params)
            if response.status_code != 200:
                print(f"API Error {response.status_code}: {response.text}")
                break
            data = response.json()
            if not data:
                break

            for pr in data:
                if len(prs) >= sample_size:
                    break
                if pr.get('merged_at') and any('bug' in label['name'].lower() for label in pr.get('labels', [])):
                    patch_url = pr.get('patch_url') or f"https://api.github.com/repos/{owner}/{name}/pulls/{pr['number']}"
                    patch_resp = requests.get(patch_url, headers=HEADERS)
                    if patch_resp.status_code == 200:
                        patch_text = patch_resp.text
                        if self.is_typescript_pr({'patch_text': patch_text}) and self.is_type_relevant(patch_text):
                            prs.append({
                                'pr_id': pr['number'],
                                'title': pr['title'],
                                'patch_text': patch_text,
                                'merged': True,
                                'agent': 'human'
                            })
            params['page'] += 1
            time.sleep(1)  # Avoid rate limit

        print(f"  Fetched {len(prs)} human PRs from {repo_full_name}")
        return prs

    def analyze_prs(self, df: pd.DataFrame, label: str = "ai") -> pd.DataFrame:
        print(f"\nAnalyzing {len(df):,} {label.upper()} PRs...")
        results = []

        for idx, row in df.iterrows():
            pr_id = row.get('id') or row.get('number', 'unknown')
            title = str(row.get('title', ''))[:100]
            agent = row.get('agent', label)
            patch_text = row.get('patch_text', '')

            patterns = self.analyze_type_patterns(patch_text)
            if not patterns:
                continue

            patterns.update({
                'pr_id': pr_id,
                'title': title,
                'agent': agent,
                'merged': pd.notna(row.get('merged_at')) if label == 'ai' else row.get('merged', True)
            })
            results.append(patterns)

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(df)}...")

        result_df = pd.DataFrame(results)
        print(f"  Done: {len(result_df)} PRs with patterns")
        return result_df

    def print_comparison(self, ai_df: pd.DataFrame, human_df: pd.DataFrame):
        print("\n" + "="*100)
        print("TYPE PATTERN COMPARISON: AI vs HUMAN")
        print("="*100)

        # Merge rate
        ai_merge = ai_df['merged'].mean() * 100
        human_merge = human_df['merged'].mean() * 100

        # Average patterns
        metrics = ['any_rate', 'infer_rate', 'satisfies_rate', 'type_annotations_rate', 'non_null_assertion_rate']
        ai_means = ai_df[metrics].mean()
        human_means = human_df[metrics].mean()

        comparison = pd.DataFrame({
            'AI': ai_means,
            'Human': human_means,
            'Diff (%)': ((ai_means - human_means) / human_means * 100).round(1)
        }).round(4)

        print(f"\nMerge Acceptance Rate:")
        print(f"  AI: {ai_merge:.1f}% | Human: {human_merge:.1f}% | Diff: {ai_merge - human_merge:+.1f}p")

        print(f"\nType Pattern Usage (per line):")
        print(comparison)

        # Save
        comparison.to_csv('type_pattern_comparison.csv')
        ai_df.to_csv('ai_type_patterns.csv', index=False)
        human_df.to_csv('human_type_patterns.csv', index=False)
        print("\nSaved: type_pattern_comparison.csv, ai_type_patterns.csv, human_type_patterns.csv")


def main():
    analyzer = TypePatternAnalyzer()
    df = analyzer.load_dataset()
    
    # Debug: Check available columns and sample data
    print("\n" + "="*100)
    print("DATASET INFO")
    print("="*100)
    print(f"PR columns: {list(df.columns)}")
    print(f"Repo columns: {list(analyzer.repo_df.columns)}")
    print("="*100 + "\n")

    # 1. Analyze AI PRs
    ts_df = analyzer.filter_typescript_prs(df)
    ai_full_dataset = ts_df.copy()
    ai_results = analyzer.analyze_prs(ai_full_dataset, label="ai")
    
    # Save AI results
    print("\nSaving results...")
    ai_results.to_csv('ai_type_patterns.csv', index=False)
    print("Saved: ai_type_patterns.csv")
    
    # Print basic statistics
    print("\n" + "="*100)
    print("AI PR TYPE PATTERN SUMMARY")
    print("="*100)
    if 'merged' in ai_results.columns:
        merged_rate = ai_results['merged'].mean() * 100
        print(f"Merge rate: {merged_rate:.1f}%")
    print(f"\nAverage patterns per line:")
    pattern_metrics = ['any_rate', 'infer_rate', 'satisfies_rate', 'type_annotations_rate']
    for metric in pattern_metrics:
        if metric in ai_results.columns:
            print(f"  {metric}: {ai_results[metric].mean():.4f}")
    print("="*100)

    # NOTE: We should collect human PRs and compare with AI PRs!!!!

    # # 2. Collect Human PRs (from repos with AI PRs)
    # # Try different possible column names for repo
    # repo_col = None
    # for col in ['repo_name', 'repository', 'repo', 'full_name']:
    #     if col in df.columns:
    #         repo_col = col
    #         break
    
    # if repo_col is None:
    #     print("WARNING: No repository column found. Skipping human PR analysis.")
    #     return
    
    # repo_names = ts_df[repo_col].dropna().unique()[:3]  # Top 3 repos
    # human_prs = []
    # for repo in repo_names:
    #     try:
    #         human_prs.extend(analyzer.fetch_human_prs(repo, sample_size=20))
    #     except Exception as e:
    #         print(f"Error fetching human PRs from {repo}: {e}")
    #         continue
    # human_df = pd.DataFrame(human_prs)
    
    # if len(human_df) > 0:
    #     human_results = analyzer.analyze_prs(human_df, label="human")
    #     # 3. Compare
    #     analyzer.print_comparison(ai_results, human_results)
    # else:
    #     print("No human PRs collected. Skipping comparison.")

    print("\n" + "="*100)
    print("DONE! Check CSV files for details.")
    print("="*100)


if __name__ == '__main__':
    main()