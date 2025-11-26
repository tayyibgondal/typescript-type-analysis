"""
C# Type Feature PR Extractor

Extracts and analyzes C# type-related PRs, counting advanced type features
and organizing results by agent.
"""

import pandas as pd
import re
from typing import List, Set, Dict, Any
import json
from datetime import datetime
from pathlib import Path


class CSharpTypePRExtractor:
    """
    Extractor for detecting and counting C# type features in PR patch additions,
    comparing AI vs. Human problem-solving styles.
    """
    
    # AI Agents filter
    AI_AGENTS = ['OpenAI_Codex', 'Devin', 'Copilot', 'Cursor', 'Claude_Code']
    
    # File extensions
    CS_EXTENSIONS = {'.cs'}
    
    # C# Type Feature Patterns
    TYPE_FEATURE_PATTERNS = [
        # 1. Delegate declarations
        (r'\bdelegate\s+(?:(?:return\s+)?[\w<>,\s\[\]?\.]+\s+)?[\w]+\s*\([^)]*\)', 'delegate_count'),
        
        # 2. Generics: List<T>, Dictionary<K,V>, Func<TResult>, etc.
        (r'\b[\w\.]+<(?:[^<>]+|<[^<>]*>)+>', 'generics_count'),
        
        # 3. var keyword
        (r'\bvar\s+\w+', 'var_count'),
        
        # 4. Nullable types: int?, string?, MyClass?, etc.
        (r'\b[\w\.\[\]]+\?(?!\?)', 'nullable_count'),  # Negative lookahead to avoid ??
        
        # 5. dynamic keyword
        (r'\bdynamic\s+\w+', 'dynamic_count'),
        
        # 6. object type usage
        (r'\bobject\s+(?:\w+|\(\s*\w+)|(?:\(\s*object\s*\))|(?:as\s+object\b)', 'object_count'),
        
        # 7. System.Reflection usage (excluding typeof which is counted separately)
        (r'(?:System\.Reflection\.[\w]+|\.GetType\s*\(|Assembly|MethodInfo|PropertyInfo|FieldInfo|Type\.)', 'reflection_count'),
        
        # 8. Explicit type casting: (Type) and "as Type"
        (r'(?:\(\s*[\w<>,\s\[\]\.?]+\s*\)\s*\w+|(?:as\s+[\w<>,\s\[\]\.?]+)\b)', 'explicit_cast_count'),
        
        # 9. Null-forgiving operator (!)
        (r'[\w\.\[\]()\?]+\!\s*[\.\[\]\(]', 'null_forgiving_count'),
        
        # 10. init keyword
        (r'\binit\s*;', 'init_count'),
        
        # 11. typeof operator
        (r'\btypeof\s*\(', 'typeof_count'),
        
        # 12. record keyword (record class/struct)
        (r'\brecord\s+(?:class|struct)?\s*\w+', 'record_count'),
        
        # 13. with expressions (C# 9.0+)
        (r'\bwith\s*\{', 'with_count'),
        
        # 14. is expressions and pattern matching
        (r'\bis\s+(?:[\w<>,\s\[\]\.?]+|not\s+null|not\s+\w+|>|<|>=|<=|and|or)', 'is_pattern_count'),
        
        # 15. Pattern matching: switch expressions, property patterns, positional patterns
        (r'(?:switch\s*\([^)]+\)|switch\s+expression|case\s+[\w<>,\s\[\]\.?{}]+:|=>\s*[\w<>,\s\[\]\.?{}]+:|\{\s*[\w\s:,]+\s*\})', 'pattern_matching_count'),
        
        # 16. Inheritance / Interface implementation: class A : Base, interface I : IBase
        (r'\b(?:class|interface|struct)\s+\w+\s*:\s*(?:[\w<>,\s\[\]\.?]+(?:\s*,\s*[\w<>,\s\[\]\.?]+)*)', 'inheritance_interface_count'),
        
        # 17. Generic constraints: where T : class, where T : new(), where T : struct, where T : IComparable
        (r'\bwhere\s+[\w\s,]+:\s*(?:[\w<>,\s\[\]\.?]+|class|struct|new\s*\(\s*\)|unmanaged|notnull|default)(?:\s*,\s*(?:[\w<>,\s\[\]\.?]+|class|struct|new\s*\(\s*\)|unmanaged|notnull))*', 'generic_constraints_count'),
        
        # 18. Variance annotations: in/out keywords in generics (e.g., IEnumerable<out T>, IComparer<in T>)
        # Match in/out before type parameters in generic declarations - must be inside <>
        (r'<\s*(?:[\w<>,\s\[\]\.?]*\s*,\s*)*(?:in|out)\s+[A-Z]\w*', 'variance_count'),
        
        # 19. Tuples: (int, string), ValueTuple<T1, T2>, Tuple<T1, T2>, named tuples (int x, string y)
        # Match tuple syntax: (Type1, Type2) or (Type1 name1, Type2 name2) - must have comma
        (r'\((?:\s*(?:[\w<>,\s\[\]\.?]+\s+)?\w+\s*,){1,}\s*(?:[\w<>,\s\[\]\.?]+\s+)?\w+\s*\)', 'tuple_count'),
        (r'\b(?:ValueTuple|Tuple)<', 'tuple_generic_count'),
        
        # 20. Function pointers: delegate*<int, int>, delegate* managed/unmanaged
        (r'\bdelegate\s*\*\s*(?:managed|unmanaged)?\s*(?:\[[^\]]+\])?\s*<', 'function_pointer_count'),
        
        # 21. Type aliases: using Alias = Type, global using Alias = Type
        (r'\busing\s+(?:global\s+)?\w+\s*=\s*[\w<>,\s\[\]\.?]+;', 'type_alias_count'),
        
        # 22. Attributes: [Attribute], [MyAttribute()], [Obsolete("message")]
        (r'\[\s*[\w\.]+(?:\s*\([^)]*\))?\s*\]', 'attribute_count'),
        
        # 23. Unsafe pointer types: int*, void**, MyStruct*, fixed buffers
        # Match type* variable_name pattern (excluding comments and strings)
        (r'\b[\w<>,\s\[\]\.?]+\*\s+[\w]+\b', 'unsafe_pointer_count'),
        (r'\bfixed\s+[\w<>,\s\[\]\.?]+\s+[\w]+\s*\[', 'fixed_buffer_count'),
        (r'\bunsafe\s+(?:class|struct|method|block|\{)', 'unsafe_keyword_count'),
        
        # 24. ref struct / readonly struct / ref readonly modifiers
        (r'\breadonly\s+struct\s+\w+', 'readonly_struct_count'),
        (r'\bref\s+struct\s+\w+', 'ref_struct_count'),
        (r'\bref\s+readonly\s+[\w]+\b', 'ref_readonly_count'),
        
        # 25. Default interface methods: methods with bodies in interfaces (C# 8.0+)
        # Detect method implementations in interfaces: returnType MethodName(...) { ... } or = expression;
        # Look for interface declaration followed by method with body or default implementation
        (r'\binterface\s+\w+[^{]*\{[^}]*?[\w<>,\s\[\]\.?]+\s+[\w]+\s*\([^)]*\)\s*(?:\{[^}]*\}|=>[^;]+;|=)', 'default_interface_method_count'),

        # 26. Nullable annotations: #nullable enable/disable/restore
        (r'\b#nullable\s+(enable|disable|restore)(\s+\w+)?\b', 'nullable_annotation_count'),
    ]
    
    # Type-related patterns (for filtering only, no scoring)
    TYPE_KEYWORDS_PATTERNS = [
        r'\b(delegate|var|dynamic|init|typeof|record|with|is|where|in\s+[A-Z]|out\s+[A-Z]|ref|readonly|unsafe|fixed)\b',
        r':\s*[\w<>,\s\[\]\.?]+\s*[;\)\}]',
        r'as\s+[\w<>,\s\[\]\.?]+',
        r'<[^<>]+>',
        r'\?[;\)\}]',  # Nullable types
        r'!\s*[\.\[\]\(]',  # Null-forgiving operator
        r'typeof\s*\(',
        r'System\.Reflection',
        r'\brecord\s+',
        r'\bwith\s+\{',
        r'\bis\s+',
        r'switch\s*expression',
        r'\b(?:class|interface|struct)\s+\w+\s*:',  # Inheritance/implementation
        r'\bwhere\s+',  # Generic constraints
        r'\busing\s+\w+\s*=',  # Type aliases
        r'\[[\w\.]',  # Attributes
        r'\*\s*\w+',  # Pointer types
        r'\([\w\s,]+\)',  # Tuples (simple detection)
        r'\bdelegate\s*\*',  # Function pointers
    ]
    
    PATCH_ADDITION_PATTERNS = [
        r'.*:\s*[\w<>,\s\[\]\.?]+\s*[;\)\}]',
        r'.*:\s*[A-Z][a-zA-Z]*\s*[;\)\}]',
        r'.*as\s+[A-Z][a-zA-Z]*',
        r'.*<[^<>]*[A-Z][a-zA-Z][^<>]*>',
        r'.*\s+var\s+',
        r'.*\s+dynamic\s+',
        r'.*\s+record\s+',
        r'.*\s+typeof\s*\(',
        r'.*\s+with\s+\{',
        r'.*\s+is\s+',
        r'.*\s+(?:class|interface|struct)\s+\w+\s*:',  # Inheritance/implementation
        r'.*\s+where\s+',  # Generic constraints
        r'.*\s+(?:in|out)\s+[A-Z]',  # Variance
        r'.*\s+using\s+\w+\s*=',  # Type aliases
        r'.*\[[\w\.]',  # Attributes
        r'.*\*\s+\w+',  # Pointer types
        r'.*\([\w\s,]+\)',  # Tuples
        r'.*\s+delegate\s*\*',  # Function pointers
        r'.*\s+(?:ref|readonly)\s+struct',  # Ref/readonly struct
    ]
    
    FP_EXCLUDE_PATTERNS = [
        r'typeof\s+\w+',  # typeof in comments/strings
        r'@type',
        r'console\.log.*type',
        r'button.*type',
        r'input.*type',
    ]

    def __init__(self):
        self.pr_df = None
        self.repo_df = None
        self.pr_commits_df = None
        self.pr_commit_details_df = None
        self.csharp_type_prs = None
        
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Pre-compile all regular expressions."""
        self.compiled_type_keywords = [
            re.compile(p, re.IGNORECASE) for p in self.TYPE_KEYWORDS_PATTERNS
        ]
        self.compiled_patch_add_patterns = [
            re.compile(p) for p in self.PATCH_ADDITION_PATTERNS
        ]
        self.compiled_fp_patterns = [re.compile(p, re.IGNORECASE) for p in self.FP_EXCLUDE_PATTERNS]
        
        # Compiled C# type feature patterns
        self.compiled_type_feature_patterns = [
            (re.compile(pattern, re.MULTILINE | re.IGNORECASE), col_name) 
            for pattern, col_name in self.TYPE_FEATURE_PATTERNS
        ]
        self.TYPE_FEATURE_COLS = [col_name for _, col_name in self.TYPE_FEATURE_PATTERNS]
        
    def load_datasets(self):
        """Load datasets from HuggingFace."""
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
        """Filters all PRs in C# repos and marks them as 'AI' or 'Human'."""
        if self.pr_df.empty or self.repo_df.empty:
            return pd.DataFrame()
            
        print("\nFiltering C# PRs and classifying by agent status...")
        
        cs_repos = self.repo_df[
            self.repo_df['language'].str.contains('C#', case=False, na=False)
        ]
        cs_repo_ids = set(cs_repos['id'].tolist())

        # Filter all PRs in C# repos
        all_cs_prs = self.pr_df[
            self.pr_df['repo_id'].isin(cs_repo_ids)
        ].copy()
        
        # Classify PRs
        all_cs_prs['group'] = all_cs_prs['agent'].apply(
            lambda x: 'AI' if x in self.AI_AGENTS else 'Human'
        )

        print(f"   Found {len(all_cs_prs):,} total PRs in C# repos ({len(all_cs_prs[all_cs_prs['group'] == 'AI']):,} AI, {len(all_cs_prs[all_cs_prs['group'] == 'Human']):,} Human)")
        return all_cs_prs

    def _has_fp(self, text: str) -> bool:
        """Check if text contains false positive patterns."""
        if pd.isna(text): 
            return False
        return any(p.search(text) for p in self.compiled_fp_patterns)

    def _has_type_keywords(self, text: str) -> bool:
        """Check if text contains type-related keywords."""
        if pd.isna(text): 
            return False
        return any(p.search(text) for p in self.compiled_type_keywords)

    def _has_patch_type_patterns(self, patch: str) -> bool:
        """Check if patch contains type-related additions."""
        if pd.isna(patch): 
            return False
        for line in patch.splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                line_content = line[1:].strip()
                for pattern in self.compiled_patch_add_patterns:
                    if pattern.search(line_content):
                        return True
        return False

    def _is_valid_cs_file(self, filename: str) -> bool:
        """Check if file is a valid C# file."""
        if pd.isna(filename): 
            return False
        path = Path(filename)
        return path.suffix.lower() in self.CS_EXTENSIONS

    def _remove_comments(self, code: str) -> str:
        """Remove single-line and multi-line comments from code."""
        if pd.isna(code) or not code:
            return code or ""
        # Remove single-line comments
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code

    def identify_type_related_prs(self, all_cs_prs: pd.DataFrame) -> pd.DataFrame:
        """Identify type-related PRs and count features."""
        if all_cs_prs.empty:
            print("No PRs to analyze.")
            return pd.DataFrame()

        print("\nIdentifying type-related PRs with feature detection...")

        # 1. False Positive removal
        print("   Applying FP filters...")
        all_cs_prs['has_fp'] = (
            all_cs_prs['title'].apply(self._has_fp) |
            all_cs_prs['body'].apply(self._has_fp)
        )
        filtered_prs = all_cs_prs[~all_cs_prs['has_fp']].copy()

        # 2. Check for type-related keywords
        print("   Checking for type-related keywords...")
        filtered_prs['has_type_keywords'] = (
            filtered_prs['title'].apply(self._has_type_keywords) |
            filtered_prs['body'].apply(self._has_type_keywords)
        )

        # 3. Patch analysis with type feature counting
        print("   Analyzing patches + counting all features...")
        cs_details = self.pr_commit_details_df[
            self.pr_commit_details_df['filename'].apply(self._is_valid_cs_file)
        ].copy()

        def analyze_patch_group(group):
            """Analyze a group of patches and count C# type features."""
            feature_counts = {col: 0 for col in self.TYPE_FEATURE_COLS}

            # Concatenate all valid C# patches for this PR
            full_patch_text = "\n".join(group['patch'].dropna().tolist())
            
            if full_patch_text:
                # Remove comments to avoid false positives
                patch_no_comments = self._remove_comments(full_patch_text)
                
                # Count features on ADDED lines only
                added_lines = "\n".join([
                    line[1:] for line in patch_no_comments.splitlines() 
                    if line.startswith('+') and not line.startswith('+++')
                ])
                
                if not added_lines.strip():
                    return pd.Series(feature_counts)
                
                # Count each feature pattern
                for pattern, col_name in self.compiled_type_feature_patterns:
                    # Use finditer for patterns that need position info, findall for others
                    if col_name in ('tuple_count', 'unsafe_pointer_count'):
                        # Use finditer to get positions directly (much faster than findall + find)
                        matches = list(pattern.finditer(added_lines))
                    else:
                        matches = pattern.findall(added_lines)
                    
                    # Special handling for generics to avoid false positives
                    if col_name == 'generics_count':
                        # Filter out comparison operators
                        filtered_matches = [
                            m for m in matches 
                            if '<' in m and '>' in m and m.count('<') == m.count('>')
                        ]
                        feature_counts[col_name] += len(filtered_matches)
                    # Special handling for nullable types to avoid ?? operator
                    elif col_name == 'nullable_count':
                        filtered_matches = [m for m in matches if '??' not in m]
                        feature_counts[col_name] += len(filtered_matches)
                    # Special handling for null-forgiving operator
                    elif col_name == 'null_forgiving_count':
                        filtered_matches = []
                        for match in matches:
                            if '!' in match:
                                parts = match.split('!')
                                if len(parts) >= 2:
                                    expr = parts[0].strip()
                                    if expr and (expr[-1].isalnum() or expr[-1] in '.?)]'):
                                        filtered_matches.append(match)
                        feature_counts[col_name] += len(filtered_matches)
                    # Special handling for variance - pattern already ensures generic context
                    elif col_name == 'variance_count':
                        # Pattern already checks for < context, just count matches
                        feature_counts[col_name] += len(matches)
                    # Special handling for tuples - ensure it's not just a method call
                    elif col_name == 'tuple_count':
                        filtered_count = 0
                        for match_obj in matches:
                            match = match_obj.group(0)
                            start_pos = match_obj.start()
                            end_pos = match_obj.end()
                            
                            # Ensure it has comma and looks like type tuple, not method call
                            if ',' in match:
                                # Check context - tuples are often in variable declarations or return types
                                before_start = max(0, start_pos - 50)
                                before_context = added_lines[before_start:start_pos]
                                # If preceded by =, :, =>, or (type declaration keywords), likely tuple
                                if re.search(r'(?:=\s*|:\s*|=>\s*|\(|return\s+|ref\s+|out\s+|in\s+)', before_context[-20:]):
                                    filtered_count += 1
                                # Also accept if it's clearly a tuple type pattern (type, type) without method name before
                                elif not re.search(r'[a-zA-Z_]\w*\s*\($', before_context[-10:]):
                                    # Check if it's followed by variable name or semicolon (tuple declaration)
                                    after_end = min(len(added_lines), end_pos + 20)
                                    after_context = added_lines[end_pos:after_end]
                                    if re.search(r'^\s*[\w]|^\s*;|^\s*\)|^\s*$', after_context):
                                        filtered_count += 1
                        feature_counts[col_name] += filtered_count
                    # Special handling for attributes - avoid array indexing [0], [1] etc.
                    elif col_name == 'attribute_count':
                        filtered_matches = [
                            m for m in matches 
                            if not re.match(r'\[\s*\d+\s*\]', m)  # Exclude numeric array indices
                            and not re.match(r'\[[\'\"][^\'"]+[\'\"]\]', m)  # Exclude string array indices
                            and re.search(r'[A-Z]', m)  # Attributes typically start with capital letter
                        ]
                        feature_counts[col_name] += len(filtered_matches)
                    # Special handling for pointer types - avoid multiplication
                    elif col_name == 'unsafe_pointer_count':
                        filtered_count = 0
                        for match_obj in matches:
                            match = match_obj.group(0)
                            start_pos = match_obj.start()
                            end_pos = match_obj.end()
                            
                            # Ensure it's not multiplication: x * y
                            before_start = max(0, start_pos - 10)
                            before_context = added_lines[before_start:start_pos]
                            after_end = min(len(added_lines), end_pos + 10)
                            after_context = added_lines[end_pos:after_end]
                            
                            # If there's a type before and variable after, it's likely a pointer declaration
                            if re.search(r'\b[\w<>,\s\[\]\.?]+\*\s+\w+', match):
                                # Check it's not multiplication context
                                if not (before_context.strip() and before_context.strip()[-1].isalnum() and 
                                       after_context.strip() and after_context.strip()[0].isalnum()):
                                    filtered_count += 1
                        feature_counts[col_name] += filtered_count
                    else:
                        feature_counts[col_name] += len(matches)
            
            return pd.Series(feature_counts)

        # Process patches with progress indication
        print(f"   Processing {cs_details['pr_id'].nunique():,} PRs with patches...")
        patch_stats = cs_details.groupby('pr_id').apply(analyze_patch_group, include_groups=False)
        
        # Merge all patch stats back into the PR dataframe
        filtered_prs = filtered_prs.merge(patch_stats, left_on='id', right_index=True, how='left')

        # Fill NaNs for all count columns
        for col in self.TYPE_FEATURE_COLS:
            filtered_prs[col] = filtered_prs[col].fillna(0).astype(int)

        # 4. C# file count
        cs_file_count = cs_details.groupby('pr_id').size().to_dict()
        filtered_prs['cs_file_count'] = filtered_prs['id'].map(cs_file_count).fillna(0).astype(int)
        filtered_prs['has_cs_files'] = filtered_prs['cs_file_count'] > 0

        # Calculate total feature count
        filtered_prs['total_feature_count'] = filtered_prs[self.TYPE_FEATURE_COLS].sum(axis=1)

        # Check if patches contain type patterns
        print("   Checking for type patterns in patches...")
        patch_type_flags = cs_details.groupby('pr_id').apply(
            lambda group: any(self._has_patch_type_patterns(patch) for patch in group['patch'] if pd.notna(patch)),
            include_groups=False
        ).to_dict()
        filtered_prs['has_patch_type_patterns'] = filtered_prs['id'].map(patch_type_flags).fillna(False)
        
        # 5. Final filtering: require C# files AND (type features OR type keywords/patterns) AND NOT has_fp
        type_prs = filtered_prs[
            filtered_prs['has_cs_files'] &
            ~filtered_prs['has_fp'] &
            (
                (filtered_prs['total_feature_count'] > 0) |
                filtered_prs['has_type_keywords'] |
                filtered_prs['has_patch_type_patterns']
            )
        ].copy()

        # Add detection method based on what triggered inclusion
        type_prs['detection_method'] = ''
        type_prs.loc[type_prs['total_feature_count'] > 0, 'detection_method'] += 'features|'
        type_prs.loc[type_prs['has_type_keywords'], 'detection_method'] += 'type_keywords|'
        type_prs.loc[type_prs['has_patch_type_patterns'], 'detection_method'] += 'patch_patterns|'
        type_prs['detection_method'] = type_prs['detection_method'].str.rstrip('|')

        print(f"\n   Found {len(type_prs):,} high-confidence type-related PRs.")
        return type_prs

    def enrich_with_commit_stats(self, type_prs: pd.DataFrame) -> pd.DataFrame:
        """Enrich PRs with commit statistics and collect patches."""
        if type_prs.empty:
            return pd.DataFrame()
        
        print("\nEnriching with commit stats and collecting full patches...")
        cs_details = self.pr_commit_details_df[
            self.pr_commit_details_df['pr_id'].isin(type_prs['id']) &
            self.pr_commit_details_df['filename'].apply(self._is_valid_cs_file)
        ]

        stats = cs_details.groupby('pr_id').agg(
            additions=('additions', 'sum'),
            deletions=('deletions', 'sum'),
            changes=('changes', 'sum'),
            cs_files_changed=('filename', 'nunique')
        )

        def collect_patches(group):
            """Collect all patches for a PR."""
            patches = []
            for _, row in group.iterrows():
                if pd.notna(row['patch']) and row['patch'].strip():
                    header = f"=== {row['filename']} (+{row['additions']}/-{row['deletions']}) ==="
                    patches.append(f"{header}\n{row['patch']}")
            return "\n\n".join(patches) if patches else ""

        patch_text = cs_details.groupby('pr_id').apply(collect_patches, include_groups=False).rename('patch_text')

        enriched = type_prs.merge(stats, left_on='id', right_index=True, how='left') \
                          .merge(patch_text, left_on='id', right_index=True, how='left')

        for col in ['additions', 'deletions', 'changes', 'cs_files_changed']:
            enriched[col] = enriched[col].fillna(0).astype(int)
        enriched['patch_text'] = enriched['patch_text'].fillna('')

        return enriched

    def export_results(self, enriched_prs: pd.DataFrame, output_file: str):
        """Export results to CSV and generate summary JSON."""
        if enriched_prs.empty:
            print("No results to export.")
            return

        print(f"\nExporting to {output_file}...")
        export_cols = [
            'id', 'number', 'title', 'body', 'agent', 'group', 'state', 'created_at', 'merged_at',
            'repo_id', 'html_url', 'additions', 'deletions', 'changes', 'cs_files_changed',
        ] + self.TYPE_FEATURE_COLS + [
            'total_feature_count', 'detection_method', 'patch_text'
        ]
        
        export_cols = [col for col in export_cols if col in enriched_prs.columns]
            
        enriched_prs[export_cols].to_csv(output_file, index=False)
        print(f"   Exported {len(enriched_prs):,} PRs")

        # Calculate agent-specific feature statistics
        agent_feature_stats = {}
        for agent in enriched_prs['agent'].unique():
            agent_data = enriched_prs[enriched_prs['agent'] == agent]
            agent_feature_stats[agent] = {
                'pr_count': len(agent_data),
                'total_features': int(agent_data['total_feature_count'].sum()),
                'feature_counts': {
                    col: int(agent_data[col].sum()) for col in self.TYPE_FEATURE_COLS
                },
                'avg_features_per_pr': round(
                    agent_data['total_feature_count'].mean() if len(agent_data) > 0 else 0, 
                    4
                ),
                'feature_density': {
                    col: round(agent_data[col].mean() if len(agent_data) > 0 else 0, 4)
                    for col in self.TYPE_FEATURE_COLS
                }
            }
        
        # Calculate group-specific feature statistics
        group_feature_stats = {}
        for group in enriched_prs['group'].unique():
            group_data = enriched_prs[enriched_prs['group'] == group]
            group_feature_stats[group] = {
                'pr_count': len(group_data),
                'total_features': int(group_data['total_feature_count'].sum()),
                'feature_counts': {
                    col: int(group_data[col].sum()) for col in self.TYPE_FEATURE_COLS
                },
                'avg_features_per_pr': round(
                    group_data['total_feature_count'].mean() if len(group_data) > 0 else 0, 
                    4
                ),
                'feature_density': {
                    col: round(group_data[col].mean() if len(group_data) > 0 else 0, 4)
                    for col in self.TYPE_FEATURE_COLS
                }
            }
        
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'total_type_prs': len(enriched_prs),
            'by_group': enriched_prs['group'].value_counts().to_dict(),
            'by_agent': enriched_prs['agent'].value_counts().to_dict(),
            'total_features': int(enriched_prs['total_feature_count'].sum()),
            'feature_totals': {
                col: int(enriched_prs[col].sum()) for col in self.TYPE_FEATURE_COLS
            },
            'agent_feature_stats': agent_feature_stats,
            'group_feature_stats': group_feature_stats,
        }
        
        summary_file = output_file.replace('.csv', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   Summary saved to {summary_file}")

    def run_pipeline(self, output_file: str = 'cs_type_prs_all_groups_final.csv'):
        """Run the complete extraction pipeline."""
        print("="*80)
        print("C# Type-Related PR Extraction (Type Feature Analysis)")
        print("="*80)

        self.load_datasets()
        all_cs_prs = self.filter_prs_by_agent_status() 
        type_prs = self.identify_type_related_prs(all_cs_prs)
        enriched = self.enrich_with_commit_stats(type_prs)
        self.export_results(enriched, output_file)
        self.csharp_type_prs = enriched

        print("\n" + "="*80)
        print("Pipeline completed!")
        print("="*80)
        return enriched


def main():
    """Main entry point."""
    extractor = CSharpTypePRExtractor()
    results = extractor.run_pipeline()
    return results


if __name__ == '__main__':
    main()
