import pandas as pd
import re
from typing import Dict, Any
import json
from datetime import datetime
from pathlib import Path

class HumanBaselineAnalyzer:
    """
    Analyzer for human PR baseline data.
    Focuses on any-related changes and advanced type features.
    """
    
    # Advanced type patterns (same as v5)
    ADVANCED_TYPE_PATTERNS_REFINED = [
        # 1. Generics: Type parameters in declarations (avoid JSX false positives)
        (r'(interface|type|class|function|declare\s+function|const\s+\w+\s*:\s*)\s*\w+\s*<[^<>]+>', 'generics_count'),
        
        # 2. Conditional Types: T extends U ? X : Y
        (r'\bextends\b\s*[^?]*\?\s*[^:]*:\s*', 'conditional_type_count'),
        
        # 3. Infer keyword inside conditional types
        (r'\binfer\s+\w+\b', 'infer_count'),
        
        # 4. Mapped Types: {[K in keyof T]: ...}
        (r'\[\s*(readonly\s+)?([A-Za-z_]\w*)\s+in\s+keyof\s+\w+\s*(\s*\?)?\s*\]', 'mapped_type_count'),
        
        # 5. Key Remapping in mapped types (as ...)
        (r'\[[^\]]+in[^\]]+\bas\s+[A-Za-z_]\w+[^\]]*\]', 'key_remap_count'),
        
        # 6. Template Literal Types: `${string}` / `${infer T}`
        (r'`[^`]*\$\{[^}]+}[^`]*`', 'template_literal_type_count'),
        
        # 7. Satisfies operator (TS 4.9+)
        (r'\s+satisfies\s+', 'satisfies_count'),
        
        # 8. As const: literal narrowing
        (r'\bas\s+const\b', 'as_const_count'),
        
        # 9. Non-null Assertion Operator
        (r'!\s*([\.\[(])', 'non_null_assertion_count'),
        
        # 10. Type Guard ('is')
        (r':\s*\(?\s*\w+\s+is\s+\w+', 'type_guard_is_count'),
        
        # 11. Type-Level Operators
        (r'\bkeyof\b', 'keyof_count'),
        (r'\btypeof\b', 'typeof_count'),
        (r'\[[^\]]*\bin\b[^\]]*\]', 'in_operator_count'),
        
        # 12. Intersection & Union Types
        (r'[A-Za-z0-9)\]]\s*&\s*[A-Za-z0-9(\[]', 'intersection_type_count'),
        (r'[A-Za-z0-9)\]]\s*\|\s*[A-Za-z0-9(\[]', 'union_type_count'),
        
        # 13. Indexed Access Types
        (r'\w+\s*\[\s*keyof\s*\w+\s*\]', 'indexed_access_count'),
        
        # 14. Utility Types
        (r'\b(Partial|Required|Readonly|Pick|Omit|Record|ReturnType|InstanceType|Parameters|ConstructorParameters|Awaited|ThisType|Exclude|Extract|NonNullable)\s*<', 'utility_type_usage_count'),
        
        # 15. Recursive type
        (r'type\s+(\w+)\s*=\s*.*\b\1\b', 'recursive_type_count'),
        
        # 16. Const Assertions
        (r'(as\s+const)|(\{[^}]+\}\s+as\s+const)', 'const_assertion_count'),
        
        # 17. Discriminated Union Hints
        (r'\b(kind|type|tag)\s*:\s*(\'[A-Za-z_0-9]+\'|"[A-Za-z_0-9]+")', 'discriminated_union_count'),
        
        # 18. Unique symbol
        (r'\bunique\s+symbol\b', 'unique_symbol_count'),
        
        # 19. Readonly modifier
        (r'\b(readonly\s+([A-Za-z_]\w+|\[))|(\{[^}]*readonly\s+\w+:[^}]*\})', 'readonly_modifier_count'),
        
        # 20. Optional modifiers
        (r'\w+\s*\?:', 'optional_property_count'),
    ]
    
    ANY_REPLACEMENT_RAW_PATTERN = r'^\-\s*.*:\s*([a-zA-Z_][\w\[\]<>]*)\s*[;\)\}]?\s*$\n^\+\s*.*:\s*any\b'
    
    # Type-related patterns for filtering
    PATCH_ADDITION_PATTERNS = [
        r'.*:\s*[a-zA-Z_][\w]*\s*[;\)\}]',
        r'.*:\s*[A-Z][a-zA-Z]*\s*[;\)\}]',
        r'(interface|type)\s+\w+',
        r'.*\s+as\s+[A-Z][a-zA-Z]*',
        r'.*<[^<>]*[A-Z][a-zA-Z][^<>]*>',
        r'.*:\s*any\b',
    ]
    FP_EXCLUDE_PATTERNS = [
        r'type\s*=\s*["\']',
        r'typeof\s+\w+',
        r'@type\s+{',
        r'console\.log.*type',
        r'button.*type',
        r'input.*type',
        r'\.d\.ts\b',
    ]
    
    def __init__(self):
        self.df = None
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile all regular expressions."""
        # Compile advanced patterns
        self.compiled_advanced_patterns = [
            (re.compile(p), col_name) for p, col_name in self.ADVANCED_TYPE_PATTERNS_REFINED
        ]
        self.ADVANCED_COUNT_COLS = [col_name for _, col_name in self.ADVANCED_TYPE_PATTERNS_REFINED]
        
        # Compile any patterns
        self.compiled_any_add = re.compile(r'^\+\s*.*:\s*any\b', re.MULTILINE)
        self.compiled_any_rem = re.compile(r'^\-\s*.*:\s*any\b', re.MULTILINE)
        self.compiled_any_replacement = re.compile(self.ANY_REPLACEMENT_RAW_PATTERN, re.MULTILINE)
        
        # Compile filtering patterns
        self.compiled_patch_add_patterns = [
            re.compile(p) for p in self.PATCH_ADDITION_PATTERNS
        ]
        self.compiled_fp_patterns = [re.compile(p, re.IGNORECASE) for p in self.FP_EXCLUDE_PATTERNS]
    
    def _has_fp(self, text: str) -> bool:
        """Check if text contains false positive patterns."""
        if pd.isna(text): return False
        return any(p.search(text) for p in self.compiled_fp_patterns)

    def _has_patch_type_patterns(self, patch_text: str) -> bool:
        """Check if patch contains type-related additions."""
        if pd.isna(patch_text): return False
        for line in patch_text.splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                line_content = line[1:].strip()
                for pattern in self.compiled_patch_add_patterns:
                    if pattern.search(line_content):
                        return True
        return False
    
    def load_data(self, csv_file: str):
        """Load human PR data from CSV."""
        print(f"Loading data from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df):,} human PRs")
        return self.df
    
    def filter_and_analyze_patches(self):
        """Filter PRs and analyze patches for any-related changes and advanced features."""
        print("\nFiltering and analyzing patches...")
        
        # 1. False Positive removal
        print("   Applying FP filters...")
        self.df['has_fp'] = (
            self.df['title'].apply(self._has_fp) |
            self.df['body'].apply(self._has_fp)
        )
        
        # 2. Analyze patches
        def analyze_single_pr(row):
            patch_text = row['patch_text'] if pd.notna(row['patch_text']) else ""
            
            # Count any-related changes
            any_add = len(self.compiled_any_add.findall(patch_text))
            any_rem = len(self.compiled_any_rem.findall(patch_text))
            any_replacements = len(self.compiled_any_replacement.findall(patch_text))
            
            # Check if patch has type patterns
            has_patch_patterns = self._has_patch_type_patterns(patch_text)
            
            # Count advanced features
            advanced_counts = {col: 0 for col in self.ADVANCED_COUNT_COLS}
            
            if patch_text:
                for pattern, col_name in self.compiled_advanced_patterns:
                    advanced_counts[col_name] = len(pattern.findall(patch_text))
            
            return pd.Series({
                'any_additions': any_add,
                'any_removals': any_rem,
                'any_replacements': any_replacements,
                'has_patch_type_patterns': has_patch_patterns,
                **advanced_counts
            })
        
        analysis = self.df.apply(analyze_single_pr, axis=1)
        self.df = pd.concat([self.df, analysis], axis=1)
        
        # 3. Filter (same logic as AI version)
        print("   Applying final filters...")
        self.df = self.df[
            ~self.df['has_fp'] &
            (
                (self.df['any_additions'] > 0) |
                (self.df['any_removals'] > 0) |
                (self.df['any_replacements'] > 0) |
                (self.df['has_patch_type_patterns'])
            )
        ].copy()
        
        print(f"Analysis complete. Filtered to {len(self.df):,} PRs.")
        return self.df
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        print("\nGenerating summary...")
        
        total_prs = len(self.df)
        
        # Calculate feature density (per PR average)
        feature_density = {}
        for col in self.ADVANCED_COUNT_COLS:
            total = float(self.df[col].sum())
            count = total_prs
            feature_density[col] = round(total / count if count > 0 else 0, 4)
        
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'total_type_prs': total_prs,
            'any_additions_total': int(self.df['any_additions'].sum()),
            'any_removals_total': int(self.df['any_removals'].sum()),
            'any_replacements_total': int(self.df['any_replacements'].sum()),
            'any_replacements_pr': int((self.df['any_replacements'] > 0).sum()),
            'advanced_feature_totals': {col: int(self.df[col].sum()) for col in self.ADVANCED_COUNT_COLS},
            'feature_density': feature_density,
        }
        
        return summary
    
    def export_results(self, output_file: str = 'human_baseline_results.csv', 
                       summary_file: str = 'human_baseline_summary.json'):
        """Export results to CSV and summary to JSON."""
        print(f"\nExporting results to {output_file}...")
        
        # Select columns to export
        export_cols = [
            'id', 'number', 'title', 'body', 'agent', 'state', 'created_at', 'merged_at',
            'repo_id', 'html_url', 'additions', 'deletions', 'changes', 'ts_files_changed',
            'any_additions', 'any_removals', 'any_replacements',
        ] + self.ADVANCED_COUNT_COLS + [
            'patch_text'
        ]
        
        # Filter to existing columns
        export_cols = [col for col in export_cols if col in self.df.columns]
        
        self.df[export_cols].to_csv(output_file, index=False)
        print(f"   Exported {len(self.df):,} PRs")
        
        # Export summary
        summary = self.generate_summary()
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   Summary saved to {summary_file}")
        
        return summary


def main():
    print("="*80)
    print("Human Baseline PR Analysis")
    print("="*80)
    
    analyzer = HumanBaselineAnalyzer()
    
    # Load data
    analyzer.load_data('human_type_prs_final_type_related.csv')
    
    # Filter and analyze patches
    analyzer.filter_and_analyze_patches()
    
    # Export results
    analyzer.export_results()
    
    print("\n" + "="*80)
    print("Analysis completed!")
    print("="*80)


if __name__ == '__main__':
    main()

