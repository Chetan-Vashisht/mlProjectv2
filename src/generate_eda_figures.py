import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def generate_all_figures(data_path='data/processed/loan_data_cleaned.csv',
                         output_dir='reports/figures'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    print(f"Loaded {df.shape[0]} rows for EDA")

    # --- Chart 1: Distribution of Interest Rate ---
    plt.figure(figsize=(8, 5))
    plt.hist(df['int_rate'], bins=25, color='#2196F3', edgecolor='white', alpha=0.85)
    plt.title('Distribution of Interest Rate', fontsize=14)
    plt.xlabel('Interest Rate (%)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_int_rate_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_int_rate_distribution.png")

    # --- Chart 2: Interest Rate by Grade (box plot) ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='grade', y='int_rate', order=sorted(df['grade'].unique()),
                palette='viridis')
    plt.title('Interest Rate by LendingClub Grade', fontsize=14)
    plt.xlabel('Grade')
    plt.ylabel('Interest Rate (%)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_rate_by_grade.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_rate_by_grade.png")

    # --- Chart 3: DTI vs Interest Rate (scatter) ---
    plt.figure(figsize=(8, 5))
    plt.scatter(df['dti'], df['int_rate'], alpha=0.1, s=5, color='#FF5722')
    plt.xlabel('Debt-to-Income Ratio')
    plt.ylabel('Interest Rate (%)')
    plt.title('DTI vs Interest Rate', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_dti_vs_rate.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_dti_vs_rate.png")

    # --- Chart 4: FICO Score vs Interest Rate (scatter) ---
    df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    plt.figure(figsize=(8, 5))
    plt.scatter(df['fico_avg'], df['int_rate'], alpha=0.1, s=5, color='#4CAF50')
    plt.xlabel('FICO Average Score')
    plt.ylabel('Interest Rate (%)')
    plt.title('FICO Score vs Interest Rate', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_fico_vs_rate.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_fico_vs_rate.png")

    # --- Chart 5: Correlation Heatmap ---
    numeric_cols = ['int_rate', 'loan_amnt', 'annual_inc', 'dti', 'open_acc',
                    'pub_rec', 'revol_util', 'total_acc', 'fico_avg', 'mort_acc']
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5)
    plt.title('Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_correlation_heatmap.png")

    # --- Chart 6: Pairplot of top 5 correlated features ---
    # Pick top 5 features most correlated with int_rate
    corr_with_target = df[numeric_cols].corr()['int_rate'].drop('int_rate').abs().sort_values(ascending=False)
    top5 = corr_with_target.head(5).index.tolist()
    print(f"  Top 5 correlated features: {top5}")
    # Subsample for pairplot performance (pairplot on 500k+ rows is too slow)
    sample_size = min(5000, len(df))
    df_sample = df[['int_rate'] + top5].sample(n=sample_size, random_state=42)
    g = sns.pairplot(df_sample, corner=True, plot_kws={'alpha': 0.3, 's': 10})
    g.figure.suptitle('Pairplot — Top 5 Correlated Features', y=1.02, fontsize=14)
    g.figure.savefig(f'{output_dir}/06_pairplot_top5.png', dpi=100, bbox_inches='tight')
    plt.close('all')
    print("  ✓ 06_pairplot_top5.png")

    # --- Chart 7: Mean Interest Rate by Grade ---
    plt.figure(figsize=(8, 5))
    df.groupby('grade')['int_rate'].mean().sort_index().plot(
        kind='bar', color='#673AB7', edgecolor='white')
    plt.title('Mean Interest Rate by Grade', fontsize=14)
    plt.xlabel('Grade')
    plt.ylabel('Mean Interest Rate (%)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_mean_rate_by_grade.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 07_mean_rate_by_grade.png")

    print(f"\nAll {len(os.listdir(output_dir))} EDA figures saved to {output_dir}/")

if __name__ == "__main__":
    generate_all_figures()
