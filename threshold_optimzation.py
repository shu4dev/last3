import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, chi2_contingency, pointbiserialr
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')

def analyze_mixed_correlations(df, target_col=None, figsize=(12, 10)):
    """
    Comprehensive correlation analysis for mixed data types
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with mixed data types
    target_col : str
        Target column name for association analysis
    figsize : tuple
        Figure size for visualizations
    
    Returns:
    --------
    results : dict
        Dictionary containing different correlation matrices and statistics
    """
    
    print("=== MIXED DATA TYPE CORRELATION ANALYSIS ===")
    
    # Identify feature types
    feature_types = identify_feature_types(df)
    
    print(f"Feature type distribution:")
    for ftype, features in feature_types.items():
        print(f"  {ftype.capitalize()}: {len(features)} features")
        if len(features) <= 10:  # Show features if not too many
            print(f"    {features}")
    print()
    
    results = {}
    
    # 1. NUMERICAL-NUMERICAL CORRELATIONS
    if len(feature_types['numerical']) > 1:
        print("1. NUMERICAL-NUMERICAL CORRELATIONS")
        num_df = df[feature_types['numerical']]
        num_corr = num_df.corr(method='pearson')
        results['numerical_corr'] = num_corr
        
        plt.figure(figsize=figsize)
        sns.heatmap(num_corr, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Numerical Features Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Find high correlations
        high_corr = find_high_correlations(num_corr, threshold=0.7)
        if high_corr:
            print("High numerical correlations (|r| >= 0.7):")
            for item in high_corr[:10]:  # Show top 10
                print(f"  {item['Feature_1']} ↔ {item['Feature_2']}: {item['Correlation']:.3f}")
        print()
    
    # 2. CATEGORICAL-CATEGORICAL ASSOCIATIONS (Cramér's V)
    if len(feature_types['categorical']) > 1:
        print("2. CATEGORICAL-CATEGORICAL ASSOCIATIONS (Cramér's V)")
        cat_corr = calculate_cramers_v_matrix(df[feature_types['categorical']])
        results['categorical_assoc'] = cat_corr
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cat_corr, annot=True, cmap='YlOrRd', square=True, 
                   fmt='.3f', cbar_kws={"shrink": .8})
        plt.title("Categorical Features Association Matrix (Cramér's V)")
        plt.tight_layout()
        plt.show()
        print()
    
    # 3. NUMERICAL-CATEGORICAL ASSOCIATIONS (ANOVA F-statistic)
    if len(feature_types['numerical']) > 0 and len(feature_types['categorical']) > 0:
        print("3. NUMERICAL-CATEGORICAL ASSOCIATIONS")
        num_cat_assoc = calculate_numerical_categorical_association(
            df, feature_types['numerical'], feature_types['categorical']
        )
        results['num_cat_assoc'] = num_cat_assoc
        
        if not num_cat_assoc.empty:
            plt.figure(figsize=(10, 6))
            pivot_data = num_cat_assoc.pivot(index='Numerical', columns='Categorical', values='F_statistic')
            sns.heatmap(pivot_data, annot=True, cmap='Greens', fmt='.2f')
            plt.title('Numerical-Categorical Associations (F-statistic)')
            plt.tight_layout()
            plt.show()
        print()
    
    # 4. BINARY-NUMERICAL CORRELATIONS (Point-biserial)
    if len(feature_types['binary']) > 0 and len(feature_types['numerical']) > 0:
        print("4. BINARY-NUMERICAL CORRELATIONS (Point-biserial)")
        bin_num_corr = calculate_point_biserial_matrix(
            df, feature_types['binary'], feature_types['numerical']
        )
        results['binary_num_corr'] = bin_num_corr
        
        if not bin_num_corr.empty:
            plt.figure(figsize=(12, 6))
            pivot_data = bin_num_corr.pivot(index='Binary', columns='Numerical', values='Correlation')
            sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0, fmt='.3f')
            plt.title('Binary-Numerical Correlations (Point-biserial)')
            plt.tight_layout()
            plt.show()
        print()
    
    # 5. TARGET VARIABLE ANALYSIS
    if target_col and target_col in df.columns:
        print(f"5. TARGET VARIABLE ANALYSIS: '{target_col}'")
        target_analysis = analyze_target_associations(df, target_col, feature_types)
        results['target_analysis'] = target_analysis
        
        # Plot target associations
        if not target_analysis.empty:
            plt.figure(figsize=(20, 25))
            
            # Sort by absolute association strength
            target_analysis_sorted = target_analysis.reindex(
                target_analysis['Association_Strength'].abs().sort_values(ascending=False).index
            )
            
            # Create color map based on association type
            colors = {'Pearson': 'blue', 'Point-biserial': 'green', 
                     'Cramers_V': 'orange', 'ANOVA_F': 'red'}
            bar_colors = [colors.get(method, 'gray') for method in target_analysis_sorted['Method']]
            
            plt.barh(range(len(target_analysis_sorted)), 
                    target_analysis_sorted['Association_Strength'].abs(),
                    color=bar_colors, alpha=0.7)
            
            plt.yticks(range(len(target_analysis_sorted)), target_analysis_sorted['Feature'])
            plt.xlabel('Association Strength (absolute value)')
            plt.title(f'Feature Associations with Target: {target_col}')
            plt.grid(axis='x', alpha=0.3)
            
            # Add legend
            handles = [plt.Rectangle((0,0),1,1, color=colors[method], alpha=0.7) 
                      for method in colors.keys()]
            plt.legend(handles, colors.keys(), loc='lower right')
            
            plt.tight_layout()
            plt.show()
        print()
    
    # 6. COMPREHENSIVE SUMMARY
    print("6. COMPREHENSIVE CORRELATION SUMMARY")
    create_comprehensive_summary(results, feature_types)
    
    return results

def identify_feature_types(df):
    """Identify different types of features in the dataframe"""
    feature_types = {
        'numerical': [],
        'categorical': [],
        'binary': [],
        'datetime': [],
        'other': []
    }
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Check if it's binary (only 2 unique values)
            unique_values = df[col].dropna().unique()
            if len(unique_values) == 2 and set(unique_values).issubset({0, 1, True, False}):
                feature_types['binary'].append(col)
            else:
                feature_types['numerical'].append(col)
        elif df[col].dtype == 'object':
            # Check if it's categorical with reasonable number of categories
            if df[col].nunique() <= 20:  # Arbitrary threshold
                feature_types['categorical'].append(col)
            else:
                feature_types['other'].append(col)
        elif df[col].dtype == 'bool':
            feature_types['binary'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            feature_types['datetime'].append(col)
        else:
            feature_types['other'].append(col)
    
    return feature_types

def calculate_cramers_v_matrix(df_cat):
    """Calculate Cramér's V for categorical variables"""
    columns = df_cat.columns
    cramers_v_matrix = pd.DataFrame(np.zeros((len(columns), len(columns))), 
                                   index=columns, columns=columns)
    
    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                cramers_v_matrix.loc[col1, col2] = 1.0
            else:
                cramers_v_matrix.loc[col1, col2] = cramers_v(df_cat[col1], df_cat[col2])
    
    return cramers_v_matrix

def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorical association"""
    try:
        confusion_matrix = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(confusion_matrix)
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
    except:
        return 0.0

def calculate_numerical_categorical_association(df, num_cols, cat_cols):
    """Calculate ANOVA F-statistic for numerical-categorical associations"""
    associations = []
    
    for num_col in num_cols:
        for cat_col in cat_cols:
            try:
                groups = [df[df[cat_col] == cat][num_col].dropna() 
                         for cat in df[cat_col].unique() if len(df[df[cat_col] == cat]) > 0]
                
                # Filter out empty groups
                groups = [group for group in groups if len(group) > 0]
                
                if len(groups) > 1:
                    f_stat, p_value = f_oneway(*groups)
                    associations.append({
                        'Numerical': num_col,
                        'Categorical': cat_col,
                        'F_statistic': f_stat,
                        'p_value': p_value
                    })
            except:
                continue
    
    return pd.DataFrame(associations)

def calculate_point_biserial_matrix(df, binary_cols, num_cols):
    """Calculate point-biserial correlation for binary-numerical associations"""
    correlations = []
    
    for bin_col in binary_cols:
        for num_col in num_cols:
            try:
                # Ensure binary column is 0/1
                binary_data = df[bin_col].astype(int)
                numerical_data = df[num_col]
                
                # Remove NaN values
                mask = ~(binary_data.isna() | numerical_data.isna())
                binary_clean = binary_data[mask]
                numerical_clean = numerical_data[mask]
                
                if len(binary_clean) > 0 and len(binary_clean.unique()) == 2:
                    corr, p_value = pointbiserialr(binary_clean, numerical_clean)
                    correlations.append({
                        'Binary': bin_col,
                        'Numerical': num_col,
                        'Correlation': corr,
                        'p_value': p_value
                    })
            except:
                continue
    
    return pd.DataFrame(correlations)

def analyze_target_associations(df, target_col, feature_types):
    """Analyze associations between all features and target variable"""
    associations = []
    
    # Determine target type
    target_type = None
    for ftype, features in feature_types.items():
        if target_col in features:
            target_type = ftype
            break
    
    for ftype, features in feature_types.items():
        for feature in features:
            if feature == target_col:
                continue
                
            try:
                if ftype == 'numerical' and target_type == 'numerical':
                    corr, p_val = pearsonr(df[feature].dropna(), df[target_col].dropna())
                    associations.append({
                        'Feature': feature,
                        'Method': 'Pearson',
                        'Association_Strength': corr,
                        'p_value': p_val
                    })
                elif ftype == 'numerical' and target_type == 'binary':
                    binary_data = df[target_col].astype(int)
                    mask = ~(binary_data.isna() | df[feature].isna())
                    if mask.sum() > 0:
                        corr, p_val = pointbiserialr(binary_data[mask], df[feature][mask])
                        associations.append({
                            'Feature': feature,
                            'Method': 'Point-biserial',
                            'Association_Strength': corr,
                            'p_value': p_val
                        })
                elif ftype == 'binary' and target_type == 'numerical':
                    binary_data = df[feature].astype(int)
                    mask = ~(binary_data.isna() | df[target_col].isna())
                    if mask.sum() > 0:
                        corr, p_val = pointbiserialr(binary_data[mask], df[target_col][mask])
                        associations.append({
                            'Feature': feature,
                            'Method': 'Point-biserial',
                            'Association_Strength': corr,
                            'p_value': p_val
                        })
                elif ftype == 'categorical':
                    if target_type == 'binary' or target_type == 'categorical':
                        cramers = cramers_v(df[feature], df[target_col])
                        associations.append({
                            'Feature': feature,
                            'Method': 'Cramers_V',
                            'Association_Strength': cramers,
                            'p_value': np.nan
                        })
            except:
                continue
    
    return pd.DataFrame(associations)

def find_high_correlations(corr_matrix, threshold=0.7):
    """Find high correlations from correlation matrix"""
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    return sorted(high_corr, key=lambda x: abs(x['Correlation']), reverse=True)

def create_comprehensive_summary(results, feature_types):
    """Create a comprehensive summary of all associations"""
    print("Summary of strongest associations:")
    
    all_associations = []
    
    # Collect all associations
    if 'numerical_corr' in results:
        high_num_corr = find_high_correlations(results['numerical_corr'], threshold=0.5)
        for item in high_num_corr[:5]:
            all_associations.append({
                'Type': 'Numerical-Numerical',
                'Feature_1': item['Feature_1'],
                'Feature_2': item['Feature_2'],
                'Strength': abs(item['Correlation']),
                'Method': 'Pearson'
            })
    
    if 'target_analysis' in results and not results['target_analysis'].empty:
        target_top = results['target_analysis'].nlargest(5, 'Association_Strength', keep='all')
        for _, row in target_top.iterrows():
            all_associations.append({
                'Type': 'Feature-Target',
                'Feature_1': row['Feature'],
                'Feature_2': 'TARGET',
                'Strength': abs(row['Association_Strength']),
                'Method': row['Method']
            })
    
    # Sort by strength and display
    all_associations.sort(key=lambda x: x['Strength'], reverse=True)
    
    print("\nTop 10 strongest associations:")
    for i, assoc in enumerate(all_associations[:10], 1):
        print(f"{i:2d}. {assoc['Feature_1']} ↔ {assoc['Feature_2']}: "
              f"{assoc['Strength']:.3f} ({assoc['Method']})")

# Quick usage function
def quick_mixed_analysis(df, target_col=None):
    """Quick mixed correlation analysis"""
    return analyze_mixed_correlations(df, target_col=target_col)