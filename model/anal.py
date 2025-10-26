import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def significance_marker(p):
    """显著性标记"""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

# ---------------------------
# 数据准备
# ---------------------------

def load_and_align_data(morphology_path, embedding_path):
    """
    加载并对齐两个CSV文件
    """
    print("=== 数据加载与对齐 ===")

    # 读取数据
    # 读取
    morph_df = pd.read_csv(morphology_path)
    embed_df = pd.read_csv(embedding_path)

    print(f"  形态学特征: {morph_df.shape}, Embedding: {embed_df.shape}")

    # 文件名列（优先 'filename'）
    morph_file_col = 'filename' if 'filename' in morph_df.columns else morph_df.columns[0]
    embed_file_col = 'filename' if 'filename' in embed_df.columns else embed_df.columns[0]

    morph_names = set(morph_df[morph_file_col].dropna())
    embed_names = set(embed_df[embed_file_col].dropna())
    common_names = morph_names & embed_names

    print(f"  文件名交集: {len(common_names)}, "
          f"仅形态学: {len(morph_names - embed_names)}, "
          f"仅embedding: {len(embed_names - morph_names)}")

    # 直接用交集索引
    morph_df = morph_df[morph_df[morph_file_col].isin(common_names)].set_index(morph_file_col)
    embed_df = embed_df[embed_df[embed_file_col].isin(common_names)].set_index(embed_file_col)

    # 对齐（索引排序保证一致）
    common_names = sorted(common_names)
    morph_df = morph_df.loc[common_names]
    embed_df = embed_df.loc[common_names]

    print(f"  对齐后: 形态学 {morph_df.shape}, Embedding {embed_df.shape}")

    return morph_df, embed_df

def prepare_analysis_data(morph_df: pd.DataFrame, embed_df: pd.DataFrame):
    """
    从对齐后的数据中提取 embedding 与形态学特征
    """
    print("\n=== 准备分析数据 ===")

    # embedding 列识别
    embedding_cols = [
        c for c in embed_df.columns
        if c.startswith('feature_') or 'feature_' in c.lower()
    ]
    if not embedding_cols:
        numeric_cols = embed_df.select_dtypes(include=[np.number]).columns
        embedding_cols = numeric_cols[-64:] if len(numeric_cols) > 64 else numeric_cols
    print(f"  embedding列数: {len(embedding_cols)}")

    # 形态学列识别
    morph_keywords = [
        'total_length', 'total_area', 'total_volume', 'radial_distance',
        'total_width', 'total_height', 'total_depth', 'number_of_sections', 'number_of_bifurcations',
        'number_of_leaves', 'number_of_neurites', 'sholl_max_crossings',
        'aspect_ratio', 'circularity', 'shape_factor', 'tortuosity_mean'
        , 'bifurcation_angle_mean', 'partition_asymmetry_mean',
        'sibling_ratio_mean'
    ]

    morph_cols = [c for c in morph_df.columns
                  if any(k in c.lower() for k in morph_keywords)]

    if not morph_cols:
        morph_cols = morph_df.select_dtypes(include=[np.number]).columns
    print(f"  形态学特征列数: {len(morph_cols)}")

    # 提取
    embeddings = embed_df[embedding_cols].to_numpy()
    features_df = morph_df[morph_cols].copy()

    print(f"  Embeddings形状: {embeddings.shape}")
    print(f"  形态学特征形状: {features_df.shape}")

    return embeddings, features_df


def perform_pca_analysis(embeddings, n_components=32):
    """
    执行PCA分析
    """
    print("\n=== PCA 分析 ===")

    # 确定合适的主成分数量
    n_components = min(n_components, embeddings.shape[1], embeddings.shape[0] - 1)
    print(f"使用主成分数量: {n_components}")

    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(embeddings)
    pc_names = [f"PC{i + 1}" for i in range(pcs.shape[1])]

    print(f"主成分解释方差比:")
    for i, ratio in enumerate(pca.explained_variance_ratio_[:10]):
        print(f"  {pc_names[i]}: {ratio:.4f} ({np.cumsum(pca.explained_variance_ratio_)[i]:.4f} cumulative)")

    return pca, pcs, pc_names


def calculate_correlations(feature_df, pcs, pc_names, alpha=0.05, adjust_pvals=True):
    """
    计算形态学特征与主成分的相关性，包含p值处理和多重检验校正

    Parameters:
    -----------
    feature_df : DataFrame
        形态学特征数据框，每行一个样本，每列一个特征
    pcs : ndarray
        主成分得分矩阵，每行一个样本，每列一个主成分
    pc_names : list
        主成分名称列表
    alpha : float, default=0.05
        显著性水平
    adjust_pvals : bool, default=True
        是否进行多重检验校正

    Returns:
    --------
    corr_df : DataFrame
        相关系数矩阵
    pval_df : DataFrame
        p值矩阵（原始或校正后）
    sig_df : DataFrame
        显著性标记矩阵
    pval_original_df : DataFrame
        原始p值矩阵
    """
    print("\n=== 相关性分析 ===")

    # 输入检查
    if feature_df.shape[0] != pcs.shape[0]:
        raise ValueError(f"样本数不一致: feature_df 有 {feature_df.shape[0]} 行, pcs 有 {pcs.shape[0]} 行")

    feature_names = feature_df.columns.tolist()
    n_features = len(feature_names)
    n_components = pcs.shape[1]

    if len(pc_names) != n_components:
        raise ValueError(f"pc_names 长度 {len(pc_names)} 与 pcs 列数 {n_components} 不一致")

    # 初始化矩阵
    corr_matrix = np.zeros((n_features, n_components))
    pval_matrix = np.zeros((n_features, n_components))

    print(f"计算 {n_features} 个特征与 {n_components} 个主成分的相关性...")
    print(f"样本量: {feature_df.shape[0]}")

    # 遍历计算相关性
    for i, feat in enumerate(feature_names):
        for j in range(n_components):
            try:
                r, p = pearsonr(feature_df[feat], pcs[:, j])
                corr_matrix[i, j] = r
                pval_matrix[i, j] = p
            except Exception as e:
                print(f"警告: 计算 {feat} 与 {pc_names[j]} 相关性时出错: {e}")
                corr_matrix[i, j] = np.nan
                pval_matrix[i, j] = np.nan

    # 创建结果DataFrame
    corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=pc_names)
    pval_original_df = pd.DataFrame(pval_matrix, index=feature_names, columns=pc_names)

    # p值处理 - 使用兼容性更好的多重检验校正
    if adjust_pvals:
        pval_df = _adjust_pvalues(pval_original_df)
        print("已进行多重检验校正")
    else:
        pval_df = pval_original_df.copy()
        print("使用原始p值，未进行多重检验校正")

    # 创建显著性标记矩阵
    sig_df = _create_significance_matrix(pval_df, alpha)

    # 报告极端p值的情况
    extreme_pvals = (pval_original_df < 1e-100).sum().sum()
    if extreme_pvals > 0:
        print(f"注意: 发现 {extreme_pvals} 个极端p值 (< 1e-100)，建议在报告中简化处理")

    # 报告显著相关的数量
    significant_corrs = (pval_df < alpha).sum().sum()
    print(f"在 {alpha} 水平下显著相关的数量: {significant_corrs}/{n_features * n_components}")

    return corr_df, pval_df, sig_df, pval_original_df


def _adjust_pvalues(pval_df, method='fdr_bh'):
    """
    多重检验校正 - 兼容不同scipy版本
    """
    pval_matrix = pval_df.values

    try:
        # 尝试使用较新版本的scipy
        from scipy.stats import false_discovery_control
        pval_flat = pval_matrix.flatten()
        pval_adjusted_flat = false_discovery_control(pval_flat)
        pval_adjusted = pval_adjusted_flat.reshape(pval_matrix.shape)

    except ImportError:
        # 回退到statsmodels或scipy的旧方法
        try:
            import statsmodels.stats.multitest as multi
            pval_flat = pval_matrix.flatten()
            _, pval_adjusted_flat, _, _ = multi.multipletests(pval_flat, alpha=0.05, method=method)
            pval_adjusted = pval_adjusted_flat.reshape(pval_matrix.shape)
            print("使用statsmodels进行多重检验校正")

        except ImportError:
            # 最后回退到简单的Bonferroni校正
            print("警告: 未找到statsmodels，使用Bonferroni校正")
            n_tests = pval_matrix.size
            pval_adjusted = np.minimum(pval_matrix * n_tests, 1.0)

    return pd.DataFrame(pval_adjusted, index=pval_df.index, columns=pval_df.columns)


def _create_significance_matrix(pval_df, alpha=0.05):
    """
    创建显著性标记矩阵
    """
    sig_df = pd.DataFrame('', index=pval_df.index, columns=pval_df.columns)

    for i in range(pval_df.shape[0]):
        for j in range(pval_df.shape[1]):
            p_val = pval_df.iloc[i, j]
            if np.isnan(p_val):
                sig_df.iloc[i, j] = 'NaN'
            elif p_val < alpha:
                if p_val < 0.001:
                    sig_df.iloc[i, j] = '***'
                elif p_val < 0.01:
                    sig_df.iloc[i, j] = '**'
                elif p_val < 0.05:
                    sig_df.iloc[i, j] = '*'
            else:
                sig_df.iloc[i, j] = 'ns'

    return sig_df
# def calculate_correlations(feature_df, pcs, pc_names):
#
#     """
#     计算形态学特征与主成分的相关性
#     """
#     print("\n=== 相关性分析 ===")
#     # 输入检查
#     if feature_df.shape[0] != pcs.shape[0]:
#         raise ValueError(f"样本数不一致: feature_df 有 {feature_df.shape[0]} 行, pcs 有 {pcs.shape[0]} 行")
#
#     feature_names = feature_df.columns.tolist()
#     n_features = len(feature_names)
#     n_components = pcs.shape[1]
#
#     if len(pc_names) != n_components:
#         raise ValueError(f"pc_names 长度 {len(pc_names)} 与 pcs 列数 {n_components} 不一致")
#
#     # 初始化矩阵
#     corr_matrix = np.zeros((n_features, n_components))
#     pval_matrix = np.zeros((n_features, n_components))
#
#     # 遍历计算相关性
#     for i, feat in enumerate(feature_names):
#         for j in range(n_components):   # 修复点
#             r, p = pearsonr(feature_df[feat], pcs[:, j])
#             corr_matrix[i, j] = r
#             pval_matrix[i, j] = p
#
#     corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=pc_names)
#     pval_df = pd.DataFrame(pval_matrix, index=feature_names, columns=pc_names)
#
#     print(f"相关性矩阵形状: {corr_df.shape}")
#     return corr_df, pval_df

def create_visualizations(corr_df, pval_df, pca, pcs, features_df, pc_names):
    """
    创建所有可视化图表
    """
    print("\n=== 创建可视化 ===")

    # 创建显著性标记
    annot_df = pval_df.applymap(significance_marker)

    # 1. 相关性热图
    plt.figure(figsize=(max(10, len(pc_names)//2), max(6, len(corr_df.index)//2)))
    sns.heatmap(corr_df,
                annot=annot_df,
                fmt="",
                cmap="RdBu_r",
                center=0,
                cbar_kws={"label": "Correlation Coefficient"},
                square=True,
                annot_kws={"size": 8})
    plt.title(f'Correlation between Morphological Features and Embedding PCs\n(n=1741 cells)',
              fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. PCA方差解释图
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    components_to_plot = min(20, len(pca.explained_variance_ratio_))
    plt.bar(range(1, components_to_plot + 1), pca.explained_variance_ratio_[:components_to_plot])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), 'o-')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Variance Explained')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pca_variance.png', dpi=300, bbox_inches='tight')
    plt.show()


    # 3. Top相关性散点图
    N = min(8, len(corr_df.index) * len(corr_df.columns))
    top_corr = corr_df.abs().unstack().sort_values(ascending=False).head(N)

    print("corr_df 行索引:", corr_df.index.tolist()[:10])
    print("pval_df 行索引:", pval_df.index.tolist()[:10])
    print("top_corr 索引示例:", list(top_corr.index[:10]))

    print(f"\nTop {len(top_corr)} 相关性:")
    for idx, corr_value in top_corr.items():
        pc, feat = idx  # 注意这里改成 PC 在前，feature 在后
        p_value = pval_df.loc[feat, pc]
        print(f"  {feat} vs {pc}: r = {corr_value:.3f}, p = {p_value:.3e}")

    # 创建散点图
    n_plots = len(top_corr)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for idx, (ax, (corr_idx, corr_value)) in enumerate(zip(axes, top_corr.items())):
        if idx >= n_plots:
            ax.set_visible(False)
            continue

        pc, feat = corr_idx  # 这里保持一致
        r = corr_df.loc[feat, pc]
        p = pval_df.loc[feat, pc]
        pc_index = pc_names.index(pc)

        # 计算置信区间
        z = np.arctanh(r)
        se = 1 / np.sqrt(len(features_df) - 3)
        ci_low = np.tanh(z - 1.96 * se)
        ci_high = np.tanh(z + 1.96 * se)

        ax.scatter(pcs[:, pc_index], features_df[feat], alpha=0.6, s=30)
        ax.set_xlabel(f'{pc}\n(Var: {pca.explained_variance_ratio_[pc_index]:.3f})')
        ax.set_ylabel(feat)
        ax.set_title(f'r = {r:.3f} (p = {p:.2e})', fontsize=10)
        ax.text(0.05, 0.95, f'CI: [{ci_low:.3f}, {ci_high:.3f}]',
                transform=ax.transAxes, fontsize=8, verticalalignment='top')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('top_correlation_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    return top_corr




def save_results(corr_df, pval_df, top_corr, pca, pc_names):
    """
    保存分析结果
    """
    print("\n=== 保存分析结果 ===")

    # 2. 保存top相关性结果
    top_results = pd.DataFrame({
        'PC': [idx[0] for idx in top_corr.index],      # PC 在前
        'Feature': [idx[1] for idx in top_corr.index], # Feature 在后
        'Correlation': top_corr.values,
        'P_value': [pval_df.loc[idx[1], idx[0]] for idx in top_corr.index],  # 注意顺序反过来
        'PC_Variance_Explained': [
            pca.explained_variance_ratio_[pc_names.index(idx[0])]
            for idx in top_corr.index
        ],
        'Significance': [
            significance_marker(pval_df.loc[idx[1], idx[0]])
            for idx in top_corr.index
        ]
    })
    top_results.to_csv('top_correlation_results.csv', index=False)

    # 3. 保存完整相关性结果
    full_results = corr_df.unstack().reset_index()
    full_results.columns = ['PC', 'Feature', 'Correlation']
    full_results['P_value'] = pval_df.unstack().values
    full_results['Significance'] = full_results['P_value'].apply(significance_marker)
    full_results.to_csv('full_correlation_analysis.csv', index=False)

    # 4. 保存矩阵
    corr_df.to_csv('correlation_matrix.csv')
    pval_df.to_csv('pvalue_matrix.csv')

    # 5. 保存PCA信息
    pca_info = pd.DataFrame({
        'PC': pc_names,
        'Explained_Variance_Ratio': pca.explained_variance_ratio_,
        'Cumulative_Variance_Ratio': np.cumsum(pca.explained_variance_ratio_)
    })
    pca_info.to_csv('pca_information.csv', index=False)

    print("保存的文件:")
    print("  - aligned_dataset.csv: 对齐后的完整数据")
    print("  - top_correlation_results.csv: Top相关性结果")
    print("  - full_correlation_analysis.csv: 完整相关性分析")
    print("  - correlation_matrix.csv: 相关性矩阵")
    print("  - pvalue_matrix.csv: p值矩阵")
    print("  - pca_information.csv: PCA信息")
    print("  - correlation_heatmap.png: 相关性热图")
    print("  - pca_variance.png: PCA方差图")
    print("  - top_correlation_scatter_plots.png: 散点图")

if __name__ == '__main__':


    morphology_path = r"bil_neuron_morphometry.csv"  # 形态学特征文件
    embedding_path = r"D:\PycharmProjects\B1\dataset\act\neuron_embeddings.csv"     # embedding文件

    # 1. 加载和对齐数据
    morph_df, embed_df = load_and_align_data(morphology_path, embedding_path)

    # 2. 准备分析数据
    embeddings, features_df = prepare_analysis_data(morph_df, embed_df)

    # 3. PCA分析
    pca, pcs, pc_names = perform_pca_analysis(embeddings, n_components=16)

    # 4. 计算相关性
    corr_df, pval_df, _,_ = calculate_correlations(features_df, pcs, pc_names)

    # 5. 创建可视化
    top_corr = create_visualizations(corr_df, pval_df, pca, pcs, features_df, pc_names)

    # 6. 保存结果
    save_results(corr_df, pval_df, top_corr, pca, pc_names)

    print("\n=== 分析完成 ===")
