import streamlit as st
import pandas as pd
from config import NUMERIC_COLUMNS

# 初始化 Session State
def initialize_state():
    required_states = {
        "uploaded_campaign_summary": pd.DataFrame(),
        "uploaded_sales_report": pd.DataFrame(),
    }
    for key, default_value in required_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def upload_file(label, key):
    """
    处理文件上传的函数。

    Parameters:
        label (str): 上传文件的提示文本。
        key (str): 组件的唯一键。

    Returns:
        pd.DataFrame or None: 上传成功返回 DataFrame，否则返回 None。
    """
    uploaded_file = st.file_uploader(label, type=["xlsx"], key=key)
    if uploaded_file:
        try:
            data = pd.read_excel(uploaded_file, skiprows=3)
            # 检查是否包含 numeric_columns
            missing_columns = [col for col in NUMERIC_COLUMNS if col not in data.columns]
            if missing_columns and key=="campaign_summary":
                st.error(f"上传的文件缺少以下列: {', '.join(missing_columns)}")
                return None
            st.success(f"{label} 上传成功！")
            return data
        except Exception as e:
            st.error(f"无法读取 {label}: {e}")
            return None
    return None

def preview_data(data, title):
    """
    显示数据预览的函数。

    Parameters:
        data (pd.DataFrame): 要显示的数据。
        title (str): 数据标题。
    """
    if data is not None:
        st.subheader(title)
        st.dataframe(data)
    else:
        st.warning(f"{title} 数据为空，无法预览。")

def calculate_by_column_type(selected_column):
    # 判断列类型
    if selected_column in ['CTR', 'ROAS']:  # 均值列
        agg_func = 'mean'
    else:  # 求和列
        agg_func = 'sum'
    return agg_func

def calculate_aggregated_score(filtered_df, weights):
    """
    计算整个时间段内的累计评分。

    Parameters:
        filtered_df (DataFrame): 过滤后的数据。
        weights (dict): 每个指标的权重，例如 {'TOTAL SALES': 0.4, 'CLICKS': 0.3, 'ROAS': 0.3}。

    Returns:
        DataFrame: 包含每个 Campaign 的总评分。
    """
    # 按 Campaign 累计指标
    aggregated_df = filtered_df.groupby('CAMPAIGN ID').agg({
        'TOTAL SALES': 'sum',
        'CLICKS': 'mean',  # 比例型指标取均值
        'SPEND': 'sum',
        # 其他需要的指标...
    }).reset_index()

    # 归一化
    for col in weights.keys():
        aggregated_df[f"{col}_norm"] = (
            aggregated_df[col] - aggregated_df[col].min()
        ) / (aggregated_df[col].max() - aggregated_df[col].min())

    # 计算总评分
    aggregated_df['Total Score'] = sum(
        weights[col] * aggregated_df[f"{col}_norm"] for col in weights.keys()
    )
    
    return aggregated_df[['CAMPAIGN ID', 'Total Score']].sort_values(by='Total Score', ascending=False)

