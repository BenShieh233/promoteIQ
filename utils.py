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