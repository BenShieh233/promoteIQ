import streamlit as st
import pandas as pd

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
            st.success(f"{label} 上传成功！")
            return data
        except Exception as e:
            st.error(f"无法读取 {label}: {e}")
    return pd.DataFrame

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