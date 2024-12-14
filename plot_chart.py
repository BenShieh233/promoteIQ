import plotly.express as px
import pandas as pd

def plot_piechart(data: pd.DataFrame,
                                  column_to_aggregate: str,
                                  selected_date: str,
                                  top_campaigns: list,
                                  other_campaigns: list,
                                  aggregation_column: str,
                                  combine_others: bool = True):
    top_summary = data[data[column_to_aggregate].isin(top_campaigns)]
    top_grouped = top_summary.groupby(column_to_aggregate)[aggregation_column].sum()
    if combine_others:
        others_sum = data[data[column_to_aggregate].isin(other_campaigns)][aggregation_column].sum()
        top_grouped['others'] = others_sum    

    # Calculate percentages
    percentages = (top_grouped / top_grouped.sum()) * 100
    # Total Sales 百分比饼图
    fig = px.pie(
        names=percentages.index,
        values=percentages.values,
        title=f"{aggregation_column} 在 {selected_date} 的百分比分布分析",
        color=percentages.index
    )
    return fig
    
def plot_percentage_histogram(data: pd.DataFrame,
                                  column_to_aggregate: str,
                                  selected_date: str,
                                  top_campaigns: list,
                                  aggregation_column: str):
    top_summary = data[data[column_to_aggregate].isin(top_campaigns)]
    top_grouped = top_summary.groupby(column_to_aggregate)[aggregation_column].mean().reset_index()
    # 使用 Plotly Express 创建直方图
    fig = px.bar(
        top_grouped,
        x = top_grouped[column_to_aggregate].astype(str),
        y = top_grouped[aggregation_column],
        title=f"{aggregation_column} 在 {selected_date} 的直方图分析",
        color=top_grouped[column_to_aggregate].astype(str)  # 确保颜色映射也基于字符串
    )

    # 强制设置 x 轴为离散类型
    fig.update_layout(
        xaxis=dict(type="category")
    )

    return fig