from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import plotly.express as px
import os

app = Flask(__name__)

# 首页：上传文件
@app.route('/')
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Campaign Data Visualization</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        </head>
        <body>
            <div class="container mt-5">
                <h1 class="text-center">Upload Your Campaign Data</h1>
                <form action="/upload" method="post" enctype="multipart/form-data" class="text-center mt-4">
                    <input type="file" name="file" required class="form-control-file mb-3">
                    <button type="submit" class="btn btn-primary">Upload</button>
                </form>
            </div>
        </body>
        </html>
    ''')

# 上传并处理文件
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Error: No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "Error: No selected file", 400

    # 保存文件
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    try:
        # 读取数据
        df = pd.read_excel(file_path, skiprows=3)

        # 检查必需列
        required_columns = [
            'DATE', 'CAMPAIGN ID', 'CAMPAIGN STATUS', 'BUDGET AMOUNT', 'IMPRESSIONS', 'CLICKS', 
            'SPEND', 'CPC', 'CPM', 'CTR', 'UNITS SOLD', 'TOTAL SALES', 'ROAS'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return f"Missing columns: {', '.join(missing_columns)}", 400

        # 保存处理后的 DataFrame
        df['DATE'] = pd.to_datetime(df['DATE'])  # 确保日期格式正确
        df.to_csv("processed_data.csv", index=False)

        # 提取唯一的 CAMPAIGN ID 和 CAMPAIGN STATUS
        campaign_ids = df['CAMPAIGN ID'].unique()
        statuses = df['CAMPAIGN STATUS'].unique()

        # 渲染选择页面
        return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Filter Campaigns</title>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            </head>
            <body>
                <div class="container mt-5">
                    <h1 class="text-center">Select Filters and Visualize</h1>
                    <form action="/visualize" method="get" class="mt-4">
                        <div class="form-group">
                            <label for="campaign_ids">Select Campaign IDs:</label>
                            <select name="campaign_ids" id="campaign_ids" multiple size="10" class="form-control">
                                {% for id in campaign_ids %}
                                    <option value="{{ id }}">{{ id }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="statuses">Select Campaign Status:</label>
                            <select name="statuses" id="statuses" multiple size="5" class="form-control">
                                {% for status in statuses %}
                                    <option value="{{ status }}">{{ status }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="parameters">Select Parameters:</label>
                            <select name="parameters" id="parameters" multiple size="10" class="form-control">
                                <option value="BUDGET AMOUNT">BUDGET AMOUNT</option>
                                <option value="IMPRESSIONS">IMPRESSIONS</option>
                                <option value="CLICKS">CLICKS</option>
                                <option value="SPEND">SPEND</option>
                                <option value="CPC">CPC</option>
                                <option value="CPM">CPM</option>
                                <option value="CTR">CTR</option>
                                <option value="UNITS SOLD">UNITS SOLD</option>
                                <option value="TOTAL SALES">TOTAL SALES</option>
                                <option value="ROAS">ROAS</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="date_range">Select Date Range:</label>
                            <input type="date" name="start_date" class="form-control" required>
                            <input type="date" name="end_date" class="form-control mt-2" required>
                        </div>
                        <button type="submit" class="btn btn-success btn-block">Visualize</button>
                    </form>
                </div>
            </body>
            </html>
        ''', campaign_ids=campaign_ids, statuses=statuses)

    except Exception as e:
        return f"Error processing file: {str(e)}", 500

# 数据可视化
@app.route('/visualize', methods=['GET'])
def visualize():
    try:
        # 读取处理好的数据
        df = pd.read_csv("processed_data.csv")
        df['DATE'] = pd.to_datetime(df['DATE'])

        # 获取筛选条件
        campaign_ids = request.args.getlist('campaign_ids')
        statuses = request.args.getlist('statuses')
        parameters = request.args.getlist('parameters')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not campaign_ids or not parameters or not start_date or not end_date:
            return "Error: Missing required filters", 400

        # 过滤数据
        filtered_df = df[
            (df['CAMPAIGN ID'].astype(str).isin(campaign_ids)) &
            (df['CAMPAIGN STATUS'].isin(statuses)) &
            (df['DATE'] >= start_date) &
            (df['DATE'] <= end_date)
        ]
        if filtered_df.empty:
            return "No data found for the selected filters", 404

        # 创建图表
        figures = []
        for param in parameters:
            if param not in df.columns:
                continue
            fig = px.line(filtered_df, x="DATE", y=param, color="CAMPAIGN ID", title=f"{param} over time")
            figures.append(fig.to_html(full_html=False))

        # 渲染图表
        return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Visualization Results</title>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            </head>
            <body>
                <div class="container mt-5">
                    <h1 class="text-center">Visualization Results</h1>
                    <div>
                        {% for figure in figures %}
                            <div>{{ figure|safe }}</div>
                            <hr>
                        {% endfor %}
                    </div>
                    <a href="/" class="btn btn-primary btn-block mt-4">Upload New Data</a>
                </div>
            </body>
            </html>
        ''', figures=figures)

    except Exception as e:
        return f"Error generating visualization: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
