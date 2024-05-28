import os
from io import BytesIO, StringIO
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 定义模型文件的根目录
ROOT_DIR = Path(__file__).parent.parent / 'experiments' / 'uspto_50k'

# 使用Streamlit的文件选择器，返回模型文件及对应的logs.csv路径
def select_model_and_logs():
    options_with_paths = []
    for category in ['without_rxn_class', 'with_rxn_class']:
        category_path = os.path.join(ROOT_DIR, category)
        for subfolder in os.listdir(category_path):
            subfolder_path = os.path.join(category_path, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.endswith('.pt'):
                        # 构建选项显示的字符串格式为子文件夹名+反斜杠+模型文件名
                        option_display = f"{os.path.basename(subfolder)}\\{file}"
                        options_with_paths.append((os.path.join(subfolder_path, file), os.path.join(subfolder_path, 'logs.csv'), option_display))
    
    selected_option_display = st.selectbox("选择一个模型文件", [option[2] for option in options_with_paths])
    
    # 根据显示的选项找到对应的文件路径
    selected_model_file, selected_logs_file = next(((option[0], option[1]) for option in options_with_paths if option[2] == selected_option_display), (None, None))
    
    return selected_model_file, selected_logs_file

def main():

    # 添加学校logo图
    st.markdown("""
    <style>
    .logo {{
        position: absolute;
        top: 0px;  /* 调整距离顶部的位置 */
        left: 600px; /* 调整距离左侧的位置 */
        width: 100px; /* 调整校徽的宽度 */
        height: 100px; /* 调整校徽的高度 */
    }}
    </style>
    <img class="logo" src="data:image/png;base64,{}" alt="校徽">
    """.format(st.session_state.img_base64), unsafe_allow_html=True)

    st.title("请选择模型文件")
    selected_model_file, selected_logs_file = select_model_and_logs()

    if selected_model_file and selected_logs_file:
        st.session_state.selected_model_file = selected_model_file
        st.write(f"请确认你选择模型文件的完整路径: {selected_model_file}")

        st.subheader("模型训练效果展示")

        # 读取文件，跳过前19行，第20行作为列名
        train_df = pd.read_csv(selected_logs_file, skiprows=19)

        # 设置正确的列名
        column_names = ['epoch', 'train_acc', 'valid_acc', 'valid_first_step_acc', 'train_loss']
        train_df.columns = column_names

        # 将第一列设置为行名
        train_df.index = train_df.iloc[:,0]
        train_df.drop(train_df.columns[0], axis=1, inplace=True)

        # 打印数据以进行检查
        st.write(train_df)

        # 获取除了'epoch'列之外的所有列名，作为可选择的字段
        available_columns = train_df.columns.tolist()

        # 创建一个选择框，让用户选择需要显示的字段
        selected_columns = st.multiselect('请选择需要显示的可视化图', available_columns)

        # 绘制折线图
        chart_data = train_df[selected_columns]
        st.line_chart(chart_data,)


        # 读取文件的前19行，作为模型超参数的配置表
        config_df = pd.read_csv(selected_logs_file, nrows=18, header=None)

        # 将第一行设置为列名
        config_df.columns = config_df.iloc[0]
        config_df = config_df.drop(0).reset_index(drop=True)

        # 将第一列设置为行名
        config_df.index = config_df.iloc[:, 0]
        config_df = config_df.drop(config_df.columns[0], axis=1)
        # 显示模型超参数配置表
        st.subheader("模型超参数配置表")
        st.write(config_df)

    # 页脚文本
    footer_text = """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 1px; background: transparent;'>
        <p> Copyright&copy; 2024 Wangz Team,SUMHS, All rights reserved.</p>
    </div>
    """
    # 渲染自定义页脚文本
    st.markdown(footer_text, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
