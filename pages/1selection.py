
import os
from io import BytesIO
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

def plot_graphs(df):
    # 创建 2x2 子图
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.tight_layout(pad=5.0)

    # 绘制训练准确率
    axs[0, 0].plot(df['epoch'], df['train_acc'], label='train_acc', color='blue')
    axs[0, 0].set_title('Train Accuracy vs Epoch')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # 绘制训练损失
    axs[0, 1].plot(df['epoch'], df['train_loss'], label='train_loss', color='red')
    axs[0, 1].set_title('Train Loss vs Epoch')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # 绘制验证准确率
    axs[1, 0].plot(df['epoch'], df['valid_acc'], label='valid_acc', color='green')
    axs[1, 0].set_title('Validation Accuracy vs Epoch')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    # 标注最佳验证准确率
    best_valid_acc = df['valid_acc'].max()
    best_epoch = df['epoch'][df['valid_acc'].idxmax()]
    axs[1, 0].scatter(best_epoch, best_valid_acc, color='black')
    axs[1, 0].annotate(f'Epoch {best_epoch}: {best_valid_acc:.4f}',
                       (best_epoch, best_valid_acc),
                       textcoords="offset points",
                       xytext=(0,-30),
                       ha='center',
                       fontsize=14)

    # 绘制验证第一步准确率
    axs[1, 1].plot(df['epoch'], df['valid_first_step_acc'], label='valid_first_step_acc', color='purple')
    axs[1, 1].set_title('Validation First Step Accuracy vs Epoch')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    # 标注最佳验证第一步准确率
    best_valid_first_step_acc = df['valid_first_step_acc'].max()
    best_epoch = df['epoch'][df['valid_first_step_acc'].idxmax()]
    axs[1, 1].scatter(best_epoch, best_valid_first_step_acc, color='black')
    axs[1, 1].annotate(f'Epoch {best_epoch}: {best_valid_first_step_acc:.4f}',
                       (best_epoch, best_valid_first_step_acc),
                       textcoords="offset points",
                       xytext=(0,-30),
                       ha='center',
                       fontsize=14)
    
    # 将图表保存为内存中的PNG图片
    png_buffer = BytesIO()
    fig.savefig(png_buffer, format='png')
    png_buffer.seek(0)

    # 使用PIL读取内存中的PNG图片，并返回
    image = Image.open(png_buffer)
    return image

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

    st.title("请选择模型文件:")
    selected_model_file, selected_logs_file = select_model_and_logs()
    if selected_model_file and selected_logs_file:
        st.write(f"请确认你选择模型文件的完整路径: {selected_model_file}")
        # 将选择的文件路径存储到session_state（如果需要的话）
        st.session_state.selected_model_file = selected_model_file

    # 读取文件，跳过前20行，并将第20行作为列名
    df = pd.read_csv(selected_logs_file, skiprows=20, header=19)

    # 设置正确的列名
    column_names = ['epoch', 'train_acc', 'valid_acc', 'valid_first_step_acc', 'train_loss']
    df.columns = column_names

    st.image(plot_graphs(df))


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