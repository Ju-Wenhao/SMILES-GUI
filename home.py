import base64
import streamlit as st

# 主函数
def main():
# 读取图片文件并转换为 Base64 编码
    with open('./logo.jpg', 'rb') as file:
        img_base64 = base64.b64encode(file.read()).decode()

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
    """.format(img_base64), unsafe_allow_html=True)

    st.title('药物分子逆合成预测工具')
    st.write('')
    # st.write('这是一个基于G2G-MAML模型的预测分子逆合成反应的工具。')
    # st.write('请输入药物分子的 SMILES 分子式，查看模型预测的反应结果。')
    # st.write('')
    # st.write('')
    # st.write('1. 在selection页面，选择需要预测的模型,并检查模型效果。')
    # st.write('2. 在prediction页面文本输入框中输入药物分子的 SMILES 式。')
    # st.write('3. 点击“预测反应结果”按钮，系统将展示预测的反应结果。')
    # st.write('4. 在visualization页面，选择“2D”或“3D”以选择分子可视化的维度。')
    # st.write('5. 在搜索框中输入结果编号，查看该编号对应的反应图详情。')
    # st.write('')
    # st.write('')
    # st.write('请注意，这里的反应结果仅供参考，不代表实际化学反应。祝你使用愉快！')
    st.markdown("""

    此工具采用 **G2G-MAML** 模型，旨在预测药物分子的逆合成反应路径。操作指南如下：

    ### 步骤概览
    1. **模型选择**  
    - **页面**: `Selection`  
    - **操作**: 选取预测模型，评估其性能。

    2. **输入分子信息**  
    - **页面**: `Prediction`  
    - **方式**: 输入目标药物的 **SMILES** 编码。

    3. **获取预测结果**  
    - **动作**: 点击“预测反应结果”，系统即刻反馈预测路径。

    4. **分子可视化**  
    - **页面**: `Visualization`  
    - **选项**: 选择“2D”或“3D”，享受不同视角的分子结构展示。

    5. **查询详细反应**  
    - **方法**: 在搜索栏键入结果编号，详览指定逆合成步骤。

    ### 注意
    - **预测性质**: 本工具提供的反应预测为理论参考，建议结合实验验证。

    """)

    # 页脚文本
    footer_text = """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 1px; background: transparent;'>
        <p> Copyright&copy; 2024 Wangz Team,SUMHS, All rights reserved.</p>
    </div>
    """
    # 渲染自定义页脚文本
    st.markdown(footer_text, unsafe_allow_html=True)

if __name__ == "__main__":
    main()




