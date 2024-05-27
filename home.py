import base64
import streamlit as st

# 主函数
def main():
# 读取图片文件并转换为 Base64 编码
    with open('./logo.jpg', 'rb') as file:
        img_base64 = base64.b64encode(file.read()).decode()
        st.session_state.img_base64 = img_base64

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
    st.write('这是一个基于G2G-MAML模型的预测分子逆合成反应的工具。')
    st.write('请输入药物分子的 SMILES 分子式，查看模型预测的反应结果。')
    st.write('')
    st.caption('1. 在prediction页面文本输入框中输入药物分子的 SMILES 式。')
    st.caption('2. 点击“预测反应结果”按钮，系统将展示预测的反应结果。')
    st.caption('3. 在visualization页面，选择“2D”或“3D”以选择分子可视化的维度。')
    st.caption('4. 在搜索框中输入结果编号，查看该编号对应的反应图详情。')
    st.write('')
    st.write('请注意，这里的反应结果仅供参考，不代表实际化学反应。祝你使用愉快！')

        # 制作背景图片
    # st.markdown("""
    #     <style>
    #     .stApp {
    #         background-image: url('https://imgur.la/images/2024/04/17/1d008d91259eae855.md.png');
    #         background-size: cover;
    #         background-repeat: no-repeat;
    #         background-position: center center;
    #     }
    #     </style>
    #     """, unsafe_allow_html=True)

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



