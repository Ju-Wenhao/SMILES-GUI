import base64
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image, ImageDraw, ImageFont
 
def draw_reaction(reactants, dimension, image_width, image_height):

    mol = Chem.MolFromSmiles(reactants)
    if dimension == '3D':
        # 加氢
        mol = AllChem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        # 优化
        AllChem.MMFFOptimizeMolecule(mol)
 
    return Draw.MolToImage(mol, size=(image_width, image_height))

def add_title_to_image(image, title_text):

    # 获取图像的宽度和高度
    img_width, img_height = image.size
    
    # 设置字体大小和类型，确保你的环境中存在该字体文件
    font_size = 60
    font_path = "arial.ttf"  # 确保这个路径指向一个有效的字体文件
    
    try:
        font = ImageFont.truetype(font_path, font_size)
        # 确保使用正确的属性来获取文本尺寸
        title_width, title_height = font.getbbox(title_text)[2:]  # 使用getbbox方法获取宽度和高度
    except IOError:
        print(f"无法加载字体文件: {font_path}")
        return image  # 如果字体加载失败，则直接返回原图
    
    title_x = (img_width - title_width) // 2
    title_y = 10  # 标题距离图像顶部的距离，可以根据需要调整
    
    # 创建一个新的画布，用于绘制标题
    title_image = Image.new('RGBA', (img_width, img_height), (255, 255, 255, 0))  # 透明背景
    draw = ImageDraw.Draw(title_image)
    
    # 绘制标题
    draw.text((title_x, title_y), title_text, fill=(0, 0, 0), font=font)
    
    # 将带有标题的新图层与原始图像合并
    final_image = Image.alpha_composite(image.convert("RGBA"), title_image)
    
    return final_image

def add_sign_to_image(image, title_text):

    # 获取图像的宽度和高度
    img_width, img_height = image.size
    
    # 设置字体大小和类型，确保你的环境中存在该字体文件
    font_size = 100
    font_path = "arial.ttf"  # 确保这个路径指向一个有效的字体文件
    
    try:
        font = ImageFont.truetype(font_path, font_size)
        # 确保使用正确的属性来获取文本尺寸
        sign_width, sign_height = font.getbbox(title_text)[2:]  # 使用getbbox方法获取宽度和高度
    except IOError:
        print(f"无法加载字体文件: {font_path}")
        return image  # 如果字体加载失败，则直接返回原图
    
    if title_text == "+":
        title_x = (img_width - sign_width) // 2
        title_y = 0
    else:
        title_x = 400
        title_y = 900
    
    # 创建一个新的画布，用于绘制标题
    title_image = Image.new('RGBA', (img_width, img_height), (255, 255, 255, 0))  # 透明背景
    draw = ImageDraw.Draw(title_image)
    
    # 绘制标题
    draw.text((title_x, title_y), title_text, fill=(0, 0, 0), font=font)
    
    # 将带有标题的新图层与原始图像合并
    final_image = Image.alpha_composite(image.convert("RGBA"), title_image)
    
    return final_image

def merge_images_complex(selected_prediction_field_1, selected_prediction_field_2, smiles_input, dimension):

    if selected_prediction_field_2:
        
        # 打开图片
        img1 = draw_reaction(selected_prediction_field_1,dimension,1000,900)
        img2 = draw_reaction(selected_prediction_field_2,dimension,1000,900)
        img3 = draw_reaction(smiles_input,dimension,2000,1100)

        #添加标题
        img1_withtitle = add_title_to_image(img1, 'reactant1')
        img2_withtitle = add_title_to_image(img2, 'reactant2')
        img3_withtitle = add_title_to_image(img3, 'biological product')

        # 水平拼接
        merged_top = Image.new('RGBA', (img1_withtitle.width + img2_withtitle.width, img1_withtitle.height))
        merged_top.paste(img1_withtitle, (0, 0))
        merged_top.paste(img2_withtitle, (img1_withtitle.width, 0))
        merged_top = add_sign_to_image(merged_top,"+")
        
        # 上下拼接
        final_image = Image.new('RGBA', (merged_top.width, merged_top.height + img3_withtitle.height))
        final_image.paste(merged_top, (0, 0))
        final_image.paste(img3_withtitle, (0, merged_top.height))

        final_image = add_sign_to_image(final_image,"=>")
        
    else:

        # 打开图片
        img1 = draw_reaction(selected_prediction_field_1,dimension,2000,900)
        img2 = draw_reaction(smiles_input,dimension,2000,1100)

        #添加标题
        img1_withtitle = add_title_to_image(img1, 'reactant')
        img2_withtitle = add_title_to_image(img2, 'biological product')
        
        # 上下拼接
        final_image = Image.new('RGBA', (img2_withtitle.width, img1_withtitle.height + img2_withtitle.height))
        final_image.paste(img1_withtitle, (0, 0))
        final_image.paste(img2_withtitle, (0, img1_withtitle.height))
        
        final_image = add_sign_to_image(final_image,"=>")


    # 保存图片
    return final_image

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

    st.title('预测反应可视化')

    # 允许用户选择2D或3D显示
    dimension = st.selectbox("选择分子可视化的维度:", options=["2D", "3D"])

    if 'results' in st.session_state:
        results = st.session_state.results
        smiles_input = st.session_state.smiles_input

        # 创建搜索框
        search_id = st.selectbox('选择逆合成预测结果编号:', st.session_state.df)

        # 处理搜索输入
        selected_result = results[search_id - 1]
        selected_prediction_field_1 = selected_result['反应物1']
        selected_prediction_field_2 = selected_result['反应物2']

        st.subheader("模型预测结果分子图")
        st.image(merge_images_complex(selected_prediction_field_1, selected_prediction_field_2,smiles_input,dimension)) 


    else:
        st.error('请先完成模型预测！')

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