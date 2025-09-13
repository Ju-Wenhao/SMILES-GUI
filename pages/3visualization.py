import base64
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image, ImageDraw, ImageFont
 
def draw_reaction(reactants, dimension, image_width, image_height):

    mol = Chem.MolFromSmiles(reactants)
    if dimension == '3D':
        mol = AllChem.AddHs(mol)  # type: ignore[attr-defined]
        AllChem.EmbedMolecule(mol)  # type: ignore[attr-defined]
        AllChem.MMFFOptimizeMolecule(mol)  # type: ignore[attr-defined]
 
    return Draw.MolToImage(mol, size=(image_width, image_height))

from PIL import ImageFont


def add_title_to_image(image, title_text, font_size=60):

    img_width, img_height = image.size

    font_size = 60
    font_path = "./assets/arial.ttf"

    try:
        font = ImageFont.truetype(font_path, font_size)
        title_width, title_height = font.getbbox(title_text)[2:]
    except IOError:
        print(f"Cannot load font file: {font_path}")
        return image

    title_x = (img_width - title_width) // 2
    title_y = 10

    title_image = Image.new('RGBA', (img_width, img_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(title_image)

    draw.text((title_x, title_y), title_text, fill=(0, 0, 0), font=font)

    final_image = Image.alpha_composite(image.convert("RGBA"), title_image)

    return final_image

def add_sign_to_image(image, title_text, font_size=100):

    img_width, img_height = image.size

    font_size = 100
    font_path = "./assets/arial.ttf"

    try:
        font = ImageFont.truetype(font_path, font_size)
        sign_width, sign_height = font.getbbox(title_text)[2:]
    except IOError:
        print(f"Cannot load font file: {font_path}")
        return image

    if title_text == "+":
        title_x = (img_width - sign_width) // 2
        title_y = 0
    else:
        title_x = 400
        title_y = 900

    title_image = Image.new('RGBA', (img_width, img_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(title_image)

    draw.text((title_x, title_y), title_text, fill=(0, 0, 0), font=font)

    final_image = Image.alpha_composite(image.convert("RGBA"), title_image)

    return final_image

def merge_images_complex(selected_prediction_field_1, selected_prediction_field_2, smiles_input, dimension):
    """Compose a combined image for one or two reactants and the product."""
    if selected_prediction_field_2:
        img1 = draw_reaction(selected_prediction_field_1, dimension, 1000, 900)
        img2 = draw_reaction(selected_prediction_field_2, dimension, 1000, 900)
        img_prod = draw_reaction(smiles_input, dimension, 2000, 1100)

        img1 = add_title_to_image(img1, 'Reactant 1')
        img2 = add_title_to_image(img2, 'Reactant 2')
        img_prod = add_title_to_image(img_prod, 'Product')

        w1, h1 = img1.size
        w2, _ = img2.size
        merged_top = Image.new('RGBA', (w1 + w2, h1))
        merged_top.paste(img1, (0, 0))
        merged_top.paste(img2, (w1, 0))
        merged_top = add_sign_to_image(merged_top, "+")

        mt_w, mt_h = merged_top.size
        _, ph = img_prod.size
        final_img = Image.new('RGBA', (mt_w, mt_h + ph))
        final_img.paste(merged_top, (0, 0))
        final_img.paste(img_prod, (0, mt_h))
        final_img = add_sign_to_image(final_img, "=>")
        return final_img
    else:
        img1 = draw_reaction(selected_prediction_field_1, dimension, 2000, 900)
        img_prod = draw_reaction(smiles_input, dimension, 2000, 1100)

        img1 = add_title_to_image(img1, 'Reactant')
        img_prod = add_title_to_image(img_prod, 'Product')

        w_prod, h_prod = img_prod.size
        _, h1 = img1.size
        final_img = Image.new('RGBA', (w_prod, h1 + h_prod))
        final_img.paste(img1, (0, 0))
        final_img.paste(img_prod, (0, h1))
        final_img = add_sign_to_image(final_img, "=>")
        return final_img

def main():
    with open('./assets/logo.jpg', 'rb') as file:
        img_base64 = base64.b64encode(file.read()).decode()

    # Logo
    st.markdown(
        f"""
        <style>
        .logo {{ position: absolute; top:0; left:600px; width:90px; height:90px; }}
        </style>
        <img class="logo" src="data:image/png;base64,{img_base64}" alt="Logo" />
        """,
        unsafe_allow_html=True,
    )
    st.title('Visualization')

    dimension = st.selectbox("Select display dimension", options=["2D", "3D"])

    if 'results' in st.session_state:
        results = st.session_state.results
        smiles_input = st.session_state.get('smiles_input', '')
        id_options = list(range(1, len(results) + 1))
        search_id = st.selectbox('Select predicted result ID', id_options, index=0)
        if search_id is None:
            st.warning('No result selected.')
            return
        sel_index = int(search_id) - 1
        selected_result = results[sel_index]
        selected_prediction_field_1 = selected_result['Reactant 1']
        selected_prediction_field_2 = selected_result['Reactant 2']
        st.subheader("Molecular Structures")
        img = merge_images_complex(selected_prediction_field_1, selected_prediction_field_2, smiles_input, dimension)
        st.image(img)
    else:
        st.error('Please run a prediction first.')

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