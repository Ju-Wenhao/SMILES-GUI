import base64
from io import BytesIO, StringIO
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
try:
    import py3Dmol  # type: ignore
    _HAS_PY3DMOL = True
except Exception:  # pragma: no cover
    _HAS_PY3DMOL = False
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont
 
def _ensure_pil(img_obj):
    """Normalize various rdkit Draw return types into a PIL Image."""
    if isinstance(img_obj, Image.Image):
        return img_obj
    if isinstance(img_obj, bytes):
        return Image.open(BytesIO(img_obj))
    if isinstance(img_obj, (tuple, list)):
        for part in img_obj:
            if isinstance(part, Image.Image):
                return part
    raise ValueError(f"Unexpected image return type: {type(img_obj)}")


def draw_reaction(smiles: str, image_width: int, image_height: int):
    smiles = (smiles or "").strip()
    if not smiles:
        raise ValueError("Empty SMILES string.")
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    # Always render as 2D for static composite; 3D interactive view handled separately.
    try:
        img_raw = Draw.MolToImage(mol, size=(image_width, image_height))
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Rendering failed: {e}")
    return _ensure_pil(img_raw)

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

def merge_images_complex(selected_prediction_field_1, selected_prediction_field_2, smiles_input):
    """Compose a combined image for one or two reactants and the product.

    Parameters
    ----------
    selected_prediction_field_1 : str
        First reactant SMILES.
    selected_prediction_field_2 : str | None
        Second reactant SMILES (may be empty / None).
    smiles_input : str
        Product SMILES passed explicitly (do not re-read from session_state to avoid KeyError).
    (Static composite now always 2D.)
    """
    smiles_input = (smiles_input or "").strip()
    if not smiles_input:
        raise ValueError("Product SMILES is empty; cannot render product structure.")

    # Two-reactant case
    if selected_prediction_field_2:
        img1 = draw_reaction(selected_prediction_field_1, 1000, 900)
        img2 = draw_reaction(selected_prediction_field_2, 1000, 900)
        img_prod = draw_reaction(smiles_input, 2000, 1100)

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

    # Single-reactant case
    img1 = draw_reaction(selected_prediction_field_1, 2000, 900)
    img_prod = draw_reaction(smiles_input, 2000, 1100)

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

    st.markdown(
        """
        <style>
        .viz-header {margin-top: -10px;}
        .stTabs [data-baseweb=\"tab-list\"] {gap: 6px;}
        .stTabs [data-baseweb=\"tab\"] {background:#f5f7fa; border-radius:6px; padding:6px 14px; box-shadow:0 1px 2px rgba(0,0,0,0.06);} 
        .stTabs [aria-selected='true'] {background:#2563eb !important; color: #fff !important;}
        .mol-card {background:#ffffff; padding:10px 14px 18px; border:1px solid #e5e7eb; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.05);}
        .action-bar {display:flex; gap:.5rem; align-items:center; margin-top:.3rem; flex-wrap:wrap;}
        .small-badge {font-size:0.65rem; padding:2px 6px; background:#eef2ff; color:#3730a3; border-radius:4px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Removed dimension selector: static composite is always 2D now.

    if 'results' in st.session_state:
        results = st.session_state.results
        # Prefer persisted product_smiles from prediction page
        smiles_input = st.session_state.get('product_smiles') or st.session_state.get('smiles_input', '')
        if not smiles_input:
            st.error('Product SMILES missing. Please run a prediction first on the Prediction page.')
            return
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
        # Shared details (visible for both tabs)
        with st.expander("Details", expanded=False):
            st.write("Reactant 1:", selected_prediction_field_1)
            if selected_prediction_field_2:
                st.write("Reactant 2:", selected_prediction_field_2)
            st.write("Product:", smiles_input)
            # Combined summary block for easy copy
            lines = [f"Reactant 1: {selected_prediction_field_1}"]
            if selected_prediction_field_2:
                lines.append(f"Reactant 2: {selected_prediction_field_2}")
            lines.append(f"Product: {smiles_input}")
            if selected_prediction_field_2:
                eq_line = f"Equation: {selected_prediction_field_1} + {selected_prediction_field_2} => {smiles_input}"
            else:
                eq_line = f"Equation: {selected_prediction_field_1} => {smiles_input}"
            lines.append(eq_line)
            combined = "\n".join(lines)
            if st.button("Copy All SMILES", key="copy_all_smiles"):
                # Display in a code block for manual copy (Streamlit cannot access clipboard directly without JS)
                st.code(combined, language='text')
        # Tabs: Static composite vs Interactive 3D
        tabs = st.tabs(["🧪 Static Composite (2D)", "🧬 Interactive 3D"])
        with tabs[0]:
            try:
                img = merge_images_complex(selected_prediction_field_1, selected_prediction_field_2, smiles_input)
            except ValueError as e:
                st.error(str(e))
            else:
                st.image(img, caption="Composite (2D)")
                col_copy, col_dl = st.columns([1,1])
                import io
                buf = io.BytesIO()
                img.convert('RGB').save(buf, format='PNG')
                col_dl.download_button("Download Composite PNG", data=buf.getvalue(), file_name="composite.png", mime="image/png")

        with tabs[1]:
            st.caption("Automatically generates lowest-energy conformer (ETKDGv3 + MMFF); first run may be slower and results are cached.")
            if not _HAS_PY3DMOL:
                st.error('py3Dmol is not installed or failed to import. Please install with: pip install py3Dmol')
            else:
                cache_key = f"3d_confs_{search_id}_{smiles_input}_{selected_prediction_field_1}_{selected_prediction_field_2}".replace('.', '_')
                regen = st.button("🔄 Regenerate Conformers", key="regen_confs")
                if regen and cache_key in st.session_state:
                    del st.session_state[cache_key]
                loading_placeholder = st.empty()
                if cache_key not in st.session_state:
                    with st.spinner('Generating 3D conformers...'):
                        conf_data = {}
                        def _gen_3d(smi: str):
                            mol = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
                            if mol is None:
                                raise ValueError(f"Invalid SMILES for 3D: {smi}")
                            mol = AllChem.AddHs(mol)  # type: ignore[attr-defined]
                            params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
                            cid_list = AllChem.EmbedMultipleConfs(mol, numConfs=10, params=params)  # type: ignore[attr-defined]
                            energies = []
                            for cid in cid_list:
                                try:
                                    AllChem.MMFFOptimizeMolecule(mol, confId=cid)  # type: ignore[attr-defined]
                                    props = AllChem.MMFFGetMoleculeProperties(mol)  # type: ignore[attr-defined]
                                    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)  # type: ignore[attr-defined]
                                    energies.append((cid, ff.CalcEnergy()))
                                except Exception:
                                    continue
                            best_cid = min(energies, key=lambda x: x[1])[0] if energies else int(cid_list[0])
                            mb = Chem.MolToMolBlock(mol, confId=best_cid)  # type: ignore[attr-defined]
                            return mb
                        try:
                            conf_data['reactant1'] = _gen_3d(selected_prediction_field_1)
                            if selected_prediction_field_2:
                                conf_data['reactant2'] = _gen_3d(selected_prediction_field_2)
                            conf_data['product'] = _gen_3d(smiles_input)
                        except Exception as e:  # pragma: no cover
                            st.error(f"3D generation failed: {e}")
                            conf_data = None
                        st.session_state[cache_key] = conf_data
                conf_data = st.session_state.get(cache_key)
                if conf_data:
                    has_second = 'reactant2' in conf_data
                    cols = st.columns(3 if has_second else 2)
                    def _embed_model(block: str, title: str, col):
                        with col:
                            st.markdown(f"**{title}**")
                            js_id = f"viewer_{hash(block) & 0xffffffff}"
                            # Escape backticks and backslashes for JS template usage
                            safe_block = block.replace('`', '\\`').replace('\\', '\\\\')
                            template = (
                                """<div id=\"__ID__\" style=\"width:100%;height:320px;position:relative;border:1px solid #e5e7eb;border-radius:8px;\"></div>\n"""
                                """<script src=\"https://3Dmol.org/build/3Dmol.js\"></script>\n"""
                                """<script>(function(){var e=document.getElementById('__ID__');if(!e)return;var v=$3Dmol.createViewer(e,{backgroundColor:'white'});var mb=`__MOLBLOCK__`;v.addModel(mb,'mol');v.setStyle({}, {stick:{}});v.zoomTo();v.render();})();</script>\n"""
                            )
                            html = template.replace('__ID__', js_id).replace('__MOLBLOCK__', safe_block)
                            components.html(html, height=340)
                    _embed_model(conf_data['reactant1'], 'Reactant 1', cols[0])
                    if has_second:
                        _embed_model(conf_data['reactant2'], 'Reactant 2', cols[1])
                        _embed_model(conf_data['product'], 'Product', cols[2])
                    else:
                        _embed_model(conf_data['product'], 'Product', cols[1])
                    # SDF download (combine available molecules)
                    try:
                        from rdkit import Chem as _Chem  # type: ignore
                        sdf_buffer = StringIO()
                        writer = _Chem.SDWriter(sdf_buffer)  # type: ignore[attr-defined]
                        def _write(name: str, mb: str):
                            m = _Chem.MolFromMolBlock(mb, sanitize=True)  # type: ignore[attr-defined]
                            if m is not None:
                                m.SetProp('_Name', name)
                                writer.write(m)
                        _write('Reactant 1', conf_data['reactant1'])
                        if has_second:
                            _write('Reactant 2', conf_data['reactant2'])
                        _write('Product', conf_data['product'])
                        writer.close()
                        st.download_button(
                            "Download 3D SDF",
                            data=sdf_buffer.getvalue(),
                            file_name="reaction_3d.sdf",
                            mime="chemical/x-mdl-sdfile"
                        )
                    except Exception as e:  # pragma: no cover
                        st.warning(f"Failed to build SDF: {e}")
                else:
                    st.info('No 3D conformer data available.')
    else:
        st.error('Please run a prediction first.')

    # Footer
    footer_text = """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 1px; background: transparent;'>
        <p> Copyright&copy; 2024 Wangz Team,SUMHS, All rights reserved.</p>
    </div>
    """
    # Render custom footer
    st.markdown(footer_text, unsafe_allow_html=True)

if __name__ == '__main__':
    main()