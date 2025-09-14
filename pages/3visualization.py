from io import BytesIO, StringIO
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import streamlit.components.v1 as components
from PIL import Image
 
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



from utils.ui import render_header, get_palette

# Lightweight toast helper (works even if st.toast not present in older versions)
def safe_toast(msg: str, icon: str = '✅'):
    try:  # Streamlit >= 1.28
        import streamlit as st_mod  # local alias
        if hasattr(st_mod, 'toast'):
            st_mod.toast(f"{icon} {msg}")
            return
    except Exception:
        pass
    # Fallback: transient info container
    st.info(f"{icon} {msg}")


def main():
    render_header(title='Visualization')

    palette = get_palette()
    # Light theme only (dark mode removed). All styling constants fixed to light palette usage.
    tab_bg = palette['secondary']
    tab_active = palette['primary']
    card_bg = '#ffffff'
    card_border = palette.get('border', '#d0d7de')
    code_border = card_border + '55'
    st.markdown(
        f"""
        <style>
        .viz-header {{margin-top: -10px;}}
        .stTabs [data-baseweb='tab-list'] {{gap: 6px;}}
        .stTabs [data-baseweb='tab'] {{background:{tab_bg}; border-radius:6px; padding:6px 14px; box-shadow:0 1px 2px rgba(0,0,0,0.18); color:{palette['text']};}}
        .stTabs [aria-selected='true'] {{background:{tab_active} !important; color: #fff !important; box-shadow:0 2px 6px rgba(0,0,0,0.25);}}
        .mol-card {{background:{card_bg}; padding:10px 14px 18px; border:1px solid {card_border}; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.30);}}
        .action-bar {{display:flex; gap:.5rem; align-items:center; margin-top:.3rem; flex-wrap:wrap;}}
    /* Removed dark mode & theme badge related styles */
        .stMarkdown pre code {{ border:1px solid {code_border}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Removed dimension selector: static composite is always 2D now.

    # Determine data source
    single_available = 'results' in st.session_state and st.session_state.get('product_smiles')
    batch_available = 'batch_results' in st.session_state and isinstance(st.session_state.get('batch_results'), dict) and st.session_state.batch_results

    if not single_available and not batch_available:
        st.error('No prediction data available. Please run Single or Batch prediction first.')
        return

    # --- Compact toolbar for source selection & identifiers ---
    toolbar_css = """
    <style>
    .viz-toolbar {display:flex; gap:.75rem; align-items:flex-end; flex-wrap:wrap; margin-bottom:.25rem;}
    .viz-toolbar .block {display:flex; flex-direction:column;}
    .viz-badge {background:#eef2f6; color:#334155; font-size:11px; padding:2px 6px; border-radius:12px; font-weight:600; letter-spacing:.5px;}
    .mode-inline label {font-size:0.82rem !important; padding:2px 10px !important; border-radius:14px !important; border:1px solid #d0d7de !important;}
    .mode-inline [aria-checked='true']{background:#2563eb !important; color:#fff !important; border-color:#2563eb !important;}
    .stSelectbox label, .stRadio label {font-weight:500; margin-bottom:2px;}
    </style>
    """
    st.markdown(toolbar_css, unsafe_allow_html=True)

    # Decide mode, then show appropriate selectors in one line
    if single_available and batch_available:
        col_tb = st.container()
        with col_tb:
            c1, c2, c3, c4 = st.columns([1.1,1.1,1.3,3])
            with c1:
                source_mode = st.radio('Mode', ['Single','Batch'], horizontal=True, key='viz_mode', label_visibility='visible')
            if source_mode == 'Single':
                results = st.session_state.results
                smiles_input = st.session_state.get('product_smiles') or st.session_state.get('smiles_input', '')
                if not smiles_input:
                    st.error('Product SMILES missing. Please run a prediction first on the Prediction page.')
                    return
                id_options = list(range(1, len(results)+1))
                with c2:
                    search_id = st.selectbox('Result ID', id_options, index=0, key='viz_result_id')
                if search_id is None:
                    st.error('No result selected.')
                    return
                sel_index = int(search_id) - 1
                selected_result = results[sel_index]
                selected_prediction_field_1 = selected_result['Reactant 1']
                selected_prediction_field_2 = selected_result['Reactant 2']
                display_title = f"Single • ID {search_id}"
                base_id = f"single_{search_id}"
            else:
                batch_struct = st.session_state.batch_results
                input_indices = sorted(batch_struct.keys())
                with c2:
                    chosen_input = st.selectbox('Input', input_indices, index=0, key='viz_batch_input')
                rec = batch_struct[chosen_input]
                smiles_input = rec['product']
                ranks = [p['rank'] for p in rec['predictions']]
                with c3:
                    chosen_rank = st.selectbox('Rank', ranks, index=0, key='viz_batch_rank')
                found = next((p for p in rec['predictions'] if p['rank']==chosen_rank), None)
                if not found:
                    st.error('Selected rank not found.')
                    return
                selected_prediction_field_1 = found['Reactant 1']
                selected_prediction_field_2 = found['Reactant 2']
                display_title = f"Batch • Input {chosen_input} • Rank {chosen_rank}"
                base_id = f"batch_{chosen_input}_{chosen_rank}"
            with c4:
                st.markdown(f"<span class='viz-badge'>{display_title}</span>", unsafe_allow_html=True)
    elif batch_available:
        source_mode = 'Batch'
        batch_struct = st.session_state.batch_results
        input_indices = sorted(batch_struct.keys())
        c1, c2, c3 = st.columns([1,1,3])
        with c1:
            chosen_input = st.selectbox('Input', input_indices, index=0, key='viz_batch_input_only')
        rec = batch_struct[chosen_input]
        smiles_input = rec['product']
        ranks = [p['rank'] for p in rec['predictions']]
        with c2:
            chosen_rank = st.selectbox('Rank', ranks, index=0, key='viz_batch_rank_only')
        found = next((p for p in rec['predictions'] if p['rank']==chosen_rank), None)
        if not found:
            st.error('Selected rank not found.')
            return
        selected_prediction_field_1 = found['Reactant 1']
        selected_prediction_field_2 = found['Reactant 2']
        display_title = f"Batch • Input {chosen_input} • Rank {chosen_rank}"
        base_id = f"batch_{chosen_input}_{chosen_rank}"
        c3.markdown(f"<span class='viz-badge'>{display_title}</span>", unsafe_allow_html=True)
    else:
        source_mode = 'Single'
        results = st.session_state.results
        smiles_input = st.session_state.get('product_smiles') or st.session_state.get('smiles_input', '')
        if not smiles_input:
            st.error('Product SMILES missing. Please run a prediction first on the Prediction page.')
            return
        id_options = list(range(1, len(results)+1))
        c1, c2 = st.columns([1,4])
        with c1:
            search_id = st.selectbox('Result ID', id_options, index=0, key='viz_result_id_only')
        if search_id is None:
            st.error('No result selected.')
            return
        sel_index = int(search_id) - 1
        selected_result = results[sel_index]
        selected_prediction_field_1 = selected_result['Reactant 1']
        selected_prediction_field_2 = selected_result['Reactant 2']
        display_title = f"Single • ID {search_id}"
        base_id = f"single_{search_id}"
        c2.markdown(f"<span class='viz-badge'>{display_title}</span>", unsafe_allow_html=True)

    # old caption replaced by badge above
    st.markdown("### Molecular Structures")
    # Top-level controls: view mode + unified layout selector (applies to 2D & 3D)
    ctrl_col1, ctrl_col2 = st.columns([1,1])
    with ctrl_col1:
        view_mode = st.radio('View Mode', ['2D', '3D', 'Both'], horizontal=True, key='viz_view_mode')
    with ctrl_col2:
        layout_mode = st.radio('Layout', ['Auto','Compact','Vertical'], horizontal=True, key='viewer_layout', help='Applies to both 2D grid arrangement and 3D viewer layout')
    with st.expander("Details", expanded=False):
        st.write("Reactant 1:", selected_prediction_field_1)
        if selected_prediction_field_2:
            st.write("Reactant 2:", selected_prediction_field_2)
        st.write("Product:", smiles_input)
        if selected_prediction_field_2:
            st.write(f"Equation: {selected_prediction_field_1} + {selected_prediction_field_2} => {smiles_input}")
        else:
            st.write(f"Equation: {selected_prediction_field_1} => {smiles_input}")

    # ---- 2D GRID RENDERING (replaces composite) ----
    def _draw_single_2d(smi: str, title: str):
        try:
            mol = Chem.MolFromSmiles(smi)  # type: ignore[attr-defined]
            if mol is None:
                st.error(f"Invalid SMILES: {smi}")
                return
            raw = Draw.MolToImage(mol, size=(400, 320))
            img = _ensure_pil(raw)
            st.markdown(f"**{title}**")
            st.image(img, use_container_width=True)
        except Exception as e:  # pragma: no cover
            st.error(f"2D render failed: {e}")

    if view_mode in ('2D','Both'):
        has_second = bool(selected_prediction_field_2)
        # Determine 2D layout using same layout_mode semantics
        if layout_mode == 'Vertical':
            # Stack reactants then product
            if has_second:
                _draw_single_2d(selected_prediction_field_1, 'Reactant 1')
                _draw_single_2d(selected_prediction_field_2, 'Reactant 2')
            else:
                _draw_single_2d(selected_prediction_field_1, 'Reactant')
            _draw_single_2d(smiles_input, 'Product')
        elif layout_mode == 'Compact':
            # Single row (3 or 2 columns)
            cols = st.columns(3 if has_second else 2)
            with cols[0]:
                _draw_single_2d(selected_prediction_field_1, 'Reactant 1' if has_second else 'Reactant')
            if has_second:
                with cols[1]:
                    _draw_single_2d(selected_prediction_field_2, 'Reactant 2')
                with cols[2]:
                    _draw_single_2d(smiles_input, 'Product')
            else:
                with cols[1]:
                    _draw_single_2d(smiles_input, 'Product')
        else:  # Auto
            if has_second:
                top = st.columns(2)
                with top[0]:
                    _draw_single_2d(selected_prediction_field_1, 'Reactant 1')
                with top[1]:
                    _draw_single_2d(selected_prediction_field_2, 'Reactant 2')
                prod_row = st.columns(1)
                with prod_row[0]:
                    _draw_single_2d(smiles_input, 'Product')
            else:
                pair = st.columns(2)
                with pair[0]:
                    _draw_single_2d(selected_prediction_field_1, 'Reactant')
                with pair[1]:
                    _draw_single_2d(smiles_input, 'Product')

    # ---- 3D INTERACTIVE SECTION (re-used; only show if chosen) ----
    if view_mode in ('3D','Both'):
        st.markdown("---")
        st.markdown("**3D Structures**")

        # Build cache key independent of single/batch labels
        cache_key = f"3d_confs_{base_id}_{smiles_input}_{selected_prediction_field_1}_{selected_prediction_field_2}".replace('.', '_')
        regen = st.button("🔄 Regenerate Conformers", key="regen_confs")
        if regen:
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            safe_toast('Regenerating 3D conformers...', icon='⏳')
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
                if conf_data:
                    safe_toast('3D conformers ready', icon='✅')
                else:
                    safe_toast('3D generation failed', icon='⚠️')
        conf_data = st.session_state.get(cache_key)
        if conf_data:
            has_second = 'reactant2' in conf_data
            row3 = None  # predefine
            if layout_mode == 'Auto':
                if has_second:
                    row1 = st.columns(2)
                    row2 = st.columns(1)
                    row3 = None
                else:
                    row1 = st.columns(2)
                    row2 = None
                    row3 = None
            elif layout_mode == 'Compact':
                row1 = st.columns(3 if has_second else 2)
                row2 = None
            else:  # Vertical
                row1 = [st.container()]
                row2 = [st.container()] if has_second else None
                row3 = [st.container()]

            viewer_bg = 'white'
            surface_border = '#e5e7eb'

            def _embed(block: str, title: str, target_col, height=340):
                with target_col:
                    st.markdown(f"**{title}**")
                    js_id = f"viewer_{hash(block) & 0xffffffff}"
                    safe_block = block.replace('`', '\\`').replace('\\', '\\\\')
                    template = (
                        f"""<div id=\"__ID__\" style=\"width:100%;height:{height}px;position:relative;border:1px solid {surface_border};border-radius:8px;\"></div>\n"""
                        """<script src=\"https://3Dmol.org/build/3Dmol.js\"></script>\n"""
                        f"""<script>(function(){{var e=document.getElementById('__ID__');if(!e)return;var v=$3Dmol.createViewer(e,{{backgroundColor:'{viewer_bg}'}});var mb=`__MOLBLOCK__`;v.addModel(mb,'mol');v.setStyle({{ }}, {{stick:{{}}}});v.zoomTo();v.render();}})();</script>\n"""
                    )
                    html = template.replace('__ID__', js_id).replace('__MOLBLOCK__', safe_block)
                    components.html(html, height=height+20)

            if layout_mode == 'Vertical':
                _embed(conf_data['reactant1'], 'Reactant 1', row1[0])
                if has_second and row2:
                    _embed(conf_data['reactant2'], 'Reactant 2', row2[0])
                if row3:
                    _embed(conf_data['product'], 'Product', row3[0])
            elif layout_mode == 'Compact':
                _embed(conf_data['reactant1'], 'Reactant 1', row1[0])
                if has_second:
                    _embed(conf_data['reactant2'], 'Reactant 2', row1[1])
                    _embed(conf_data['product'], 'Product', row1[2])
                else:
                    _embed(conf_data['product'], 'Product', row1[1])
            else:  # Auto
                if has_second and row2:
                    _embed(conf_data['reactant1'], 'Reactant 1', row1[0])
                    _embed(conf_data['reactant2'], 'Reactant 2', row1[1])
                    _embed(conf_data['product'], 'Product', row2[0])
                else:
                    _embed(conf_data['reactant1'], 'Reactant 1', row1[0])
                    _embed(conf_data['product'], 'Product', row1[1])
            # SDF download
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
            st.caption("Automatically generates lowest-energy conformer (ETKDGv3 + MMFF); first run may be slower and results are cached.")
            layout_mode = st.radio('3D Layout', ['Auto','Compact','Vertical'], horizontal=True, key='viewer_layout')

            # Build cache key independent of single/batch labels
            cache_key = f"3d_confs_{base_id}_{smiles_input}_{selected_prediction_field_1}_{selected_prediction_field_2}".replace('.', '_')
            regen = st.button("🔄 Regenerate Conformers", key="regen_confs")
            if regen:
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
                safe_toast('Regenerating 3D conformers...', icon='⏳')
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
                    if conf_data:
                        safe_toast('3D conformers ready', icon='✅')
                    else:
                        safe_toast('3D generation failed', icon='⚠️')
            conf_data = st.session_state.get(cache_key)
            if conf_data:
                has_second = 'reactant2' in conf_data
                # Determine layout
                row3 = None  # predefine
                if layout_mode == 'Auto':
                    if has_second:
                        row1 = st.columns(2)
                        row2 = st.columns(1)
                        row3 = None
                    else:
                        row1 = st.columns(2)
                        row2 = None
                        row3 = None
                elif layout_mode == 'Compact':
                    # Force single row (fallback: original behavior)
                    row1 = st.columns(3 if has_second else 2)
                    row2 = None
                else:  # Vertical
                    row1 = [st.container()]
                    row2 = [st.container()] if has_second else None
                    row3 = [st.container()]

                viewer_bg = 'white'
                surface_border = '#e5e7eb'

                def _embed(block: str, title: str, target_col, height=340):
                    with target_col:
                        st.markdown(f"**{title}**")
                        js_id = f"viewer_{hash(block) & 0xffffffff}"
                        safe_block = block.replace('`', '\\`').replace('\\', '\\\\')
                        template = (
                            f"""<div id=\"__ID__\" style=\"width:100%;height:{height}px;position:relative;border:1px solid {surface_border};border-radius:8px;\"></div>\n"""
                            """<script src=\"https://3Dmol.org/build/3Dmol.js\"></script>\n"""
                            f"""<script>(function(){{var e=document.getElementById('__ID__');if(!e)return;var v=$3Dmol.createViewer(e,{{backgroundColor:'{viewer_bg}'}});var mb=`__MOLBLOCK__`;v.addModel(mb,'mol');v.setStyle({{ }}, {{stick:{{}}}});v.zoomTo();v.render();}})();</script>\n"""
                        )
                        html = template.replace('__ID__', js_id).replace('__MOLBLOCK__', safe_block)
                        components.html(html, height=height+20)

                if layout_mode == 'Vertical':
                    _embed(conf_data['reactant1'], 'Reactant 1', row1[0])
                    if has_second and row2:
                        _embed(conf_data['reactant2'], 'Reactant 2', row2[0])
                    if row3:
                        _embed(conf_data['product'], 'Product', row3[0])
                elif layout_mode == 'Compact':
                    _embed(conf_data['reactant1'], 'Reactant 1', row1[0])
                    if has_second:
                        _embed(conf_data['reactant2'], 'Reactant 2', row1[1])
                        _embed(conf_data['product'], 'Product', row1[2])
                    else:
                        _embed(conf_data['product'], 'Product', row1[1])
                else:  # Auto
                    if has_second and row2:
                        _embed(conf_data['reactant1'], 'Reactant 1', row1[0])
                        _embed(conf_data['reactant2'], 'Reactant 2', row1[1])
                        _embed(conf_data['product'], 'Product', row2[0])
                    else:
                        _embed(conf_data['reactant1'], 'Reactant 1', row1[0])
                        _embed(conf_data['product'], 'Product', row1[1])
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
    # (All data source absence handled earlier)

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