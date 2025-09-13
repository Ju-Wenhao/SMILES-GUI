import base64
import os
import torch
import pandas as pd
import streamlit as st
from pathlib import Path
from rdkit import RDLogger
from models import Graph2Edits, BeamSearch


lg = RDLogger.logger()
lg.setLevel(4)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Root experiments directory
ROOT_DIR = Path(__file__).parent.parent / 'experiments' / 'uspto_50k'

example_smiles = "[O:1]=[S:19]([c:18]1[cH:17][c:15]([Cl:16])[c:14]2[o:13][c:12]3[c:28]([c:27]2[cH:26]1)[CH2:29][N:9]([C:7]([O:6][C:3]([CH3:2])([CH3:4])[CH3:5])=[O:8])[CH2:10][CH2:11]3)[c:20]1[cH:21][cH:22][cH:23][cH:24][cH:25]1"

def predict_reactions(selected_file, molecule_string, use_rxn_class=False, beam_size=10, max_steps=9):
    
    checkpoint_path = os.path.join(ROOT_DIR, selected_file)

    if DEVICE == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load((checkpoint_path), map_location=torch.device('cpu'))

    config = checkpoint['saveables']

    model = Graph2Edits(**config, device=DEVICE)
    model.load_state_dict(checkpoint['state'])
    model.to(DEVICE)
    model.eval()

    beam_model = BeamSearch(model=model, step_beam_size=10, beam_size=beam_size, use_rxn_class=use_rxn_class)

    with torch.no_grad():
        top_k_results = beam_model.run_search(prod_smi=molecule_string, max_steps=max_steps)

    def format_prediction(beam_idx, path):
        pred_smi = path['final_smi']
        prob = path['prob']
        str_edits = '|'.join(f'({str(edit)};{p})' for edit, p in zip(path['rxn_actions'], path['edits_prob']))

        # Split pred_smi by '.' and store the parts in separate fields
        pred_smi_parts = pred_smi.split('.')
        new_pred_smi_field_1 = pred_smi_parts[0]
        new_pred_smi_field_2 = pred_smi_parts[1] if len(pred_smi_parts) > 1 else None
        # reaction_smiles = str(pred_smi) + '>>' + str(molecule_string)
        return {
            'Reactant 1': new_pred_smi_field_1,
            'Reactant 2': new_pred_smi_field_2,
            'Confidence': f'{float(prob * 100):.2f}%',
        }
    
    # Filter out predictions with 'prediction': 'final_smi_unmapped'
    filtered_predictions = [format_prediction(beam_idx, path) for beam_idx, path in enumerate(top_k_results) if path['final_smi'] != 'final_smi_unmapped']

    # Remove duplicate predictions based on the 'prediction' key
    unique_predictions = []
    seen_predictions = set()
    for prediction in filtered_predictions:
        if prediction['Reactant 1'] not in seen_predictions:
            unique_predictions.append(prediction)
            seen_predictions.add(prediction['Reactant 1'])
    print(unique_predictions)

    return unique_predictions

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
    st.title('Prediction')

    # 初始化 key（避免首次访问不存在）
    if 'smiles_input' not in st.session_state:
        st.session_state.smiles_input = ''

    # 示例按钮使用回调，避免与已实例化同 key 的 text_area 冲突
    def _load_example():
        st.session_state.smiles_input = example_smiles

    def _clear_input():
        st.session_state.smiles_input = ''

    st.markdown(
        """
        <style>
        .smi-box-wrapper { position: relative; }
        .smi-controls { display: flex; gap: .5rem; align-items: center; justify-content: flex-end; margin-top: -0.2rem; }
        .smi-length { margin-right: auto; font-size: 0.78rem; color: #666; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.text_area(
        'Enter product SMILES',
        height=5,
        max_chars=500,
        help='Max length 500 characters',
        key='smiles_input'
    )
    col_len, col_spacer, col_btn1, col_btn2, col_btn3 = st.columns([2,0.3,1,1,1])
    with col_len:
        st.caption(f"Current SMILES length: {len(st.session_state.smiles_input)}")
    with col_btn1:
        st.button('Use example', on_click=_load_example, help='Load example')
    with col_btn2:
        st.button('Clear', on_click=_clear_input, help='Clear input')
    with col_btn3:
        predict_clicked = st.button('Predict', type='primary', help='Run retrosynthesis prediction')
    if 'predict_clicked' not in st.session_state:
        st.session_state.predict_clicked = False
    if 'predict_clicked_flag' in locals() and predict_clicked:
        st.session_state.predict_clicked = True
    elif predict_clicked:
        st.session_state.predict_clicked = True

    model_path = st.session_state.get('selected_model_file')
    if not model_path:
        st.info('Select a model on the Selection page to enable prediction.')

    # (example button handled above with rerun)

    current_smi = st.session_state.get('smiles_input','').strip()
    if 'predict_clicked' in st.session_state and st.session_state.predict_clicked:
        if not current_smi:
            st.warning('Input SMILES is empty.')
        elif not model_path:
            st.warning('No model selected.')
        else:
            st.subheader("Predicted Precursors")
            try:
                results = predict_reactions(model_path, current_smi)
                st.session_state.results = results
                df = pd.DataFrame(results)
                df.insert(0, 'ID', range(1, len(df) + 1))
                df = df.set_index('ID')
                st.write(df)
                df.insert(0, 'ID', range(1, len(df) + 1))
                st.session_state.df = df
            except Exception:
                st.error('Invalid SMILES string.')
    elif st.session_state.get('results'):
        st.subheader("Last Prediction")
        df_prev = pd.DataFrame(st.session_state.results)
        if 'ID' not in df_prev.columns:
            df_prev.insert(0, 'ID', range(1, len(df_prev) + 1))
        df_prev = df_prev.set_index('ID')
        st.write(df_prev)

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