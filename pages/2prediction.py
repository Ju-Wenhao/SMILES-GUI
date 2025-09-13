import base64
import os
import torch
import pandas as pd
import streamlit as st
from pathlib import Path
from rdkit import RDLogger
from rdkit import Chem
from models import Graph2Edits, BeamSearch


lg = RDLogger.logger()
lg.setLevel(4)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Root experiments directory
ROOT_DIR = Path(__file__).parent.parent / 'experiments' / 'uspto_50k'

example_smiles = "[O:1]=[S:19]([c:18]1[cH:17][c:15]([Cl:16])[c:14]2[o:13][c:12]3[c:28]([c:27]2[cH:26]1)[CH2:29][N:9]([C:7]([O:6][C:3]([CH3:2])([CH3:4])[CH3:5])=[O:8])[CH2:10][CH2:11]3)[c:20]1[cH:21][cH:22][cH:23][cH:24][cH:25]1"

def predict_reactions(selected_file, molecule_string, use_rxn_class=False, beam_size=10, max_steps=9):
    # Basic SMILES validation
    if not molecule_string or not isinstance(molecule_string, str):
        raise ValueError('Input SMILES is empty or not a string.')
    mol = Chem.MolFromSmiles(molecule_string)
    if mol is None:
        raise ValueError('Failed to parse input SMILES (MolFromSmiles returned None).')
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        raise ValueError(f'Invalid SMILES structure: {e}')

    checkpoint_path = os.path.join(ROOT_DIR, selected_file)

    try:
        load_mode = 'weights_only'
        if DEVICE == 'cuda':
            checkpoint = torch.load(checkpoint_path, weights_only=True)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
    except Exception:
        load_mode = 'full_pickle'
        if DEVICE == 'cuda':
            checkpoint = torch.load(checkpoint_path, weights_only=False)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)

    # Prefer new format (format_version == 2)
    from utils.rxn_graphs import Vocab
    model = None
    version = None
    if isinstance(checkpoint, dict) and checkpoint.get('format_version', None) == 2:
        version = 2
        ma = checkpoint.get('model_args', {})
        cfg = ma.get('config')
        atom_vocab_list = ma.get('atom_vocab_list')
        bond_vocab_list = ma.get('bond_vocab_list')
        if not (cfg and atom_vocab_list and bond_vocab_list):
            raise ValueError('format_version=2 checkpoint missing required fields.')
        atom_vocab = Vocab(atom_vocab_list)
        bond_vocab = Vocab(bond_vocab_list)
        model = Graph2Edits(config=cfg, atom_vocab=atom_vocab, bond_vocab=bond_vocab, device=DEVICE)
        state_dict = checkpoint['state']
        model.load_state_dict(state_dict)
    else:
        version = checkpoint.get('format_version', 1) if isinstance(checkpoint, dict) else 'legacy'
        # Backward compatibility for legacy / v1 format
        if 'saveables' not in checkpoint or 'state' not in checkpoint:
            raise ValueError('Checkpoint missing required keys: saveables/state')
        saveables = checkpoint['saveables']
        if isinstance(saveables, dict) and 'config' in saveables and 'atom_vocab' in saveables and 'bond_vocab' in saveables:
            cfg = saveables['config']
            atom_vocab = saveables['atom_vocab']
            bond_vocab = saveables['bond_vocab']
            model = Graph2Edits(config=cfg, atom_vocab=atom_vocab, bond_vocab=bond_vocab, device=DEVICE)
        else:
            model = Graph2Edits(**saveables, device=DEVICE)
        model.load_state_dict(checkpoint['state'])

    model.to(DEVICE)
    model.eval()
    _set_cached_model(checkpoint_path, model, version, load_mode)

    beam_model = BeamSearch(model=model, step_beam_size=10, beam_size=beam_size, use_rxn_class=use_rxn_class)

    with torch.no_grad():
        top_k_results = beam_model.run_search(prod_smi=molecule_string, max_steps=max_steps)

    def format_prediction(beam_idx, path):
        pred_smi = path['final_smi']
        prob = path['prob']
        str_edits = '|'.join(f'({str(edit)};{p})' for edit, p in zip(path['rxn_actions'], path['edits_prob']))
        pred_smi_parts = pred_smi.split('.')
        new_pred_smi_field_1 = pred_smi_parts[0]
        new_pred_smi_field_2 = pred_smi_parts[1] if len(pred_smi_parts) > 1 else None
        return {
            'Reactant 1': new_pred_smi_field_1,
            'Reactant 2': new_pred_smi_field_2,
            'Confidence': f'{float(prob * 100):.2f}%',
        }

    filtered_predictions = [format_prediction(beam_idx, path) for beam_idx, path in enumerate(top_k_results) if path.get('final_smi') != 'final_smi_unmapped']

    unique_predictions = []
    seen_predictions = set()
    for prediction in filtered_predictions:
        if prediction['Reactant 1'] not in seen_predictions:
            unique_predictions.append(prediction)
            seen_predictions.add(prediction['Reactant 1'])

    return unique_predictions, load_mode

def _get_cached_model(checkpoint_path):
    cache = st.session_state.get('_model_cache')
    if cache and cache.get('path') == checkpoint_path:
        return cache.get('model'), cache.get('version'), cache.get('load_mode')
    return None, None, None

def _set_cached_model(checkpoint_path, model, version, load_mode):
    st.session_state['_model_cache'] = {
        'path': checkpoint_path,
        'model': model,
        'version': version,
        'load_mode': load_mode,
    }

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

    # Initialize key (avoid first access KeyError)
    if 'smiles_input' not in st.session_state:
        st.session_state.smiles_input = ''

    # Example buttons use callbacks to avoid key conflicts with text_area
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
                results, load_mode = predict_reactions(model_path, current_smi)
                st.session_state.results = results
                # Persist original product SMILES for visualization page
                st.session_state.product_smiles = current_smi
                df = pd.DataFrame(results)
                df.insert(0, 'ID', range(1, len(df) + 1))
                df = df.set_index('ID')
                df.columns = df.columns.map(str)
                st.write(df)
                df.insert(0, 'ID', range(1, len(df) + 1))
                st.session_state.df = df
                if load_mode == 'full_pickle':
                    st.info('Model loaded with full_pickle mode (fallback from weights_only). Ensure the checkpoint is from a trusted source.')
            except ValueError as ve:
                st.error(f'SMILES validation failed: {ve}')
            except FileNotFoundError as fe:
                st.error(f'Model file not found: {fe}')
            except Exception as e:
                import traceback
                traceback.print_exc()
                st.error(f'Unexpected error during prediction: {e}')
    elif st.session_state.get('results'):
        st.subheader("Last Prediction")
        df_prev = pd.DataFrame(st.session_state.results)
        if 'ID' not in df_prev.columns:
            df_prev.insert(0, 'ID', range(1, len(df_prev) + 1))
        df_prev = df_prev.set_index('ID')
        st.write(df_prev)

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