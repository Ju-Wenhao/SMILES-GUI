import base64
import os
import torch
import pandas as pd
import streamlit as st
from pathlib import Path
from rdkit import RDLogger
from rdkit import Chem
from models import Graph2Edits, BeamSearch
from utils.ui import render_header

# Toast helper (compat with older Streamlit)
def safe_toast(msg: str, icon: str = '✅'):
    try:
        if hasattr(st, 'toast'):
            st.toast(f"{icon} {msg}")
            return
    except Exception:
        pass
    st.info(f"{icon} {msg}")


lg = RDLogger.logger()
lg.setLevel(4)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Root experiments directory
ROOT_DIR = Path(__file__).parent.parent / 'experiments' / 'uspto_50k'

example_smiles = "[O:1]=[S:19]([c:18]1[cH:17][c:15]([Cl:16])[c:14]2[o:13][c:12]3[c:28]([c:27]2[cH:26]1)[CH2:29][N:9]([C:7]([O:6][C:3]([CH3:2])([CH3:4])[CH3:5])=[O:8])[CH2:10][CH2:11]3)[c:20]1[cH:21][cH:22][cH:23][cH:24][cH:25]1"

# Multi-line batch example (3 SMILES provided by user)
batch_example_smiles = """[O:1]=[C:2]1[CH2:3][c:4]2[cH:5][c:6](-[c:7]3[n:8][n:9][c:10](-[c:11]4[cH:12][cH:13][cH:14][cH:15][cH:16]4)[s:17]3)[cH:18][cH:19][c:20]2[NH:21]1
[CH3:1][O:2][C:3](=[O:4])[O:5][CH2:6][c:7]1[cH:8][cH:9][cH:10][c:11]([NH:12][C:13](=[O:14])[c:15]2[n:16][c:17](-[c:18]3[cH:19][cH:20][cH:21][cH:22][cH:23]3)[nH:24][c:25]2[CH2:26][CH2:27][C:28]23[CH2:29][CH:30]4[CH2:31][CH:32]([CH2:33][CH:34]([CH2:35]4)[CH2:36]2)[CH2:37]3)[cH:38]1
[CH3:1][Si:2]([CH3:3])([CH3:4])[C:5]#[C:6][c:7]1[cH:8][c:9](-[c:10]2[cH:11][c:12](-[c:13]3[cH:14][cH:15][c:16]([CH2:17][N:18]4[CH2:19][CH2:20][CH2:21][CH2:22][CH2:23]4)[cH:24][cH:25]3)[cH:26][n:27][c:28]2[F:29])[c:30]([NH2:31])[cH:32][n:33]1"""

def predict_reactions(selected_file, molecule_string, use_rxn_class=False, beam_size=10, max_steps=9):
    # Basic SMILES validation
    if not molecule_string or not isinstance(molecule_string, str):
        raise ValueError('Input SMILES is empty or not a string.')
    mol = Chem.MolFromSmiles(molecule_string)  # type: ignore[attr-defined]
    if mol is None:
        raise ValueError('Failed to parse input SMILES (MolFromSmiles returned None).')
    try:
        Chem.SanitizeMol(mol)  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
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
    # Shared header with theme + logo toggle
    render_header(title='Prediction')

    # --- CSS (unified) ---
    st.markdown(
        """
        <style>
        .toolbar {display:flex; gap:.5rem; flex-wrap:wrap; align-items:center; margin:0.25rem 0 0.4rem;}
        .toolbar .spacer {flex:1 1 auto;}
        .summary-badge {padding:4px 10px; border-radius:12px; font-size:0.72rem; background:rgba(120,120,120,0.15);}
        .mono-small {font-size:0.7rem; font-family: var(--font-mono, monospace); opacity:0.8;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Initialize session keys ---
    st.session_state.setdefault('smiles_input', '')
    st.session_state.setdefault('predict_clicked', False)
    st.session_state.setdefault('batch_smiles_input', '')

    input_mode = st.radio("Input Mode", ["Single", "Batch"], horizontal=True, label_visibility='collapsed')

    # Helper callbacks
    def _single_example():
        st.session_state.smiles_input = example_smiles
    def _single_clear():
        # Clear input and associated single prediction artifacts
        st.session_state.smiles_input = ''
        # Remove previous results so display section disappears
        for k in ['results', 'product_smiles', 'df']:
            if k in st.session_state:
                del st.session_state[k]
        # Reset prediction trigger flag
        st.session_state.predict_clicked = False
        safe_toast('Single input cleared', icon='🧹')
    def _batch_example():
        # Insert multi-line batch example (3 SMILES)
        st.session_state.batch_smiles_input = batch_example_smiles
    def _batch_clear():
        st.session_state.batch_smiles_input = ''
        safe_toast('Batch inputs cleared', icon='🧹')

    single_predict_clicked = False
    batch_predict_clicked = False

    # --- Render Input Area ---
    if input_mode == 'Single':
        st.text_area(
            'Product SMILES',
            key='smiles_input',
            height=90,
            max_chars=500,
            help='Enter a single product SMILES (max 500 chars)'
        )
        with st.container():
            # Re-ordered: Example | Clear | Length | Predict (rightmost)
            c1, c2, c3, c4 = st.columns([1,1,2,1])
            with c1:
                st.button('Load Example', on_click=_single_example, help='Load example SMILES')
            with c2:
                st.button('Clear', on_click=_single_clear, help='Clear input')
            with c3:
                st.markdown(f"<div class='mono-small'>Length: {len(st.session_state.smiles_input)}</div>", unsafe_allow_html=True)
            with c4:
                single_predict_clicked = st.button('Single Predict', type='primary', help='Run retrosynthesis prediction')
    else:
        st.text_area(
            'Product SMILES (one per line)',
            key='batch_smiles_input',
            height=160,
            help='One product SMILES per line; blank lines ignored; max 100 lines.'
        )
        lines = [l.strip() for l in st.session_state.batch_smiles_input.splitlines() if l.strip()]
        with st.container():
            # Re-ordered to mirror single mode: Example | Clear | Length | Predict(right)
            c1, c2, c3, c4 = st.columns([1,1,2,1])
            with c1:
                st.button('Load Example', key='batch_example', on_click=_batch_example, help='Insert one example line')
            with c2:
                st.button('Clear', key='batch_clear', on_click=_batch_clear, help='Clear all lines')
            with c3:
                st.markdown(f"<div class='mono-small'>Valid lines: {len(lines)}</div>", unsafe_allow_html=True)
            with c4:
                batch_predict_clicked = st.button('Batch Predict', type='primary', help='Run prediction for all lines')

    # Normalize state for single predictions
    if input_mode == 'Single':
        if single_predict_clicked:
            st.session_state.predict_clicked = True
    else:
        st.session_state.predict_clicked = False  # disable single prediction flag in batch mode

    # --- Model availability ---
    model_path = st.session_state.get('selected_model_file')
    if not model_path:
        st.info('Select a model on the Selection page to enable prediction.')

    # --- SINGLE MODE EXECUTION ---
    if input_mode == 'Single':
        current_smi = st.session_state.smiles_input.strip()
        if st.session_state.predict_clicked:
            if not current_smi:
                st.warning('Input is empty.')
            elif not model_path:
                st.warning('No model selected.')
            else:
                try:
                    with st.spinner('Running prediction...'):
                        results, load_mode = predict_reactions(model_path, current_smi)
                    st.session_state.results = results
                    st.session_state.product_smiles = current_smi
                    df = pd.DataFrame(results)
                    df.insert(0, 'ID', range(1, len(df) + 1))
                    st.subheader('Predicted Precursors')
                    st.dataframe(df.set_index('ID'), use_container_width=True)
                    # Export utilities (single mode) - right aligned
                    exp_sp, exp_dl = st.columns([4,1])
                    with exp_dl:
                        st.download_button('Download CSV', data=df.to_csv(index=False).encode('utf-8'), file_name='single_prediction.csv', mime='text/csv', key='single_download_csv')
                    st.session_state.df = df
                    safe_toast(f'Single prediction finished: {len(df)} candidates', icon='✅')
                    if load_mode == 'full_pickle':
                        st.info('Checkpoint loaded with fallback full pickle (ensure it is trusted).')
                    # Reset flag to avoid re-run on refresh
                    st.session_state.predict_clicked = False
                except ValueError as ve:
                    st.error(f'Validation failed: {ve}')
                except FileNotFoundError as fe:
                    st.error(f'Model file not found: {fe}')
                except Exception as e:  # pragma: no cover
                    import traceback
                    traceback.print_exc()
                    st.error(f'Unexpected error: {e}')
        elif st.session_state.get('results'):
            # Show last prediction for continuity
            st.subheader('Last Prediction')
            df_prev = pd.DataFrame(st.session_state.results)
            if 'ID' not in df_prev.columns:
                df_prev.insert(0, 'ID', range(1, len(df_prev) + 1))
            st.dataframe(df_prev.set_index('ID'), use_container_width=True)

    # --- BATCH MODE EXECUTION ---
    else:
        raw_text = st.session_state.batch_smiles_input
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        if batch_predict_clicked:
            if not lines:
                st.warning('No valid SMILES lines provided.')
            elif len(lines) > 100:
                st.warning('Exceeded 100 line limit. Please reduce input.')
            elif not model_path:
                st.warning('No model selected.')
            else:
                aggregated = []
                errors = []
                with st.spinner('Running batch prediction...'):
                    for idx, smi in enumerate(lines, start=1):
                        try:
                            preds, _mode = predict_reactions(model_path, smi)
                            for rank, row in enumerate(preds, start=1):
                                aggregated.append({
                                    'Input_Index': idx,
                                    'Input_SMILES': smi,
                                    'Rank': rank,
                                    'Reactant_1': row['Reactant 1'],
                                    'Reactant_2': row['Reactant 2'],
                                    'Confidence': row['Confidence'],
                                })
                        except Exception as e:  # noqa
                            errors.append({'Input_Index': idx, 'Input_SMILES': smi, 'Error': str(e)})
                success_inputs = {r['Input_Index'] for r in aggregated}
                failed_inputs = {e['Input_Index'] for e in errors}
                st.markdown(
                    f"<div class='summary-badge'>Inputs: {len(lines)} | Success: {len(success_inputs)} | Failed: {len(failed_inputs)}</div>",
                    unsafe_allow_html=True
                )
                if aggregated:
                    st.subheader('Batch Prediction Results')
                    df_all = pd.DataFrame(aggregated)
                    st.dataframe(df_all, use_container_width=True)
                    # Structure for visualization page
                    structured = {}
                    for row in aggregated:
                        idx = row['Input_Index']
                        rec = structured.setdefault(idx, {'product': row['Input_SMILES'], 'predictions': []})
                        rec['predictions'].append({
                            'rank': row['Rank'],
                            'Reactant 1': row['Reactant_1'],
                            'Reactant 2': row['Reactant_2'],
                            'Confidence': row['Confidence'],
                        })
                    for rec in structured.values():
                        rec['predictions'].sort(key=lambda x: x['rank'])
                    st.session_state.batch_results = structured
                    # Download & copy utilities (batch) - right aligned
                    csv_data = df_all.to_csv(index=False).encode('utf-8')
                    b_sp, b_dl = st.columns([4,1])
                    with b_dl:
                        st.download_button('Download CSV', data=csv_data, file_name='batch_predictions.csv', mime='text/csv')
                    safe_toast(f'Batch prediction done: {len(success_inputs)} success / {len(failed_inputs)} failed', icon='✅' if not failed_inputs else 'ℹ️')
                else:
                    st.info('No successful predictions.')
                if errors:
                    with st.expander('Errors', expanded=False):
                        st.dataframe(pd.DataFrame(errors), use_container_width=True)
        else:
            # Passive info when not running
            st.caption(f'Valid lines: {len(lines)}')

    # --- Footer ---
    footer_text = """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 1px; background: transparent;'>
        <p>Copyright &copy; 2024 Wangz Team, SUMHS. All rights reserved.</p>
    </div>
    """
    st.markdown(footer_text, unsafe_allow_html=True)
    
if __name__ == '__main__':
    main()