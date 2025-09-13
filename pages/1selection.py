import base64
import os
import streamlit as st
import pandas as pd
from pathlib import Path
import json
from rdkit import Chem  # kept for potential future extension
import torch
from subprocess import CalledProcessError, run

ROOT_DIR = Path(__file__).parent.parent / 'experiments' / 'uspto_50k'


def _probe_checkpoint_version(path: str):
    try:
        ck = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(ck, dict) and ck.get('format_version') == 2:
            return 2, 'weights_only'
    except Exception:
        pass
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(ck, dict):
            fv = ck.get('format_version')
            if fv == 2:
                return 2, 'full_pickle'
            elif fv == 1:
                return 1, 'full_pickle'
            elif 'saveables' in ck and 'state' in ck:
                return 'legacy', 'full_pickle'
    except Exception:
        return 'error', 'error'
    return 'unknown', 'unknown'


def _convert_checkpoint(path: str):
    # Invoke existing conversion script to produce _safe file
    safe_path = Path(path).with_name(Path(path).stem + '_safe.pt')
    cmd = [
        'python', 'convert_checkpoint.py',
        '--input', path,
        '--output', str(safe_path),
        '--verify'
    ]
    try:
        r = run(cmd, capture_output=True, text=True, check=True)
        return True, safe_path, r.stdout
    except CalledProcessError as e:
        return False, safe_path, e.stderr


def select_model_and_logs():
    options_with_paths = []
    for category in ['without_rxn_class', 'with_rxn_class']:
        category_path = os.path.join(ROOT_DIR, category)
        for subfolder in os.listdir(category_path):
            subfolder_path = os.path.join(category_path, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.endswith('.pt'):
                        full_path = os.path.join(subfolder_path, file)
                        fv, mode = _probe_checkpoint_version(full_path)
                        label = f"{os.path.basename(subfolder)}\\{file} [v:{fv}|{mode}]"
                        options_with_paths.append(
                            (full_path, os.path.join(subfolder_path, 'logs.csv'), label, fv)
                        )
    if not options_with_paths:
        st.warning('No checkpoints found.')
        return None, None

    selected_label = st.selectbox("Select a model checkpoint", [o[2] for o in options_with_paths])
    # Batch conversion button
    if st.button('Batch Convert All Legacy/V1 to V2 Safe'):
        with st.spinner('Batch converting...'):
            import subprocess, sys
            # Use unified script (convert_checkpoint.py) batch mode
            cmd = [sys.executable, 'convert_checkpoint.py', '--batch-root', str(ROOT_DIR)]
            out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode == 0:
            st.success('Batch conversion complete')
            st.expander('Details').code(out.stdout)
        else:
            st.error('Batch conversion failed')
            st.expander('Details (stderr)').code(out.stderr)
    selected = next((o for o in options_with_paths if o[2] == selected_label), None)
    if not selected:
        return None, None

    model_path, logs_path, label, fv = selected

    if fv != 2 and fv != 'error':
        if st.button('Convert to format_version=2 (safe)'):
            with st.spinner('Converting...'):
                ok, safe_path, msg = _convert_checkpoint(model_path)
            if ok:
                st.success(f'Converted -> {safe_path}')
            else:
                st.error('Conversion failed')
                st.code(msg)
    return model_path, logs_path


def main():
    with open('./assets/logo.jpg', 'rb') as file:
        img_base64 = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>
        .logo {{ position: absolute; top:0; left:600px; width:90px; height:90px; }}
        </style>
        <img class="logo" src="data:image/png;base64,{img_base64}" alt="Logo" />
        """,
        unsafe_allow_html=True,
    )

    st.title("Selection")
    selected_model_file, selected_logs_file = select_model_and_logs()

    if selected_model_file and selected_logs_file:
        st.session_state.selected_model_file = selected_model_file
        st.caption(f"Path: {selected_model_file}")

        st.subheader("Training Curves")
        train_df = pd.read_csv(selected_logs_file, skiprows=19)
        column_names = ['epoch', 'train_acc', 'valid_acc', 'valid_first_step_acc', 'train_loss']
        train_df.columns = column_names
        # set epoch as index
        train_df = train_df.set_index(train_df.columns[0])
        st.write(train_df)

        available_columns = train_df.columns.tolist()
        selected_columns = st.multiselect('Select metrics to plot', available_columns, default=['valid_acc'])
        if selected_columns:
            chart_data = train_df[selected_columns]
            st.line_chart(chart_data)

        config_df = pd.read_csv(selected_logs_file, nrows=18, header=None)
        config_df.columns = config_df.iloc[0]
        config_df = config_df.drop(0).reset_index(drop=True)
        # set first column as index (label column)
        config_df = config_df.set_index(config_df.columns[0])
        st.subheader("Hyperparameters")
        st.write(config_df)

    footer_text = """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 1px; background: transparent;'>
        <p style='font-size:12px; color:#666;'>© 2024 Wangz Team, SUMHS. All rights reserved.</p>
    </div>
    """
    st.markdown(footer_text, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
