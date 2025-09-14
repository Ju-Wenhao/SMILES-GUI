import os
import streamlit as st
import pandas as pd
from pathlib import Path
import torch

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
    selected = next((o for o in options_with_paths if o[2] == selected_label), None)
    if not selected:
        return None, None

    model_path, logs_path, label, fv = selected
    return model_path, logs_path


from utils.ui import render_header


def main():
    render_header(title="Selection")
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

        # Read raw config header block
        config_df = pd.read_csv(selected_logs_file, nrows=18, header=None)
        # Defensive: drop completely empty rows (all NaN)
        config_df = config_df.dropna(how='all')
        if not config_df.empty:
            # First row considered header
            header_row = config_df.iloc[0]
            config_df = config_df[1:]
            config_df.columns = header_row
            # If first column name is NaN/None, assign a placeholder
            first_col_name = config_df.columns[0]
            if pd.isna(first_col_name) or first_col_name is None or first_col_name == '':
                first_col_name = 'param'
                cols = list(config_df.columns)
                cols[0] = first_col_name
                config_df.columns = cols
            # Set first column as index
            config_df = config_df.set_index(first_col_name)
            # Remove columns that are entirely NaN (avoid serialization of all-NaN columns)
            config_df = config_df.dropna(axis=1, how='all')
            # Replace remaining NaN with None so Streamlit JSON serialization uses null (valid JSON)
            config_df = config_df.where(pd.notnull(config_df), None)
            # Also ensure index has no NaN
            new_index = ['(blank)' if (isinstance(ix, float) and (ix != ix)) else ix for ix in config_df.index]
            # Use set_axis to avoid direct assignment issues in some static analyzers
            config_df = config_df.set_axis(new_index, axis=0)
            st.subheader("Hyperparameters")
            try:
                st.write(config_df)
            except Exception as e:
                st.warning(f"Could not render table directly ({e}); showing plain text.")
                # Fallback textual representation
                lines = []
                for idx, row in config_df.iterrows():
                    # Join non-null values
                    vals = [str(v) for v in row.tolist() if v is not None]
                    lines.append(f"{idx}: {' | '.join(vals)}")
                st.code('\n'.join(lines))
        else:
            st.subheader("Hyperparameters")
            st.info('No hyperparameter rows found in log header.')

    footer_text = """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 1px; background: transparent;'>
        <p style='font-size:12px; color:#666;'>© 2024 Wangz Team, SUMHS. All rights reserved.</p>
    </div>
    """
    st.markdown(footer_text, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
