import base64
import os
import streamlit as st
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent / 'experiments' / 'uspto_50k'


def select_model_and_logs():
    """Return selected model checkpoint and its log CSV."""
    options_with_paths = []
    for category in ['without_rxn_class', 'with_rxn_class']:
        category_path = os.path.join(ROOT_DIR, category)
        for subfolder in os.listdir(category_path):
            subfolder_path = os.path.join(category_path, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.endswith('.pt'):
                        option_display = f"{os.path.basename(subfolder)}\\{file}"
                        options_with_paths.append(
                            (os.path.join(subfolder_path, file), os.path.join(subfolder_path, 'logs.csv'), option_display)
                        )
    selected_option_display = st.selectbox("Select a model checkpoint", [o[2] for o in options_with_paths])
    selected_model_file, selected_logs_file = next(
        ((o[0], o[1]) for o in options_with_paths if o[2] == selected_option_display), (None, None)
    )
    return selected_model_file, selected_logs_file


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
