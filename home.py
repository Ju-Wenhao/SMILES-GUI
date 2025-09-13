import base64
import streamlit as st


def main():
    """Landing page of the RetroSynthesis Predictor UI."""
    # Load logo as Base64
    with open('./assets/logo.jpg', 'rb') as file:
        img_base64 = base64.b64encode(file.read()).decode()

    # Logo (positioned absolutely)
    st.markdown(
        f"""
        <style>
        .logo {{ position: absolute; top:0; left:600px; width:90px; height:90px; }}
        </style>
        <img class="logo" src="data:image/png;base64,{img_base64}" alt="Logo" />
        """,
        unsafe_allow_html=True,
    )

    st.title('Retrosynthesis Prediction Tool')
    st.caption('Powered by the G2G-MAML model')

    st.markdown(
        """
        ### Quick Start
        1. **Select Model**: In the sidebar page `Selection`, choose a trained model and review its metrics.
        2. **Enter Molecule**: On `Prediction`, paste a product SMILES string (or use the example).
        3. **View Candidates**: The model suggests plausible precursor sets ranked by confidence.
        4. **Visualize**: On `Visualization`, render molecules in 2D or 3D and inspect a specific result.

        ### Notes
        * Predictions are theoretical and must be validated experimentally.
        * Confidence is a relative probability within the beam.
        """
    )

    footer_text = """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 1px; background: transparent;'>
        <p style='font-size:12px; color:#666;'>© 2024 Wangz Team, SUMHS. All rights reserved.</p>
    </div>
    """
    st.markdown(footer_text, unsafe_allow_html=True)


if __name__ == "__main__":
    main()




