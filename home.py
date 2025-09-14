import streamlit as st
from utils.ui import render_header


def main():
    """Landing page of the RetroSynthesis Predictor UI."""
    # Unified header (logo + theme toggle + page title)
    render_header(title='Retrosynthesis Prediction Tool')
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




