
import os
import torch
import pandas as pd
import streamlit as st
from pathlib import Path
from rdkit import RDLogger
from models import Graph2Edits, BeamSearch
from st_aggrid import AgGrid, GridOptionsBuilder, AgGridTheme


lg = RDLogger.logger()
lg.setLevel(4)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 定义模型文件的根目录
ROOT_DIR = Path(__file__).parent.parent / 'experiments' / 'uspto_50k'

# 创建一个示例SMILES字符串
example_smiles = "[O:1]=[S:19]([c:18]1[cH:17][c:15]([Cl:16])[c:14]2[o:13][c:12]3[c:28]([c:27]2[cH:26]1)[CH2:29][N:9]([C:7]([O:6][C:3]([CH3:2])([CH3:4])[CH3:5])=[O:8])[CH2:10][CH2:11]3)[c:20]1[cH:21][cH:22][cH:23][cH:24][cH:25]1"

def predict_reactions(selected_file,molecule_string, use_rxn_class=False, beam_size=10, max_steps=9):
    
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
            '反应物1': new_pred_smi_field_1,
            '反应物2': new_pred_smi_field_2,
            '可靠性': f'{float(prob * 100):.2f}%',
            #'编辑步骤': str_edits
        }
    
    # Filter out predictions with 'prediction': 'final_smi_unmapped'
    filtered_predictions = [format_prediction(beam_idx, path) for beam_idx, path in enumerate(top_k_results) if path['final_smi'] != 'final_smi_unmapped']

    # Remove duplicate predictions based on the 'prediction' key
    unique_predictions = []
    seen_predictions = set()
    for prediction in filtered_predictions:
        if prediction['反应物1'] not in seen_predictions:
            unique_predictions.append(prediction)
            seen_predictions.add(prediction['反应物1'])
    print(unique_predictions)

    return unique_predictions

def main():

    # 添加学校logo图
    st.markdown("""
    <style>
    .logo {{
        position: absolute;
        top: 0px;  /* 调整距离顶部的位置 */
        left: 600px; /* 调整距离左侧的位置 */
        width: 100px; /* 调整校徽的宽度 */
        height: 100px; /* 调整校徽的高度 */
    }}
    </style>
    <img class="logo" src="data:image/png;base64,{}" alt="校徽">
    """.format(st.session_state.img_base64), unsafe_allow_html=True)

    st.title('预测反应结果')

    # 初始化session_state，如果之前没有设置过，则设置默认值
    if 'smiles_input' not in st.session_state:
        st.session_state.smiles_input = ''

    # 使用session_state中的值作为文本输入的默认值
    smiles_input = st.text_area(label = '请输入药物分子的SMILES式', 
                        value=st.session_state.smiles_input, 
                        height=5, 
                        max_chars=500, 
                        help='最大长度限制为500')

    # 创建一个按钮，允许用户选择使用示例SMILES
    use_example = st.button('使用示例SMILES进行测试')

    # 更新session_state中的值，以便在后续使用中保持一致性
    st.session_state.smiles_input = smiles_input

    try:
        selected_model_file = st.session_state.selected_model_file
    except:
        st.error('请先选择模型文件！')

    # 根据按钮状态设置SMILES输入
    if use_example:
        # 当用户点击示例按钮时，更新session_state中的SMILES输入
        st.session_state.smiles_input = example_smiles
        # 重新赋值给smiles_input，以便在当前运行中使用
        smiles_input = example_smiles

    if smiles_input:

        st.subheader("模型预测结果表")

        try:
            
            # 调用 predict_reactions 函数
            results = predict_reactions(selected_model_file,smiles_input)
            st.session_state.results = results

            # 将结果转换为DataFrame
            df = pd.DataFrame(results)
            
            # 在DataFrame中添加从1开始的编号作为新的一列
            df.insert(0, '编号', range(1, len(df) + 1))
            
            # 将第一列设置为行名
            df.index = df.iloc[:,0]
            df.drop(df.columns[0], axis=1, inplace=True)
            st.write(df)

            df.insert(0, '编号', range(1, len(df) + 1))
            st.session_state.df = df
            
        except:
            st.error('输入的SMILES式不合法！')

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