import pandas as pd
import re
from rdkit.Chem import AllChem
import sys
import json

fp_test_predict=open(sys.argv[1])
fp_result=open(sys.argv[2],"w")

answer_dict={}
i=0
for line_id in fp_test_predict:
    str_i = str(i)
    _0_len = 6 - len(str_i)
    i_id = "0" * _0_len + str_i
    i+=1
    line_predict=fp_test_predict.readline().strip()
    answer_dict[i_id]=line_predict.replace(" ", "")

if len(answer_dict)!=238282:
    print("len %s err"%(i))
    exit(1)


answer_df = pd.DataFrame.from_dict(answer_dict,orient='index')
answer_df.reset_index(level=0, inplace=True)
answer_df.columns = ['id','reactants']

def standardize(str_tokens):
    remove_atom_mapping_tokens = re.sub(r'\:\d*','',str_tokens)
    standarded_smiles = []
    for token in remove_atom_mapping_tokens.split('.'):
        # 如果你的答案中出现了多余的双引号，去除
        if '\"' in token:
            token = token.strip('\"')
        # 开始统一标准化，如果分子是有效存在的，转成SMILES格式
        mol = AllChem.MolFromSmiles(token, sanitize=False)
        if mol:
            mol_smiles = AllChem.MolToSmiles(mol)
            standarded_smiles.append(mol_smiles)
    standarded_smiles = str.join('.',standarded_smiles)
    return standarded_smiles

answer_df['reactants'] = answer_df['reactants'].apply(standardize)
answer_df.set_index('id')['reactants'].to_json('submit_answer.json')