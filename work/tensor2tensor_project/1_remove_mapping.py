# coding: utf-8

from __future__ import print_function
import os
import re
import json
import traceback
import sys
import random
import pickle, gzip
from rdkit.Chem import AllChem
from rdkit import Chem
import copy
#import parser.Smipar as Smipar
#sys.argv=["","../data/","./tmp"]
def cano(smiles): # canonicalize smiles by MolToSmiles function
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) if (smiles != '') else ''

def main():
    from_data_path=sys.argv[1]
    to_data_path=sys.argv[2]
    fp_train = open(from_data_path+"/train_all.txt")
    fp_test = open(from_data_path + "/test_all.txt")

    fp_train_source = open(to_data_path+"/inputs.train.txt", "w")
    fp_train_target=open(to_data_path+"/targets.train.txt", "w")
    fp_train_id = open(to_data_path + "/train_id", "w")
    fp_eval_source = open(to_data_path + "/inputs.eval.txt", "w")
    fp_eval_target = open(to_data_path + "/targets.eval.txt", "w")
    fp_eval_id = open(to_data_path + "/eval_id", "w")
    fp_test_source  = open(to_data_path+"/test_sample", "w")
    fp_test_id = open(to_data_path+"/test_id", "w")
    fp_vocab=open(to_data_path +"/vocab.txt","w")
    vocab= {}

    token_regex = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|= |  # |-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    # id,reactants>reagents>production
    fp_train.readline()
    for line in fp_train:
        line = line.replace('"', '')
        if " |" in line:
            tmp_fields = line.strip().split(" |")
            line = tmp_fields[0]
        fields = line.strip().split(",")
        sample_id = fields[0]
        rxn_str = fields[1]
        try:
            rxn = AllChem.ReactionFromSmarts(rxn_str, useSmiles=True)
        except:
            print(rxn_str, file=sys.stderr)
            print(sample_id, file=sys.stderr)
            traceback.print_exc()
            print("\n", file=sys.stderr)
            continue
        AllChem.RemoveMappingNumbersFromReactions(rxn)  # 去原子的号码
        no_mapping_smiles = AllChem.ReactionToSmiles(rxn)

        #resort
        reactant,reagent, product = no_mapping_smiles.strip().split(">")
        """
        try:
            reactant=cano(reactant)
            product = cano(product)
            reactant = cano(reactant)
        except:
            print(rxn_str, file=sys.stderr)
            print(sample_id, file=sys.stderr)
            traceback.print_exc()
            print("\n", file=sys.stderr)
            #continue
        """
        reactant_fields=reactant.strip().split(".")
        reactant_fields_fields=[]
        for an_reactant in reactant_fields:
            an_reactant_token_fields=[]
            token_list = re.split(token_regex, an_reactant)
            if "." not in vocab:
                vocab["."]=0
            vocab["."]+=1
            for token in token_list:
                token = token.strip()
                if token == '':
                    continue
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1
                an_reactant_token_fields.append(token)
            reactant_fields_fields.append(an_reactant_token_fields)

        product_fields = product.strip().split(".")
        product_fields_fields = []
        for an_product in product_fields:
            an_product_token_fields = []
            token_list = re.split(token_regex, an_product)
            if "." not in vocab:
                vocab["."] = 0
            vocab["."] += 1
            for token in token_list:
                token = token.strip()
                if token == '':
                    continue
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1
                an_product_token_fields.append(token)
            product_fields_fields.append(an_product_token_fields)

        reagent_fields = reagent.strip().split(".")
        reagent_fields_fields = []
        for an_reagent in reagent_fields:
            an_reagent_token_fields = []
            token_list = re.split(token_regex, an_reagent)
            if "." not in vocab:
                vocab["."] = 0
            vocab["."] += 1
            for token in token_list:
                token = token.strip()
                if token == '':
                    continue
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1
                an_reagent_token_fields.append(token)
            reagent_fields_fields.append(an_reagent_token_fields)



        if int(sample_id)%10==0:
            out_reagent_fields=[]
            for an_reagent_token_fields in reagent_fields_fields:
                out_an_reagent =" ".join(an_reagent_token_fields)
                out_reagent_fields.append(out_an_reagent)
            out_reagent=" . ".join(out_reagent_fields)
            out_product_fields = []
            for an_product_token_fields in product_fields_fields:
                out_an_product = " ".join(an_product_token_fields)
                out_product_fields.append(out_an_product)
            out_product = " . ".join(out_product_fields)
            out_reactant_fields = []
            for an_reactant_token_fields in reactant_fields_fields:
                out_an_reactant = " ".join(an_reactant_token_fields)
                out_reactant_fields.append(out_an_reactant)
            out_reactant = " . ".join(out_reactant_fields)
            out_reagent_product=" > ".join([out_reagent,out_product])

            print(out_reagent_product,file=fp_eval_source)
            print(out_reactant, file=fp_eval_target)
            print(sample_id, file=fp_eval_id)
        else:

            for i in range(5):
                random.shuffle(reagent_fields_fields)
                random.shuffle(product_fields_fields)
                random.shuffle(reactant_fields_fields)

                out_reagent_fields=[]
                for an_reagent_token_fields in reagent_fields_fields:
                    out_an_reagent =" ".join(an_reagent_token_fields)
                    out_reagent_fields.append(out_an_reagent)
                out_reagent=" . ".join(out_reagent_fields)
                out_product_fields = []
                for an_product_token_fields in product_fields_fields:
                    out_an_product = " ".join(an_product_token_fields)
                    out_product_fields.append(out_an_product)
                out_product = " . ".join(out_product_fields)
                out_reactant_fields = []
                for an_reactant_token_fields in reactant_fields_fields:
                    out_an_reactant = " ".join(an_reactant_token_fields)
                    out_reactant_fields.append(out_an_reactant)
                out_reactant = " . ".join(out_reactant_fields)
                out_reagent_product=" > ".join([out_reagent,out_product])

                print(out_reagent_product, file=fp_train_source)
                print(out_reactant, file=fp_train_target)
                print(sample_id, file=fp_train_id)


    # id,reagents>production
    fp_test.readline()
    for line in fp_test:
        line = line.replace('"', '')
        if " |" in line:
            tmp_fields = line.strip().split(" |")
            line = tmp_fields[0]
        fields = line.strip().split(",")
        sample_id = fields[0]
        rxn_str = fields[1]
        rxn_str = "O>" + rxn_str
        try:
            rxn = AllChem.ReactionFromSmarts(rxn_str, useSmiles=True)
        except:
            print(rxn_str, file=sys.stderr)
            print(sample_id)
            traceback.print_exc()
            continue
        AllChem.RemoveMappingNumbersFromReactions(rxn)  # 去原子的号码
        no_mapping_smiles = AllChem.ReactionToSmiles(rxn)

        # resort
        _, reagent,product = no_mapping_smiles.strip().split(">")
        """
        try:
            reactant=cano(reactant)
            product = cano(product)
        except:
            print(rxn_str, file=sys.stderr)
            print(sample_id, file=sys.stderr)
            traceback.print_exc()
            print("\n", file=sys.stderr)
            continue
        """
        product_fields = product.strip().split(".")
        #product_fields.sort(key=lambda p: len(p))
        product = ".".join(product_fields)
        reagent_fields = reagent.strip().split(".")
        #reagent_fields.sort(key=lambda p: len(p))
        reagent = ".".join(reagent_fields)
        reagent_product=reagent+">"+product

        #vocab
        reagent_product_list=[]
        token_list=re.split(token_regex, reagent_product)
        for token in token_list:
            token=token.strip()
            if token=='':
                continue
            if token not in vocab:
                vocab[token]=0
            vocab[token]+=1
            reagent_product_list.append(token)
        print(" ".join(list(reagent_product_list)), file=fp_test_source)
        print(sample_id, file=fp_test_id)

    print(sorted(vocab.items(),key=lambda p:p[1]))
    out_vocab_list=[]
    for token in vocab:
        if vocab[token]>=10:
            out_vocab_list.append(token)
    print("\n".join(out_vocab_list), file=fp_vocab)

if __name__=="__main__":
    main()



    """
    reactant_list = []
    agent_list = []
    product_list = []
    split_rsmi = output_smiles.split('>')
    try:
        #reactants = cano(split_rsmi[0]).split('.')
        reactants = split_rsmi[0].split('.')
    except:
        print(split_rsmi[0])
        print(id)
        traceback.print_exc()
        continue
    try:
        #agents = cano(split_rsmi[1]).split('.')
        agents = split_rsmi[1].split('.')
    except:
        print(split_rsmi[1])
        print(id)
        traceback.print_exc()
        continue
    try:
        #products = cano(split_rsmi[2]).split('.')
        products = split_rsmi[2].split('.')
    except:
        print(split_rsmi[2])
        print(id)
        traceback.print_exc()
        continue 
    token_regex = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|= |  # |-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    for reactant in reactants:
        #print(re.split(token_regex, reactant))
        #reactant_list += Smipar.parser_list(reactant)
        reactant_list+=re.split(token_regex, reactant)
        reactant_list += '.'
    for agent in agents:
        #agent_list += Smipar.parser_list(agent)
        agent_list+=re.split(token_regex, agent)
        agent_list += '.'
    for product in products:
        #product_list += Smipar.parser_list(product)
        product_list+=re.split(token_regex, product)
        product_list += '.'
    reactant_list.pop()  # to pop last '.'
    agent_list.pop()
    product_list.pop()
    product_list += '>'
    product_list += agent_list
    out_product_list=[]
    out_reactant_list=[]
    for token in product_list:
        strip_token=token.strip()
        if strip_token!="":
            out_product_list.append(strip_token)
    for token in reactant_list:
        strip_token = token.strip()
        if strip_token != "":
            out_reactant_list.append(strip_token)
    print(" ".join(out_product_list),file=fp_train_sources)
    print(" ".join(out_reactant_list), file=fp_train_targets)
    for reactant_token in reactant_list:
        if reactant_token in vocab:
            vocab[reactant_token] += 1
        else:
            if reactant_token=="Cl":
                print("sdfsf")
            if reactant_token=="6":
                print("sdfsf")
            vocab[reactant_token] = 1
    for product_token in product_list:
        if product_token in vocab:
            vocab[product_token] += 1
        else:
            if product_token=="Cl":
                print("sdfsf")
            if product_token=="6":
                print("sdfsf")
            vocab[product_token] = 1
    """
