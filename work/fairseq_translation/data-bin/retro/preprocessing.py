import sys
import random
fp_train_sources=open("train_sources")
fp_train_targets=open("train_targets")



fp_dict_sources=open("dict.sources.txt","w")
fp_dict_targets=open("dict.targets.txt","w")

fp_train_sources_out=open("train.sources-targets.sources","w")
fp_train_targets_out=open("train.sources-targets.targets","w")
fp_test_sources_out=open("test.sources-targets.sources","w")
fp_test_targets_out=open("test.sources-targets.targets","w")
fp_valid_sources_out=open("valid.sources-targets.sources","w")
fp_valid_targets_out=open("valid.sources-targets.targets","w")


vocab_dict={}
for source_line in fp_train_sources:
    target_line=fp_train_targets.readline().rstrip()
    source_line=source_line.rstrip()
    target_fields=source_line.split(" ")
    for token in target_fields:
        vocab_dict[token]=vocab_dict.get(token,0)+1
    source_fields=source_line.split(" ")
    for token in source_fields:
        vocab_dict[token]=vocab_dict.get(token,0)+1

    print(source_line, file=fp_train_sources_out)
    print(target_line, file=fp_train_targets_out)

    if random.random()<0.1:
        print(source_line, file=fp_test_sources_out)
        print(target_line, file=fp_test_targets_out)

        print(source_line, file=fp_valid_sources_out)
        print(target_line, file=fp_valid_targets_out)


for token,freq in vocab_dict.items():
    print("%s %s"%(token,freq),file=fp_dict_targets)
    print("%s %s" % (token, freq), file=fp_dict_sources)
