# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generator and model for translation
with multiple source features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import os

# Dependency imports

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

class VocabType(object):
  """Available text vocabularies."""
  SUBWORD = "subwords"
  TOKEN = "tokens"


"""
@registry.register_hparams
def transformer_sfeats_hparams():
  # define initial transformer hparams here
  hp = transformer.transformer_base()
  #hp = transformer.transformer_big()

  # feature vector size setting
  # the order of the features is the same
  # as in the source feature file. All
  # sizes are separated by ':'
  hp.add_hparam("source_feature_embedding_sizes", "16:56:8")
  # set encoder hidden size
  ehs = sum([int(size) for size in hp.source_feature_embedding_sizes.split(':')])
  ehs += hp.hidden_size
  hp.add_hparam("enc_hidden_size", ehs)
  return hp
"""

@registry.register_problem
class MyReactionToken(text_problems.Text2TextProblem):
    """Problem spec for translation with source features."""

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return True


    @property
    def approx_vocab_size(self):
        return 2 ** 9

    @property
    def vocab_filename(self):
        return "vocab.token"

    @property
    def sfeat_delimiter(self):
        r"""Source feature delimiter in feature file"""
        return '|'

    @property
    def use_subword_tags(self):
        r"""use subword tags: these will be generated
        when the source words are subword encoded.
        This source feature is the last one among
        all other features and its vector size must
        be set in hparams (source_feature_embedding_sizes).
        """
        return False

    @property
    def oov_token(self):
        """Out of vocabulary token. Only for VocabType.TOKEN."""
        return "<UNK>"


    @property
    def vocab_type(self):
        """What kind of vocabulary to use.

        `VocabType`s:
          * `SUBWORD`: `SubwordTextEncoder`, an invertible wordpiece vocabulary.
            Must provide `self.approx_vocab_size`. Generates the vocabulary based on
            the training data. To limit the number of samples the vocab generation
            looks at, override `self.max_samples_for_vocab`. Recommended and
            default.
          * `CHARACTER`: `ByteTextEncoder`, encode raw bytes.
          * `TOKEN`: `TokenTextEncoder`, vocabulary based on a file. Must provide a
            vocabulary file yourself (`TokenTextEncoder.store_to_file`) because one
            will not be generated for you. The vocab file should be stored in
            `data_dir/` with the name specified by `self.vocab_filename`.

        Returns:
          VocabType constant
        """
        return VocabType.TOKEN

    def vocab_sfeat_filenames(self, f_id):
        r"""
            reweite the father class

        One vocab per feature type"""
        return "vocab.enfr.sfeat.%d" % f_id


    def generate_samples(self, data_dir, tmp_dir, dataset_split):

        if self.vocab_type != "tokens":
            raise ValueError("VocabType not supported")

        if dataset_split==dataset_split == problem.DatasetSplit.TRAIN:
            in_file_name="train_sources"
            out_file_name="train_targets"
            feature_file_name="train_features"

        else:#验证集
            in_file_name="train_sources"
            out_file_name="train_targets"
            feature_file_name="train_features"
        in_file = os.path.join(tmp_dir,in_file_name)
        out_file = os.path.join(tmp_dir, out_file_name)
        feature_file = os.path.join(tmp_dir,feature_file_name)
        in_fp=open(in_file)
        out_fp = open(out_file)
        feature_fp = open(feature_file)
        in_list=in_fp.readlines()
        out_list=out_fp.readlines()
        feature_list=feature_fp.readlines()

        for line1, line2,line3 in zip(in_list, out_list,feature_list):
            input_line=" ".join(line1.replace("\n", " %s " % EOS).split())
            targets_line= " ".join(line2.replace("\n", " %s " % EOS).split())
            feature_line = " ".join(line3.replace("\n", " %s " % EOS).split())
            yield {
              "inputs": line1,
              "targets": line2,
              "sfeats":line3
            }

    def create_src_feature_vocabs(self, data_dir, tmp_dir):
        r"""
        Generate as many vocabularies as there are source feature types.
        """
        source = self.vocab_data_files()[0][2]
        vocab_file = os.path.join(data_dir, self.vocab_sfeat_filenames(0))
        if os.path.isfile(vocab_file):
            tf.logging.info("Found source feature vocabs: %s", vocab_file[:-1] + "*")
            return

        filepath = os.path.join(tmp_dir, source)
        sfeat_vocab_lists = defaultdict(lambda: set())
        tf.logging.info("Generating source feature vocabs from %s", filepath)
        with tf.gfile.GFile(filepath, mode="r") as source_file:
            for line in source_file:
                feat_sets = [fs.split(self.sfeat_delimiter) for fs in line.strip().split()]
                for f_id, _ in enumerate(feat_sets[0]):
                    feat = [fs[f_id] for fs in feat_sets]
                    sfeat_vocab_lists[f_id].update(feat)

        if self.use_subword_tags:
            tf.logging.info("Generating subword tag vocab")
            f_id = len(sfeat_vocab_lists)
            sfeat_vocab_lists[f_id] = {"B", "I", "E", "O"}

        sfeat_vocabs = {}
        for f_id in sfeat_vocab_lists:
            vocab = text_encoder.TokenTextEncoder(
                vocab_filename=None,
                vocab_list=sfeat_vocab_lists[f_id])
            vocab_filepath = os.path.join(data_dir, self.vocab_sfeat_filenames(f_id))
            vocab.store_to_file(vocab_filepath)

    def compile_sfeat_data(self, tmp_dir, datasets, filename):
        filename = os.path.join(tmp_dir, filename)
        src_feat_fname = filename + '.sfeat'
        for dataset in datasets:
            try:
                src_feat_filepath = os.path.join(tmp_dir, dataset[2])
            except IndexError:
                if self.use_subword_tags:
                    raise IndexError("No source feature file given.",
                                     "Using only subword tags is not allowed.")
                else:
                    raise IndexError("No source feature file given.")
            with tf.gfile.GFile(src_feat_fname, mode="w") as sf_resfile:
                with tf.gfile.Open(src_feat_filepath) as f:
                    for src_feats in f:
                        sf_resfile.write(src_feats.strip())
                        sf_resfile.write("\n")
        return filename

    def get_subword_tags(self, subword_nb):
        r"""Get subword tags as an additional feature:
        B: beginning of a word
        I: inside
        E: end
        O: full word
        """
        if subword_nb == 1:
            feat = ['O']
        else:
            feat = ['B', 'E']
            while len(feat) < subword_nb:
                feat.insert(1, 'I')
        return feat

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        """
        ********************which call the generate_samples

        :param data_dir:
        :param tmp_dir:
        :param dataset_split:
        :return:
        """
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        txt_encoder = self.get_or_create_vocab(data_dir, tmp_dir)
        feat_encoders = self.get_src_feature_encoders(data_dir, tmp_dir)

        for sample in generator:
            new_input = []
            split_counter = []
            for token in sample["inputs"].split():
                if self.vocab_type == "subwords":
                    new_toks = txt_encoder.encode_without_tokenizing(token)
                elif self.vocab_type == "tokens":
                    new_toks = txt_encoder.encode(token)
                else:
                    raise ValueError("VocabType not supported")
                new_input += new_toks
                split_counter.append(len(new_toks))
            sample["inputs"] = new_input
            sample["inputs"].append(EOS)

            feat_seqs = defaultdict(lambda: [])
            for tok_id, feat_set in enumerate(sample["sfeats"].split()):
                for f_id, feat in enumerate(feat_set.split(self.sfeat_delimiter)):
                    # synchronize feature with subword
                    feat = [feat] * split_counter[tok_id]
                    feat_seqs[f_id] += feat
                if self.use_subword_tags:
                    f_id += 1
                    feat = self.get_subword_tags(split_counter[tok_id])
                    feat_seqs[f_id] += feat

            del sample["sfeats"]

            for f_id in range(len(feat_seqs)):
                fs = feat_encoders[f_id].encode(' '.join(feat_seqs[f_id]))
                fs.append(EOS)
                assert len(fs) == len(sample["inputs"]), "Source word and feature sequences must have the same length"
                sample["sfeats.%d" % f_id] = fs

            new_sample = []
            for token in sample["targets"].split():
                if self.vocab_type == "subwords":
                    new_toks = txt_encoder.encode_without_tokenizing(token)
                elif self.vocab_type == "tokens":
                    new_toks = txt_encoder.encode(token)
                else:
                    raise ValueError("VocabType not supported")
                new_sample += new_toks

            sample["targets"] = new_sample
            sample["targets"].append(EOS)

            yield sample

    def feature_encoders(self, data_dir):
        """
        ************rewrite the father class*****************88
        data generation for training"""
        encoders = super().feature_encoders(data_dir)
        feat_encoders = self.get_src_feature_encoders(data_dir)
        for f_id, encoder in enumerate(feat_encoders):
            encoders["sfeats.%d" % f_id] = encoder
        return encoders

    def example_reading_spec(self):
        data_fields, data_items_to_decoders = super().example_reading_spec()
        # add source features
        sfeats = [feat for feat in self.get_feature_encoders() if feat.startswith("sfeats")]
        for sfeat in sfeats:
            data_fields[sfeat] = tf.VarLenFeature(tf.int64)
        return (data_fields, data_items_to_decoders)

    def get_src_feature_encoders(self, data_dir, tmp_dir=None):
        r"""build source feature encoders"""
        feat_encoders = []
        i = 0
        current_path = data_dir + '/' + self.vocab_sfeat_filenames(i)
        if not os.path.isfile(current_path):
            self.create_src_feature_vocabs(data_dir, tmp_dir)

        # Search for feature vocab files on disc
        """
    
        this position determine the num of additional feature
    
        """
        while os.path.isfile(current_path):
            feat_encoders.append(text_encoder.TokenTextEncoder(current_path))
            i += 1
            current_path = data_dir + '/' + self.vocab_sfeat_filenames(i)
        return feat_encoders

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.stop_at_eos = int(True)
        source_vocab_size = self._encoders["inputs"].vocab_size
        p.input_modality = {
            "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
        }

        # include source features
        sfeat_nb = len([feat for feat in self.get_feature_encoders() if feat.startswith("sfeats")])

        for f_number in range(sfeat_nb):
            sfeat = "sfeats." + str(f_number)
            p.input_modality[sfeat] = ("symbol:sfeature",
                                       {"f_number": f_number,
                                        "vocab_size": self.get_feature_encoders()[sfeat].vocab_size})

        target_vocab_size = self._encoders["targets"].vocab_size
        p.target_modality = (registry.Modalities.SYMBOL, target_vocab_size)

        p.sfeat_delimiter = self.sfeat_delimiter
        p.use_subword_tags = self.use_subword_tags
        p.vocab_type = self.vocab_type

        if self.packed_length:
            identity = (registry.Modalities.GENERIC, None)
            if self.has_inputs:
                p.input_modality["inputs_segmentation"] = identity
                p.input_modality["inputs_position"] = identity
            p.input_modality["targets_segmentation"] = identity
            p.input_modality["targets_position"] = identity


class SubwordTextEncoder(text_encoder.SubwordTextEncoder):
    r"""Allow subword encoding with an arbitrary non-native tokenization.
    The delimiter between tokens is a space. Tokenization is thus handled
    with python str.split() method.
    """

    @classmethod
    def build_from_generator(cls,
                             generator,
                             target_vocab_size,
                             max_subtoken_length=None,
                             reserved_tokens=None):
        """Builds a SubwordTextEncoder from the generated text.

        Args:
          generator: yields text.
          target_vocab_size: int, approximate vocabulary size to create.
          max_subtoken_length: Maximum length of a subtoken. If this is not set,
            then the runtime and memory use of creating the vocab is quadratic in
            the length of the longest token. If this is set, then it is instead
            O(max_subtoken_length * length of longest token).
          reserved_tokens: List of reserved tokens. The global variable
            `RESERVED_TOKENS` must be a prefix of `reserved_tokens`. If this
            argument is `None`, it will use `RESERVED_TOKENS`.

        Returns:
          SubwordTextEncoder with `vocab_size` approximately `target_vocab_size`.
        """
        token_counts = defaultdict(int)
        for item in generator:
            for tok in item.split():
                token_counts[tok] += 1
        encoder = cls.build_to_target_size(
            target_vocab_size, token_counts, 1, 1e3,
            max_subtoken_length=max_subtoken_length,
            reserved_tokens=reserved_tokens)
        return encoder

    def decode(self, subtokens):
        return " ".join(self._subtoken_ids_to_tokens(subtokens))



