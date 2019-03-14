import os

from fairseq.data import Dictionary, LanguagePairDataset
from fairseq.tasks import FairseqTask, register_task


@register_task('simple_classification')
class SimpleClassificationTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--max-positions', default=1024, type=int,
                            help='max input length')

    @classmethod
    def setup_task(cls, args, **kwargs):
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just load the Dictionaries.
        import pdb
        pdb.set_trace()
        input_vocab = Dictionary.load(os.path.join(args.data, 'dict.input.txt'))
        label_vocab = Dictionary.load(os.path.join(args.data, 'dict.label.txt'))
        print('| [input] dictionary: {} types'.format(len(input_vocab)))
        print('| [label] dictionary: {} types'.format(len(label_vocab)))

        return SimpleClassificationTask(args, input_vocab, label_vocab)

    def __init__(self, args, input_vocab, label_vocab):
        super().__init__(args)
        self.input_vocab = input_vocab
        self.label_vocab = label_vocab

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        prefix = os.path.join(self.args.data, '{}.input-label'.format(split))

        # Read input sentences.
        sentences, lengths = [], []
        with open(prefix + '.input', encoding='utf-8') as file:
            for line in file:
                sentence = line.strip()

                # Tokenize the sentence, splitting on spaces
                tokens = self.input_vocab.encode_line(
                    sentence, add_if_not_exist=False,
                )

                sentences.append(tokens)
                lengths.append(tokens.numel())

        # Read labels.
        labels = []
        with open(prefix + '.label', encoding='utf-8') as file:
            for line in file:
                label = line.strip()
                labels.append(
                    # Convert label to a numeric ID.
                    torch.LongTensor([self.label_vocab.add_symbol(label)])
                )

        assert len(sentences) == len(labels)
        print('| {} {} {} examples'.format(self.args.data, split, len(sentences)))

        # We reuse LanguagePairDataset since classification can be modeled as a
        # sequence-to-sequence task where the target sequence has length 1.
        self.datasets[split] = LanguagePairDataset(
            src=sentences,
            src_sizes=lengths,
            src_dict=self.input_vocab,
            tgt=labels,
            tgt_sizes=torch.ones(len(labels)),  # targets have length 1
            tgt_dict=self.label_vocab,
            left_pad_source=False,
            max_source_positions=self.args.max_positions,
            max_target_positions=1,
            # Since our target is a single class label, there's no need for
            # input feeding. If we set this to ``True`` then our Model's
            # ``forward()`` method would receive an additional argument called
            # *prev_output_tokens* that would contain a shifted version of the
            # target sequence.
            input_feeding=False,
        )

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and the "target"
        # has max length 1.
        return (self.args.max_positions, 1)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.input_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.label_vocab

    # We could override this method if we wanted more control over how batches
    # are constructed, but it's not necessary for this tutorial since we can
    # reuse the batching provided by LanguagePairDataset.
    #
    # def get_batch_iterator(
    #     self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
    #     ignore_invalid_inputs=False, required_batch_size_multiple=1,
    #     seed=1, num_shards=1, shard_id=0,
    # ):
    #     (...)



import torch
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


from fairseq.models import BaseFairseqModel, register_model

# Note: the register_model "decorator" should immediately precede the
# definition of the Model class.

@register_model('rnn_classifier')
class FairseqRNNClassifier(BaseFairseqModel):

    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add a new command-line argument to configure the
        # dimensionality of the hidden state.
        parser.add_argument(
            '--hidden-dim', type=int, metavar='N',
            help='dimensionality of the hidden state',
        )

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a FairseqRNNClassifier instance.

        # Initialize our RNN module
        rnn = RNN(
            # We'll define the Task in the next section, but for now just
            # notice that the task holds the dictionaries for the "source"
            # (i.e., the input sentence) and "target" (i.e., the label).
            input_size=len(task.source_dictionary),
            hidden_size=args.hidden_dim,
            output_size=len(task.target_dictionary),
        )

        # Return the wrapped version of the module
        return FairseqRNNClassifier(
            rnn=rnn,
            input_vocab=task.source_dictionary,
        )

    def __init__(self, rnn, input_vocab):
        super(FairseqRNNClassifier, self).__init__()

        self.rnn = rnn
        self.input_vocab = input_vocab

        # The RNN module in the tutorial expects one-hot inputs, so we can
        # precompute the identity matrix to help convert from indices to
        # one-hot vectors. We register it as a buffer so that it is moved to
        # the GPU when ``cuda()`` is called.
        self.register_buffer('one_hot_inputs', torch.eye(len(input_vocab)))

    def forward(self, src_tokens, src_lengths):
        # The inputs to the ``forward()`` function are determined by the
        # Task, and in particular the ``'net_input'`` key in each
        # mini-batch. We'll define the Task in the next section, but for
        # now just know that *src_tokens* has shape `(batch, src_len)` and
        # *src_lengths* has shape `(batch)`.
        bsz, max_src_len = src_tokens.size()

        # Initialize the RNN hidden state. Compared to the original PyTorch
        # tutorial we'll also handle batched inputs and work on the GPU.
        hidden = self.rnn.initHidden()
        hidden = hidden.repeat(bsz, 1)  # expand for batched inputs
        hidden = hidden.to(src_tokens.device)  # move to GPU

        for i in range(max_src_len):
            # WARNING: The inputs have padding, so we should mask those
            # elements here so that padding doesn't affect the results.
            # This is left as an exercise for the reader. The padding symbol
            # is given by ``self.input_vocab.pad()`` and the unpadded length
            # of each input is given by *src_lengths*.

            # One-hot encode a batch of input characters.
            input = self.one_hot_inputs[src_tokens[:, i].long()]

            # Feed the input to our RNN.
            output, hidden = self.rnn(input, hidden)

        # Return the final output state for making a prediction
        return output


from fairseq.models import register_model_architecture

# The first argument to ``register_model_architecture()`` should be the name
# of the model we registered above (i.e., 'rnn_classifier'). The function we
# register here should take a single argument *args* and modify it in-place
# to match the desired architecture.

@register_model_architecture('rnn_classifier', 'pytorch_tutorial_rnn')
def pytorch_tutorial_rnn(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.hidden_dim = getattr(args, 'hidden_dim', 128)