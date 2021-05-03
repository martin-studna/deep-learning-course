#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

# 2f67b427-a885-11e7-a937-00505601122b
# c751264b-78ee-11eb-a1a9-005056ad4f31

use_neptune = True
if use_neptune:
    import neptune
    neptune.init(project_qualified_name='amdalifuk/tagger')

import numpy as np
import tensorflow as tf
from tensorflow import keras

import tensorflow_addons as tfa
from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=int(512), type=int, help="Batch size.") #256 just a sip better
parser.add_argument("--learning_rate", default=0.01, type=int, help="Batch size.") #0.2much, 0.01 same #0.1 bad 0.05 great
parser.add_argument("--epochs", default=14, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=12, type=int, help="Maximum number of threads to use.")

# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--cle_dim", default=12, type=int,  #12 good,   6bad, 8ok, 16ok
                    help="CLE embedding dimension.")
parser.add_argument("--max_sentences", default=100000, type=int, #8*2
                    help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--cosine_decay", default=True, help="use cosine_decay.")
parser.add_argument("--rnn_cell", default="GRU", #LSTM better
                    type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=128, #32 low, 128 better
                    type=int, help="RNN cell dimension.")
parser.add_argument("--char_rnn_cell_dim", default=128, 
                    type=int, help="RNN cell dimension.")
parser.add_argument("--we_dim", default=320, type=int, #64 too low, 256 little slower
                    help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.15, type=float, #0.2 was much, 0 low
                    help="Mask words with the given probability.")
parser.add_argument("--char_masking", default=0.08, type=float, #0.2 was much, 0 low
                    help="Mask chars with the given probability.")
                    
parser.add_argument("--concatenate", default='both', help="which to concatate: both, words, chars")

parser.add_argument("--clip_gradient", default=0.1,
                    type=float, help="Norm for gradient clipping.")
parser.add_argument("--recurrent_dropout", default=0.1,
                    type=float, help="recurrent_dropout.")

class Network(tf.keras.Model):
    def __init__(self, args, train):
        # Implement a one-layer RNN network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(
            shape=[None], dtype=tf.string, ragged=True, name='a')

        # TODO(tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.

        word_predictions = train.forms.word_mapping(words)  ##FORMS
        word_predictions = tf.cast(word_predictions, tf.float32)


        # TODO: With a probability of `args.word_masking`, replace the input word by an
        # unknown word (which has index 0).
        #
        # Use a `tf.keras.layers.Dropout` to achieve this, even if it is a bit
        # hacky, because Dropout cannot process integral inputs. One way is to
        # use `tf.ones_like` to create a ragged tensor of float32 ones with the same
        # shape as `hidden`, pass them through a dropout layer with `args.word_masking`
        # rate, and finally set the input word ids to 0 where the result of dropout is zero.

        ones = tf.ones_like(word_predictions, dtype=tf.float32)
        dropout_outputs = tf.keras.layers.Dropout(rate=args.word_masking)(ones)
        word_predictions = tf.where(dropout_outputs == 0, tf.constant(
            0, dtype=tf.float32), word_predictions)


        # TODO(tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocab_size()` call returning the number of unique words in the mapping.

        word_predictions = tf.keras.layers.Embedding( train.forms.word_mapping.vocab_size(), args.we_dim)(word_predictions) ##FORMS
        #word wocab size 20037

        # TODO: Flatten a list of input words using `words.values` and pass
        # the flattened list through `tf.unique`, obtaining a list of
        # unique words and indices of the original flattened words in the
        # unique word list.

        flattened_words = tf.reshape(words.values, [-1])
        unique_words, unique_word_idx = tf.unique(
            flattened_words)


        # TODO: Create sequences of letters by passing the unique words through
        # `tf.strings.unicode_split` call; use "UTF-8" as `input_encoding`.

        char_predictions = tf.strings.unicode_split(unique_words, "UTF-8")
        # TODO: Map the letters into ids by using `char_mapping` of `train.forms`.

        char_predictions = train.forms.char_mapping(char_predictions) ##FORMS
        
        char_predictions = tf.cast(char_predictions, tf.float32)
        ones = tf.ones_like(char_predictions, dtype=tf.float32)
        dropout_outputs = tf.keras.layers.Dropout(rate=args.char_masking)(ones)
        char_predictions = tf.where(dropout_outputs == 0, tf.constant(
            0, dtype=tf.float32), char_predictions)
        # TODO: Embed the input characters with dimensionality `args.cle_dim`.

        char_predictions = tf.keras.layers.Embedding(
            train.forms.char_mapping.vocab_size(), args.cle_dim)(char_predictions) ##FORMS   #######char_mapping
        #char vocab 119

        # TODO: Pass the embedded letters through a bidirectional GRU layer
        # with dimensionality `args.cle_dim`, obtaining representations of the
        # whole words, **concatenating** the outputs of the forward and backward RNNs.

        rnn = tf.keras.layers.GRU(args.char_rnn_cell_dim)#LSTM nepomue tady
        char_predictions = tf.keras.layers.Bidirectional(
            rnn, merge_mode='concat')(char_predictions)
        # TODO: Use `tf.gather` with the indices generated by `tf.unique` to create
        # representation of the flattened (non-unique) words.

        char_predictions = tf.gather(char_predictions, unique_word_idx)

        # TODO: Then, convert the flattened list into a RaggedTensor of the same shape
        # as `words` using `words.with_values` call.

        char_predictions = word_predictions.with_values(
            char_predictions)

        # TODO: Concatenate the word-level embeddings and the computed character-level WEs
        # (in this order).
        if args.concatenate == 'both':
            predictions = tf.keras.layers.Concatenate()(
                [word_predictions, char_predictions])
        elif args.concatenate == 'words':
            predictions = tf.keras.layers.Concatenate()(
                [word_predictions])
        elif args.concatenate == 'chars':
            predictions = tf.keras.layers.Concatenate()(
                [char_predictions])


        # TODO(tagger_we): Create the specified `args.rnn_cell` RNN cell (LSTM, GRU) with
        # dimension `args.rnn_cell_dim`. The cell should produce an output for every
        # sequence element. Then apply it in a bidirectional way on
        # the word representations, **summing** the outputs of forward and backward RNNs.
        
        rnn = None
        if args.rnn_cell == 'LSTM':
            rnn = tf.keras.layers.LSTM(
                args.rnn_cell_dim, return_sequences=True, recurrent_dropout=args.recurrent_dropout)
        elif args.rnn_cell == 'GRU':
            rnn = tf.keras.layers.GRU(args.rnn_cell_dim, return_sequences=True,recurrent_dropout=args.recurrent_dropout)


        predictions = tf.keras.layers.Bidirectional(
            rnn, merge_mode='sum')(predictions)


        # TODO(tagge_we): Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train tags`. However, because we are applying the
        # the Dense layer to a ragged tensor, we need to wrap the Dense layer in
        # a tf.keras.layers.TimeDistributed.

        output_layer = keras.layers.LayerNormalization()
        predictions = tf.keras.layers.TimeDistributed(
            output_layer)(predictions)

        output_layer = tf.keras.layers.Dense(
            train.tags.word_mapping.vocab_size(), activation='softmax')
        predictions = tf.keras.layers.TimeDistributed(
            output_layer)(predictions)

        super().__init__(inputs=words, outputs=predictions)
        lr = args.learning_rate
        print(f"train.size {train.size}")
        if args.cosine_decay:
            decay_steps = args.epochs * train.size / args.batch_size
            lr = keras.experimental.CosineDecay(args.learning_rate, decay_steps, alpha=args.learning_rate/100)

        self.compile(optimizer=tf.optimizers.Adam(lr, global_clipnorm=args.clip_gradient),
                     loss=tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),
                     #loss=keras.losses.CategoricalCrossentropy(),
                     metrics=[tf.metrics.CategoricalAccuracy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(
            args.logdir, update_freq=100, profile_batch=0)
        # A hack allowing to keep the writers open.
        self.tb_callback._close_writers = lambda: None

    # Note that in TF 2.4, computing losses and metrics on RaggedTensors is not yet
    # supported (it will be in TF 2.5). Therefore, we override the `train_step` method
    # to support it, passing the "flattened" predictions and gold data to the loss
    # and metrics.
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y.values, y_pred.values, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y.values, y_pred.values)
        return {m.name: m.result() for m in self.metrics}

    # Analogously to `train_step`, we also need to override `test_step`.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compiled_loss(
            y.values, y_pred.values, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y.values, y_pred.values)
        return {m.name: m.result() for m in self.metrics}



def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt", args.max_sentences)
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    #import pickle
    #with open('mymorpho', 'wb') as m_file:   
    #    pickle.dump(morpho, m_file)

    # Create the network and train
    network = Network(args, morpho.train)

    #tf.data.experimental.enable_debug_mode()
    #tf.config.run_functions_eagerly(True)

    # TODO(tagger_we): Construct dataset for training, which should contain pairs of
    # - ragged tensor of string words (forms) as input
    # - ragged tensor of integral tag ids as targets.
    # To create the identifiers, use the `word_mapping` of `morpho.train.tags`.
    def tagging_dataset(forms, lemmas, tags):
        tags = morpho.train.tags.word_mapping(tags)
        tags = tf.cast(tags, tf.int32)
        num_classes = morpho.train.tags.word_mapping.vocab_size()
        tags = tf.one_hot(tags, num_classes)
        #tags = keras.utils.to_categorical( tags  , morpho.train.tags.word_mapping.vocab_size() )
        #keras.utils.to_categorical( morpho.train.tags.word_mapping( list(morpho.train.dataset.take(1))[0][2] ) , morpho.train.tags.word_mapping.vocab_size() )
        return forms, tags

    train = morpho.train.dataset.map(tagging_dataset).apply(
        tf.data.experimental.dense_to_ragged_batch(args.batch_size)).shuffle(2048*4).prefetch(tf.data.AUTOTUNE)
    dev = morpho.dev.dataset.map(tagging_dataset).apply(
        tf.data.experimental.dense_to_ragged_batch(args.batch_size)).prefetch(tf.data.AUTOTUNE)
    test = morpho.test.dataset.map(tagging_dataset).apply(
        tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        
    callbacks=[network.tb_callback]

    if use_neptune:       
        neptune.create_experiment(params={
            'batch_size': args.batch_size,
            'cle_dim': args.cle_dim,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'max_sentences': args.max_sentences,
            'rnn_cell': args.rnn_cell,
            'rnn_cell_dim': args.rnn_cell_dim,
            'seed': args.seed,
            'threads': args.threads,
            'we_dim': args.we_dim,
            'word_masking': args.word_masking,
            'cosine_decay': args.cosine_decay,
            'char_rnn_cell_dim': args.char_rnn_cell_dim,
            'concatenate': args.concatenate,
            'clip_gradient': args.clip_gradient,
            'recurrent_dropout': args.recurrent_dropout,
        },abort_callback=lambda: neptune.stop() )
        neptune.send_artifact('tagger_competition.py')

        from tensorflow.keras.callbacks import Callback

        class NeptuneCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                print(self.model.optimizer._decayed_lr(tf.float32) )
                neptune.log_metric('loss', logs['loss'])
                neptune.log_metric('1-accuracy', 1-logs['accuracy'])

                if 'val_loss' in logs:
                    neptune.log_metric('val_loss', logs['val_loss'])
                    neptune.log_metric('1-val_accuracy', 1-logs['val_accuracy'])
        callbacks.append(NeptuneCallback())

    network.summary()
    keras.utils.plot_model(network, show_shapes=True)

    network.fit(train, epochs=args.epochs, validation_data=dev, callbacks=callbacks)

    test_logs = network.evaluate(dev, return_dict=True)
    network.tb_callback.on_epoch_end(args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})
    #network.save(f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S") }_{test_logs["accuracy"]:.2f}.h5')
    # Generate test set annotations, but in args.logdir to allow parallel execution.
    #os.makedirs(args.logdir, exist_ok=True)
    with open("tagger_competition.txt", "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set; update the following prediction
        # command if you use other output structre than in tagger_we.
        predictions = network.predict(test, batch_size=args.batch_size)
        tag_strings = morpho.test.tags.word_mapping.get_vocabulary()
        for sentence in predictions:
            for word in sentence:
                print(tag_strings[np.argmax(word)], file=predictions_file)
            print(file=predictions_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
