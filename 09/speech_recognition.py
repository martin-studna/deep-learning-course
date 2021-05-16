#!/usr/bin/env python3
from tensorflow.keras.callbacks import Callback
from common_voice_cs import CommonVoiceCs
import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
import re
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20,
                    type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=16, type=int,
                    help="Maximum number of threads to use.")
parser.add_argument("--rnn_cell_dim", default=32,
                    type=int, help="rnn_cell_dim")
parser.add_argument("--ctc_beam", default=12,
                    type=int, help="ctc beam")
parser.add_argument("--dropout", default=0.003, type=float,
                    help="Dropout regularization.")
parser.add_argument("--l2", default=0.002, type=float,
                    help="L2 regularization.")
parser.add_argument("--learning_rate", default=0.002,
                    type=float, help="ctc beam")
parser.add_argument("--clip_gradient", default=0.1,
                    type=float, help="Norm for gradient clipping.")
parser.add_argument(
    "--hidden_layers", default=[100], nargs="*", type=int, help="Hidden layer sizes.")

use_neptune = True
if use_neptune:
    import neptune
    neptune.init(project_qualified_name='martin.studna/speech-recognition')


class NeptuneCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(self.model.optimizer._decayed_lr(tf.float32))
        neptune.log_metric('loss', logs['loss'])
        #neptune.log_metric('1-accuracy', 1-logs['accuracy'])

        if 'val_edit_distance' in logs:
            neptune.log_metric('val_edit_distance', logs['val_edit_distance'])
            #neptune.log_metric('1-val_accuracy', 1-logs['val_accuracy'])


class Network(tf.keras.Model):
    def __init__(self, args):
        self._ctc_beam = args.ctc_beam

        inputs = tf.keras.layers.Input(
            shape=[None, CommonVoiceCs.MFCC_DIM], dtype=tf.float32, ragged=True)

        # TODO: Create a suitable model. You should:
        # - use a bidirectional RNN layer(s) to contextualize the input sequences.
        #
        #   Beware that currently when LSTM/GRU process RaggedTensors, they do not use
        #   cuDNN implementation on GPUs, so they are unnecessary slow. As a workaround,
        #   you can convert them to/from dense tensor before/after the rnn call
        #   as described at https://ufal.mff.cuni.cz/courses/npfl114/2021-summer#faq_gpu
        #
        # - optionally use suitable regularization
        #
        # - and finally generate logits for CRC loss/prediction as RaggedTensors.
        #   The logits should be generated by a dense layer with `1 + len(CommonVoiceCs.LETTERS)`
        #   outputs (the plus one is for the CTC blank symbol). Note that no
        #   activation should be used (the CTC operations will take care of it).
        #   Do not forget to use `tf.keras.layers.TimeDistributed`.
        rnn = tf.keras.layers.LSTM(args.rnn_cell_dim, return_sequences=True)
        predictions = tf.keras.layers.Bidirectional(
            rnn, merge_mode='sum')(inputs.to_tensor())
        predictions = tf.RaggedTensor.from_tensor(
            predictions, inputs.row_lengths())

        predictions = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dropout(rate=args.dropout))(predictions)

        for hidden_layer_neurons_count in args.hidden_layers:
            hidden_layer = tf.keras.layers.Dense(
                hidden_layer_neurons_count)
            predictions = tf.keras.layers.TimeDistributed(
                hidden_layer)(predictions)
        # predictions = tf.keras.layers.TimeDistributed(
        #     tf.keras.layers.BatchNormalization())(predictions)
        # predictions = tf.keras.layers.TimeDistributed(
        #     tf.keras.layers.Activation('relu'))(predictions)
        # predictions = tf.keras.layers.TimeDistributed(
        #     tf.keras.layers.Dropout(rate=args.dropout))(predictions)

        output_layer = tf.keras.layers.Dense(
            1 + len(CommonVoiceCs.LETTERS), kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=args.l2))
        predictions = tf.keras.layers.TimeDistributed(
            output_layer)(predictions)
        logits = predictions

        super().__init__(inputs=inputs, outputs=logits)

        # We compile the model without loss, because `train_step` will directly call
        # the `selt.ctc_loss` method.
        self.compile(optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate, global_clipnorm=args.clip_gradient),
                     metrics=[CommonVoiceCs.EditDistanceMetric()])

        self.tb_callback = tf.keras.callbacks.TensorBoard(
            args.logdir, update_freq=100, profile_batch=0)
        # A hack allowing to keep the writers open.
        self.tb_callback._close_writers = lambda: None

    def ctc_loss(self, gold_labels, logits):
        assert isinstance(
            gold_labels, tf.RaggedTensor), "Gold labels given to CTC loss must be RaggedTensors"
        assert isinstance(
            logits, tf.RaggedTensor), "Logits given to CTC loss must be RaggedTensors"

        # TODO: Use tf.nn.ctc_loss to compute the CTC loss.
        # - Convert the gold_labels to SparseTensor and pass `None` as `label_length`.
        # - Convert `logits` to a dense Tensor and then either transpose the
        #   logits to `[max_audio_length, batch, dim]` or set `logits_time_major=False`
        # - Use `logits.row_lengths()` method to obtain the `logit_length`
        # - Use the last class (the one with the highest index) as the `blank_index`.
        #
        # The `tc.nn.ctc_loss` returns a value for a single batch example, so average
        # them to produce a single value and return it.
        single_batch_result = tf.nn.ctc_loss(tf.cast(gold_labels.to_sparse(), dtype=tf.int32), tf.cast(logits.to_tensor(
        ), dtype=tf.float32), None, tf.cast(logits.row_lengths(), dtype=tf.int32), logits_time_major=False, blank_index=len(CommonVoiceCs.LETTERS))

        return tf.reduce_mean(single_batch_result)

    def ctc_decode(self, logits):
        assert isinstance(
            logits, tf.RaggedTensor), "Logits given to CTC predict must be RaggedTensors"

        # TODO: Run `tf.nn.ctc_greedy_decoder` or `tf.nn.ctc_beam_search_decoder`
        # to perform prediction.
        # - Convert the `logits` to a dense Tensor and then transpose them
        #   to shape `[max_audio_length, batch, dim]` using `tf.transpose`
        # - Use `logits.row_lengths()` method to obtain the `sequence_length`
        # - Convert the result of the decoded from a SparseTensor to a RaggedTensor
        beams, _ = tf.nn.ctc_beam_search_decoder(tf.transpose(
            logits.to_tensor(), [1, 0, 2]), tf.cast(logits.row_lengths(), dtype=tf.int32))

        predictions = tf.RaggedTensor.from_sparse(beams[0])

        assert isinstance(
            predictions, tf.RaggedTensor), "CTC predictions must be RaggedTensors"
        return predictions

    # We override the `train_step` method, because:
    # - computing losses on RaggedTensors is not supported in TF 2.4
    # - we do not want to evaluate the training data, because CTC decoding is slow
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.ctc_loss(y, y_pred)
            if self.losses:  # Add regularization losses if present
                loss += tf.math.add_n(self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": loss}

    # We override `predict_step` to run CTC decoding during prediction
    def predict_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        y_pred = self(data, training=False)
        y_pred = self.ctc_decode(y_pred)
        return y_pred

    # We override `test_step` to use `predict_step` to obtain CTC predictions.
    def test_step(self, data):
        x, y = data
        y_pred = self.predict_step(data)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if use_neptune:
        neptune.create_experiment(params={
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'rnn_cell_dim': args.rnn_cell_dim,
            'seed': args.seed,
            'threads': args.threads,
            'dropout': args.dropout,
            'learning_rate': args.learning_rate,
            'clip_gradient': args.clip_gradient,
            'hidden_layers': args.hidden_layers
        }, abort_callback=lambda: neptune.stop())
        neptune.send_artifact('speech_recognition.py')

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data. Using analyses is only optional.
    cvcs = CommonVoiceCs()

    # Create input data pipeline.
    def create_dataset(name):
        def prepare_example(example):
            # TODO: Create suitable batch examples.
            # - example["mfccs"] should be used as input
            # - the example["sentence"] is a UTF-8-encoded string with the target sentence
            #   - split it to unicode characters by using `tf.strings.unicode_split`
            #   - then pass it through the `cvcs.letters_mapping` layer to map
            #     the unicode characters to ids
            return example["mfccs"], cvcs.letters_mapping(tf.strings.unicode_split(example["sentence"], 'UTF-8'))

        dataset = getattr(cvcs, name).map(prepare_example)
        dataset = dataset.shuffle(
            len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    train, dev, test = create_dataset(
        "train"), create_dataset("dev"), create_dataset("test")

    # TODO: Create the model and train it
    model = Network(args)

    model.fit(train, epochs=args.epochs, validation_data=dev,
              callbacks=[NeptuneCallback()])

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "speech_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the CommonVoice sentences.
        predictions = model.predict(test, batch_size=args.batch_size)

        for sentence in predictions:
            print("".join(CommonVoiceCs.LETTERS[char]
                  for char in sentence), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
