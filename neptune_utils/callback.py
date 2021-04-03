from tensorflow.keras.callbacks import Callback

import neptune
neptune.init(project_qualified_name='amdalifuk/cifar')  # add your


class NeptuneCallback(object, Callback):
    def on_epoch_end(self, epoch, logs=None):

        neptune.log_metric('loss', logs['loss'])
        neptune.log_metric('1-accuracy', 1-logs['accuracy'])

        if 'val_loss' in logs:
            neptune.log_metric('val_loss', logs['val_loss'])
            neptune.log_metric('1-val_accuracy', 1-logs['val_accuracy'])
