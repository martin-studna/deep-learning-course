import neptune
neptune.init(project_qualified_name='amdalifuk/c10', api_token='TVUJ TOKEN') # add your 


PARAMS = {
              'learning_rate': 420,
             
         }
neptune.create_experiment(params=PARAMS)
neptune.send_artifact('neptune_example.py')


from tensorflow.keras.callbacks import Callback
class CustomCallback(Callback):        
    def on_epoch_end(self, epoch, logs=None):

        neptune.log_metric('loss', logs['loss'])
        if 'val_loss' in logs:
            neptune.log_metric('val_loss', logs['val_loss'])

model.fit(x,y, epochs=500, callbacks=[CustomCallback()] )
        



