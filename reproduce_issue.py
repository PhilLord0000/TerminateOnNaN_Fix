import numpy as np
import tensorflow as tf
from tensorflow import keras

# Create a simple dataset
x_train = np.random.randn(1000, 10)
y_train = np.random.randn(1000, 1)

# Create a simple model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])

# Use a normal learning rate initially
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    loss='mse'
)

# Callback to increase learning rate after first epoch
class IncreaseLR(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            print("\nIncreasing learning rate to cause NaN...")
            self.model.optimizer.learning_rate.assign(1e100)

# Set up callbacks
backup_callback = keras.callbacks.BackupAndRestore(backup_dir='./backup')
increase_lr_callback = IncreaseLR()
terminate_callback = keras.callbacks.TerminateOnNaN()

# Train the model
print("Starting training with standard TerminateOnNaN...")
print("Will complete epoch 0, create backup, then hit NaN in epoch 1...")
model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    callbacks=[backup_callback, increase_lr_callback, terminate_callback],
    verbose=1
)

print("\nTraining ended. Check if ./backup directory still exists.")
print("Expected: backup directory is deleted (this is the problem)")