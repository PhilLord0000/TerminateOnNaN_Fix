import numpy as np
import tensorflow as tf
from tensorflow import keras
import tempfile
import os

# Create a simple dataset
x_train = np.random.randn(1000, 10)
y_train = np.random.randn(1000, 1)

# Create a simple model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])

# Use a very high learning rate to intentionally cause NaN
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=1e10),  # Extremely high LR
    loss='mse'
)

# Custom callback that raises an error instead
class HardTerminateOnNaN(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None and (tf.math.is_nan(loss) or tf.math.is_inf(loss)):
            print(f"\nNaN detected at batch {batch}. Terminating immediately.")
            raise RuntimeError("NaN loss encountered.")

# Create new backup directory
backup_dir = tempfile.mkdtemp()
print(f"\n\nTesting with HardTerminateOnNaN:")
print(f"Backup directory: {backup_dir}")

backup_callback = keras.callbacks.BackupAndRestore(backup_dir=backup_dir)
hard_terminate_callback = HardTerminateOnNaN()

# Reset model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(1)
])
model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e10), loss='mse')

print("\nStarting training with HardTerminateOnNaN...")
try:
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        callbacks=[backup_callback, hard_terminate_callback],
        verbose=1
    )
except Exception as e:
    print(f"\nTraining stopped with exception: {e}")

# Check if backup directory still exists
print(f"\nBackup directory exists: {os.path.exists(backup_dir)}")
if os.path.exists(backup_dir):
    print(f"Contents: {os.listdir(backup_dir)}")
    print("Backup preserved - can restore from last good epoch!")

# Cleanup
if os.path.exists(backup_dir):
    import shutil
    shutil.rmtree(backup_dir)