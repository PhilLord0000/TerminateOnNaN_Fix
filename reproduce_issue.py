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

# Create a temporary directory for BackupAndRestore
backup_dir = tempfile.mkdtemp()
print(f"Backup directory: {backup_dir}")

# Set up callbacks
backup_callback = keras.callbacks.BackupAndRestore(backup_dir=backup_dir)
terminate_callback = keras.callbacks.TerminateOnNaN()

# Train the model (will hit NaN very quickly)
print("\nStarting training...")
try:
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        callbacks=[backup_callback, terminate_callback],
        verbose=1
    )
    print("\nTraining completed normally")
except Exception as e:
    print(f"\nTraining stopped with exception: {e}")

# Check if backup directory still exists
print(f"\nBackup directory exists: {os.path.exists(backup_dir)}")
if os.path.exists(backup_dir):
    print(f"Contents: {os.listdir(backup_dir)}")
else:
    print("Backup directory was deleted - cannot restore from last good epoch!")

# Cleanup
if os.path.exists(backup_dir):
    import shutil
    shutil.rmtree(backup_dir)

# **Expected behaviour:**
# When NaN is detected, the backup directory should remain so you can restore from the last good epoch.

# **Actual behaviour:**
# The backup directory is deleted because `on_train_end()` is called, printing:

# Backup directory exists: False
# Backup directory was deleted - cannot restore from last good epoch!