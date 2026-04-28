# -*- coding:utf-8 -*-

import os
import time

from tensorflow import keras

root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir(root_dir):
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    log_path = os.path.join(root_dir, run_id)
    
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    return log_path

current_run_logdir = get_run_logdir(root_logdir)

tensorboard_cb = keras.callbacks.TensorBoard(current_run_logdir)

history = your_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64, 
    validation_data=(X_valid, y_valid),
    callbacks=[tensorboard_cb],
    verbose=1 # 习惯性显式指定一下日志级别
)

