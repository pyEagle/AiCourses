# -*- coding:utf-8 -*-

import tensorflow as tf

from tensorflow import keras
from tqdm import tqdm


EPOCHS = 5
BATCH_SIZE = 32
LR = 0.01 
opt = keras.optimizers.Nadam(learning_rate=LR)
loss_func = keras.losses.mean_squared_error

avg_loss = keras.metrics.Mean(name="loss")
mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x, training=True)
        # 有些 loss 算出来维度不对，习惯性加个 reduce_mean 稳一点
        main_loss = tf.reduce_mean(loss_func(y, y_hat))
        # 别忘了 model.losses，不然正则化（L2）就白写了
        total_loss = tf.add_n([main_loss] + model.losses)
    
    grads = tape.gradient(total_loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    
    return total_loss, y_hat

for epoch in range(1, EPOCHS + 1):
    print(f"\n正在跑第 {epoch}/{EPOCHS} 个 Epoch")
    
    pbar = tqdm(range(len(X_train_scaled) // BATCH_SIZE), desc=f"Epoch {epoch}")
    
    for step in pbar:
        x_batch, y_batch = random_batch(X_train_scaled, y_train, BATCH_SIZE)
        
        current_loss, y_pred = train_step(your_model, x_batch, y_batch)
        
        avg_loss(current_loss)
        mae_metric(y_batch, y_pred)
        
        if step % 10 == 0:
            pbar.set_postfix({
                "loss": f"{avg_loss.result():.4f}",
                "mae": f"{mae_metric.result():.4f}"
            })

    avg_loss.reset_states()
    mae_metric.reset_states()

