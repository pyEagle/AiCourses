# -*- coding:utf-8 -*-

from tensorflow import keras


# 1.训练期间使用训练集
checkpoint_cb = keras.callbacks.ModelCheckpoint("./model/mlp_c.h5", 
    save_best_only=True)

# 2.早停训练
checkpoint_cb = keras.callbacks.EarlyStopping(patience=10
    , restore_best_weights=True)

# 3. 在模型训练的时候，将回调函数传给callbacks
history = your_model.fit(X_train,y_train,
    epochs=30,
    validation_data=(X_valid, y_valid),
    callbacks=[checkpoint_cb，checkpoint_cb]
    )
