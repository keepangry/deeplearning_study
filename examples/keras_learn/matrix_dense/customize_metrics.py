
def pos_recall(y_true, y_pred):
    y_pred = K.cast(y_pred >= 0.9, 'float32')
    P = K.sum(y_true)
    TP = K.sum(y_pred * y_true)
    result = TP / P
    return result


def pos_precision(y_true, y_pred):
    # 预测为1
    y_pred = K.cast(y_pred >= 0.9, 'float32')
    P = K.sum(y_pred)
    TP = K.sum(y_pred * y_true)
    result = TP / (P+0.1)
    return result


def train(train_generator, validation_generator):
    model = base_embedding_lstm(max_len=params.time_steps)
    print("model name: %s" % params.model_name)

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy', pos_recall, pos_precision])
    callback = [
        keras.callbacks.EarlyStopping(patience=params.early_stop, monitor='val_loss'),
        # tf.keras.callbacks.ModelCheckpoint(params.model_hdfs_file,
        #                                    save_best_only=True,
        #                                    verbose=1)
        # ModelSaver(path=params.model_hdfs_file),
        # TrainLogWriter(model_name=params.model_name)
    ]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=params.steps_per_epoch,
        epochs=params.epochs,
        validation_data=validation_generator,
        validation_steps=1,
        callbacks=callback,
        verbose=1,
        class_weight={0: 1.0, 1: params.pos_class_weight})

    model.save(params.model_filename)
    tf.gfile.Copy(params.model_filename, params.model_hdfs_file, overwrite=True)
    print("model save finished!")
