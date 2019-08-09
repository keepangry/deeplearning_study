
def read_batch_generator_old(path, feature_dim, time_steps, label_dim, batch_size):
    files = tf.gfile.ListDirectory(path)
    while True:
        labels = []
        user_seq_features = []

        for file in files:
            file_path = os.path.join(path, file)
            with tf.gfile.GFile(file_path) as fr:
                for line in fr:
                    label, user_seq_feature = line.strip().split('\t')
                    user_seq_features.append(user_seq_feature)
                    labels.append(float(label))

                    if len(labels) == batch_size:
                        for i in range(len(user_seq_features)):
                            user_seq_feature = user_seq_features[i]
                            user_seq_feature = np.array(user_seq_feature.split(' ')).astype(params.float_type)
                            user_seq_feature = user_seq_feature.reshape(int(user_seq_feature.shape[0]/feature_dim), feature_dim)
                            # TODO：这个pad_sequences有问题，以后面的维度作为步长
                            seq_feature = pad_sequences(user_seq_feature.T, maxlen=time_steps, dtype=params.float_type).T
                            user_seq_features[i] = seq_feature
                        #
                        user_seq_features = np.array(user_seq_features)
                        emb_input = [user_seq_features[..., i] for i in range(12)]
                        emb_input.append(user_seq_features[..., 12:14])

                        yield emb_input, np.array(labels)
                        user_seq_features, labels = [], []



def read_batch_generator(path, feature_dim, time_steps, label_dim, batch_size):
    files = tf.gfile.ListDirectory(path)
    while True:
        labels = []
        user_seq_features = []

        for file in files:
            file_path = os.path.join(path, file)
            with tf.gfile.GFile(file_path) as fr:
                for line in fr:
                    label, user_seq_feature = line.strip().split('\t')
                    user_seq_features.append(user_seq_feature)
                    labels.append(float(label))

                    if len(labels) == batch_size:
                        for i in range(len(user_seq_features)):
                            user_seq_feature = user_seq_features[i]
                            user_seq_feature = np.array(user_seq_feature.split(' ')).astype(params.float_type)
                            user_seq_feature = user_seq_feature.reshape(int(user_seq_feature.shape[0]/feature_dim), feature_dim)
                            # TODO：这个pad_sequences有问题，以后面的维度作为步长
                            seq_feature = pad_sequences(user_seq_feature.T, maxlen=time_steps, dtype=params.float_type).T
                            user_seq_features[i] = seq_feature
                        #
                        user_seq_features = np.array(user_seq_features)

                        yield user_seq_features, np.array(labels)
                        user_seq_features, labels = [], []
