import os
from collections import Counter
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
random.seed(1)
import tensorflow as tf
from autoencoder import ContrastiveAE, Autoencoder
from keras import backend as K
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# import hdbscan



def create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)


def create_parent_folder(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))


def split_by_families(X, y, train_families):
    train_mask = np.isin(y, train_families)
    test_mask  = ~train_mask
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    
    return X_train, y_train, X_test, y_test


def adjust_labels(X_train, y_train, X_test, y_test):
    le = LabelEncoder()
    y_train_prime = le.fit_transform(y_train)
    mapping = {}
    for i in range(len(y_train)):
        mapping[y_train[i]] = y_train_prime[i]

    print(f'LabelEncoder mapping: {mapping}')

    y_test_prime = np.zeros(shape=y_test.shape, dtype=np.int32)
    for i in range(len(y_test)):
        if y_test[i] not in y_train:
            y_test_prime[i] = 7
        else:
            y_test_prime[i] = mapping[y_test[i]]

    y_train_prime = np.array(y_train_prime, dtype=np.int32)
    print(f'After relabeling training: {Counter(y_train_prime)}')
    print(f'After relabeling testing: {Counter(y_test_prime)}')

    return X_train, y_train_prime, X_test, y_test_prime


def get_model_dims(model_name, input_layer_num, hidden_layer_num, output_layer_num):
    if '-' not in hidden_layer_num:
        dims = [input_layer_num, int(hidden_layer_num), output_layer_num]
    else:
        hidden_layers = [int(dim) for dim in hidden_layer_num.split('-')]
        dims = [input_layer_num]
        for dim in hidden_layers:
            dims.append(dim)
        dims.append(output_layer_num)

    print(f'{model_name} dims: {dims}')
    return dims


def get_latent_representation_keras(dims, best_weights_file, X_train, X_test):
    K.clear_session()
    ae = Autoencoder(dims)
    ae_model, encoder_model = ae.build()
    encoder_model.load_weights(best_weights_file, by_name=True)

    z_train = encoder_model.predict(X_train)
    z_test = encoder_model.predict(X_test)

    print(f'z_train shape: {z_train.shape}')
    print(f'z_test shape: {z_test.shape}')
    print(f'z_train[0]: {z_train[0]}')

    return z_train, z_test


def get_latent_data_for_each_family(z_train, y_train):
    N = len(np.unique(y_train))
    N_family = [len(np.where(y_train == family)[0]) for family in range(N)]
    z_family = []
    for family in range(N):
        z_tmp = z_train[np.where(y_train == family)[0]]
        z_family.append(z_tmp)

    z_len = [len(z_family[i]) for i in range(N)]
    print(f'z_family length: {z_len}')

    return N, N_family, z_family


def get_latent_distance_between_sample_and_centroid(z_family, centroids, margin, N, N_family):
    dis_family = []  # two-dimension list

    for i in range(N): # i: family index
        dis = [np.linalg.norm(z_family[i][j] - centroids[i]) for j in range(N_family[i])]
        dis_family.append(dis)

    dis_len = [len(dis_family[i]) for i in range(N)]
    print(f'dis_family length: {dis_len}')

    return dis_family


def get_MAD_for_each_family(dis_family, N, N_family):
    mad_family = []
    for i in range(N):
        median = np.median(dis_family[i])
        print(f'family {i} median: {median}')
        diff_list = [np.abs(dis_family[i][j] - median) for j in range(N_family[i])]
        mad = 1.4826 * np.median(diff_list)  # 1.4826: assuming the underlying distribution is Gaussian
        mad_family.append(mad)
    print(f'mad_family: {mad_family}')

    return mad_family


def _cluster_latents(z_drift, min_cluster_size=8, min_samples=None):
    if z_drift.shape[0] < 5:
        eps = 0.5
    else:
        sample = z_drift if z_drift.shape[0] <= 200 else z_drift[np.random.choice(z_drift.shape[0], 200, replace=False)]
        dists = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)
        med = np.median(dists[np.triu_indices_from(dists, k=1)])
        eps = float(max(med * 0.5, 1e-4))
    db = DBSCAN(eps=eps, min_samples=min_samples if min_samples is not None else max(3, int(min_cluster_size/2)))
    labels = db.fit(z_drift)
    return labels
    

def detect_drift_samples(X_train, y_train, X_test, y_test,
                       dims, margin, mad_threshold, best_weights_file,
                       all_detect_path, simple_detect_path,
                       training_info_for_detect_path):
    if os.path.exists(all_detect_path) and os.path.exists(simple_detect_path):
        os.remove(all_detect_path)
        os.remove(simple_detect_path)
        os.remove(all_detect_path.replace(".csv", "_clusters.csv"))


    '''get latent data for the entire training and testing set'''
    z_train, z_test = get_latent_representation_keras(dims, best_weights_file, X_train, X_test)

    '''get latent data for each family in the training set'''
    N, N_family, z_family = get_latent_data_for_each_family(z_train, y_train)

    '''get centroid for each family in the latent space'''
    centroids = [np.mean(z_family[i], axis=0) for i in range(N)]
    # centroids = [np.median(z_family[i], axis=0) for i in range(N)]
    print(f'centroids: {centroids}')

    '''get distance between each training sample and their family's centroid in the latent space '''
    dis_family = get_latent_distance_between_sample_and_centroid(z_family, centroids,
                                                                    margin,
                                                                    N, N_family)

    '''get the MAD for each family'''
    mad_family = get_MAD_for_each_family(dis_family, N, N_family)

    np.savez_compressed(training_info_for_detect_path,
                        z_train=z_train,
                        z_family=z_family,
                        centroids=centroids,
                        dis_family=dis_family,
                        mad_family=mad_family)

    drifted_samples = []
    drifted_labels = []

    '''detect drifting in the testing set'''
    with open(all_detect_path, 'w') as f1:
        f1.write('sample_idx,is_drift,closest_family,real_label,min_distance,min_anomaly_score\n')
        with open(simple_detect_path, 'w') as f2:
            f2.write('sample_idx,closest_family,real_label,min_distance,min_anomaly_score\n')

            for k in tqdm(range(len(X_test)), desc='detect', total=X_test.shape[0]):
                z_k = z_test[k]
                '''get distance between each testing sample and each centroid'''
                dis_k = [np.linalg.norm(z_k - centroids[i]) for i in range(N)]
                anomaly_k = [np.abs(dis_k[i] - np.median(dis_family[i])) / mad_family[i] for i in range(N)]
                print(f'sample-{k} - dis_k: {dis_k}')
                print(f'sample-{k} - anomaly_k: {anomaly_k}')

                closest_family = np.argmin(dis_k)
                min_dis = np.min(dis_k)
                min_anomaly_score = np.min(anomaly_k)

                if min_anomaly_score > mad_threshold:
                    print(f'testing sample {k} is drifting')
                    f1.write(f'{k},Y,{closest_family},{y_test[k]},{min_dis},{min_anomaly_score}\n')
                    f2.write(f'{k},{closest_family},{y_test[k]},{min_dis},{min_anomaly_score}\n')

                    drifted_samples.append(z_k)
                    drifted_labels.append(y_test[k])
                else:
                    f1.write(f'{k},N,{closest_family},{y_test[k]},{min_dis},{min_anomaly_score}\n')

    if len(drifted_samples) > 0:
        drifted_samples = np.array(drifted_samples)
        clustering =_cluster_latents(drifted_samples)
        # clustering = DBSCAN(eps=1.5, min_samples=5).fit(drifted_samples)
        drifted_cluster_labels = clustering.labels_
        print("New candidate families:", set(drifted_cluster_labels))

        with open(all_detect_path.replace(".csv", "_clusters.csv"), "w") as fc:
            fc.write("sample_idx,real_label,assigned_new_family\n")
            for i, (z, true_label) in enumerate(zip(drifted_samples, drifted_labels)):
                new_family = drifted_cluster_labels[i]
                fc.write(f"{i},{true_label},{new_family},\n")

        ari = adjusted_rand_score(drifted_labels, drifted_cluster_labels)
        nmi = normalized_mutual_info_score(drifted_labels, drifted_cluster_labels)
        print(f"Clustering ARI: {ari:.3f}, NMI: {nmi:.3f}")


def main():
    data = np.load("data/drebin_new_7.npz")
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    unique_labels, counts = np.unique(y, return_counts=True)
    print("Unique labels:", unique_labels)
    print("Counts per label:", dict(zip(unique_labels, counts)))

    train_f = '012345'
    X_train, y_train, X_test, y_test = split_by_families(X, y, train_families=[int(d) for d in train_f]) # test [5,6,7]
    print(f'Before label adjusting: y_train: {Counter(y_train)}\n  y_test: {Counter(y_test)}')

    # X_train, y_train, X_test, y_test = adjust_labels(X_train, y_train, X_test, y_test)
    print(f'Loaded train: {X_train.shape}, {y_train.shape}')
    print(f'Loaded test: {X_test.shape}, {y_test.shape}')
    print(f'y_train labels: {np.unique(y_train)}')
    print(f'y_test labels: {np.unique(y_test)}')
    print(f'y_train: {Counter(y_train)}')
    print(f'y_test: {Counter(y_test)}')

    num_features = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    cae_hidden = '512-128-32'
    cae_lr = 0.0001
    cae_batch_size = 64
    cae_lambda_1 = 1e-1
    cae_epochs = 250
    similar_ratio = 0.25
    margin = 10.0
    display_interval = 10

    cae_dims = get_model_dims('Contrastive AE', num_features, cae_hidden, num_classes)

    optimizer = tf.train.AdamOptimizer

    create_folder('models')
    ae_weights_path = os.path.join('models', f'cae_weights_{train_f}.h5')

    cae = ContrastiveAE(cae_dims, optimizer, cae_lr)
    cae.train(X_train, y_train,
            cae_lambda_1, cae_batch_size, cae_epochs, similar_ratio, margin,
            ae_weights_path, display_interval)
    print('Training contrastive autoencoder finished')

    print('Detect drifting samples in the testing set...')
    mad_threshold  = 3.5

    create_folder('reports-v1')
    create_folder('reports-v1/'+train_f)
    all_detect_path = os.path.join('reports-v1', train_f, 'detect_results_all_.csv')
    simple_detect_path = os.path.join('reports-v1', train_f, 'detect_results_simple.csv')
    training_info_for_all_detect_path = os.path.join('reports-v1', train_f, 'training_info_for_detect.npz')

    detect_drift_samples(X_train, y_train, X_test, y_test,
                                cae_dims, margin, mad_threshold, ae_weights_path,
                                all_detect_path, simple_detect_path,
                                training_info_for_all_detect_path)

main()