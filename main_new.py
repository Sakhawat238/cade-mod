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
import warnings
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# Try HDBSCAN first (recommended). If missing, we fall back to sklearn.DBSCAN.
# try:
#     import hdbscan
#     HAS_HDBSCAN = True
# except Exception:
#     HAS_HDBSCAN = False
HAS_HDBSCAN = False

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


# ----------------------------
# Multiclass drift detection
# ----------------------------
def _cluster_latents(z_drift, min_cluster_size=10, min_samples=None):

    if z_drift.shape[0] == 0:
        return np.array([], dtype=int)

    if HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    prediction_data=False)
        labels = clusterer.fit_predict(z_drift)
        return labels
    else:
        # DBSCAN: eps heuristic based on sample pairwise median distance
        if z_drift.shape[0] < 5:
            eps = 0.5
        else:
            sample = z_drift if z_drift.shape[0] <= 200 else z_drift[np.random.choice(z_drift.shape[0], 200, replace=False)]
            dists = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)
            med = np.median(dists[np.triu_indices_from(dists, k=1)])
            eps = float(max(med * 0.5, 1e-4))
        db = DBSCAN(eps=eps, min_samples=min_samples if min_samples is not None else max(3, int(min_cluster_size/2)))
        labels = db.fit_predict(z_drift)
        return labels


def _classify_cluster(cluster_centroid, class_centroids, mad_family, mad_multiplier_known=3.0, mixed_ratio=1.25):
    """
    Decide whether a cluster centroid corresponds to:
      - known-class drift: closest class distance < mad_multiplier_known * class_mad
      - novel-class drift: far from all classes
      - mixed/boundary drift: close to multiple classes (top2 distances within mixed_ratio)
    Returns (type_str, closest_class, distances_sorted)
    """
    dists = [np.linalg.norm(cluster_centroid - c) for c in class_centroids]
    order = np.argsort(dists)
    closest = int(order[0])
    closest_dist = float(dists[closest])
    class_mad = mad_family[closest] if closest < len(mad_family) and mad_family[closest] > 0 else 1e-6
    if closest_dist <= mad_multiplier_known * class_mad:
        return "known_class_drift", closest, dists
    if len(order) >= 2:
        if dists[order[0]] * mixed_ratio >= dists[order[1]]:
            return "mixed_boundary_drift", closest, dists
    return "novel_class_drift", closest, dists


def top_k_feature_deltas(X_cluster, X_train_class, k=10):
    """
    Simple explanation: compute mean input difference between cluster and class training samples,
    return top-k feature indices and delta values (cluster_mean - class_mean).
    """
    if X_cluster.shape[0] == 0 or X_train_class.shape[0] == 0:
        return []
    cluster_mean = np.mean(X_cluster, axis=0)
    class_mean = np.mean(X_train_class, axis=0)
    delta = cluster_mean - class_mean
    top_idxs = np.argsort(np.abs(delta))[::-1][:k]
    return [(int(idx), float(delta[idx])) for idx in top_idxs]


def detect_and_cluster_multiclass(
        X_train, y_train, X_test, y_test,
        dims, margin, mad_threshold, best_weights_file,
        out_folder='reports',
        min_cluster_size=10,
        explain_top_k=10,
        save_training_info=True):
    """
    Full pipeline: compute latent representations, detect drifted test samples using MAD gate,
    cluster drifts, label clusters, produce CSV report and per-cluster explanation.
    Returns a dict with summary and detailed cluster info.
    """

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # 1) Latents
    z_train, z_test = get_latent_representation_keras(dims, best_weights_file, X_train, X_test)
    N, N_family, z_family = get_latent_data_for_each_family(z_train, y_train)
    class_centroids = [np.mean(z_family[i], axis=0) for i in range(N)]
    dis_family = get_latent_distance_between_sample_and_centroid(z_family, class_centroids, margin, N, N_family)
    mad_family = get_MAD_for_each_family(dis_family, N, N_family)

    if save_training_info:
        training_info_for_detect_path = os.path.join(out_folder, 'training_info_for_detect.npz')
        np.savez_compressed(training_info_for_detect_path,
                            z_train=z_train,
                            z_family=z_family,
                            centroids=class_centroids,
                            dis_family=dis_family,
                            mad_family=mad_family)

    # 2) Gate test samples
    is_drift = np.zeros(len(X_test), dtype=bool)
    closest_family_arr = np.zeros(len(X_test), dtype=int)
    min_dist_arr = np.zeros(len(X_test), dtype=float)
    min_anomaly_score_arr = np.zeros(len(X_test), dtype=float)

    for k in range(len(X_test)):
        z_k = z_test[k]
        dis_k = [np.linalg.norm(z_k - class_centroids[i]) for i in range(N)]
        anomaly_k = [np.abs(dis_k[i] - np.median(dis_family[i])) / (mad_family[i] if mad_family[i] > 0 else 1e-6) for i in range(N)]

        closest_family = int(np.argmin(dis_k))
        min_dis = float(np.min(dis_k))
        min_anom = float(np.min(anomaly_k))

        closest_family_arr[k] = closest_family
        min_dist_arr[k] = min_dis
        min_anomaly_score_arr[k] = min_anom

        if min_anom > mad_threshold:
            is_drift[k] = True

    # Save simple detection file
    simple_detect_path = os.path.join(out_folder, 'detect_results_simple.csv')
    with open(simple_detect_path, 'w') as f:
        f.write('sample_idx,closest_family,real_label,min_distance,min_anomaly_score,is_drift\n')
        for k in range(len(X_test)):
            f.write(f'{k},{closest_family_arr[k]},{y_test[k]},{min_dist_arr[k]:.6f},{min_anomaly_score_arr[k]:.6f},{int(is_drift[k])}\n')

    # 3) Cluster the drifted samples in latent space
    drift_indices = np.where(is_drift)[0]
    z_drift = z_test[drift_indices] if len(drift_indices) > 0 else np.empty((0, z_train.shape[1]))
    X_drift = X_test[drift_indices] if len(drift_indices) > 0 else np.empty((0, X_test.shape[1]))
    y_drift = y_test[drift_indices] if len(drift_indices) > 0 else np.array([], dtype=y_test.dtype)

    if z_drift.shape[0] == 0:
        warnings.warn("No drifted samples detected with current threshold. Exiting clustering.")
        return {"clusters": [], "drift_indices": drift_indices.tolist(), "summary": {"n_drifted": 0}}

    labels = _cluster_latents(z_drift, min_cluster_size=min_cluster_size, min_samples=None)

    df = pd.DataFrame({
        "idx": range(len(y_test)),
        "actual_label": y_test,
        "cluster_family_label": labels
    })
    df.to_csv("reports/cluster_assignments.csv", index=False)

    ari = adjusted_rand_score(y_test, labels)
    nmi = normalized_mutual_info_score(y_test, labels)
    print(f"ARI: {ari:.3f}, NMI: {nmi:.3f}")

    unique_labels = np.unique(labels)
    cluster_infos = []
    for lbl in unique_labels:
        cluster_idx_mask = (labels == lbl)
        cluster_indices = drift_indices[cluster_idx_mask]
        cluster_size = int(np.sum(cluster_idx_mask))
        cluster_centroid = np.mean(z_drift[cluster_idx_mask], axis=0) if cluster_size > 0 else np.zeros(z_train.shape[1])
        ctype, closest_class, dists = _classify_cluster(cluster_centroid, class_centroids, mad_family)
        X_cluster = X_test[cluster_indices] if cluster_size > 0 else np.empty((0, X_test.shape[1]))
        X_train_class = X_train[y_train == closest_class] if closest_class < len(np.unique(y_train)) else np.empty((0, X_train.shape[1]))
        top_features = top_k_feature_deltas(X_cluster, X_train_class, k=explain_top_k)

        cluster_infos.append({
            "cluster_label": int(lbl),
            "size": cluster_size,
            "indices": cluster_indices.tolist(),
            "type": ctype,
            "closest_class": int(closest_class),
            "distances": [float(d) for d in dists],
            "top_features": top_features
        })

    # 4) Save cluster report
    cluster_report_path = os.path.join(out_folder, 'multiclass_drift_clusters.csv')
    with open(cluster_report_path, 'w') as f:
        f.write('cluster_label,size,type,closest_class,distances,top_features,example_index\n')
        for c in cluster_infos:
            example = c["indices"][0] if len(c["indices"])>0 else -1
            f.write(f'{c["cluster_label"]},{c["size"]},{c["type"]},{c["closest_class"]},"{c["distances"]}","{c["top_features"]}",{example}\n')

    summary = {
        "n_test": len(X_test),
        "n_drifted": int(len(drift_indices)),
        "n_clusters": int(len([c for c in cluster_infos if c["cluster_label"] != -1])),
        "cluster_breakdown": Counter([c["type"] for c in cluster_infos])
    }

    return {"clusters": cluster_infos, "drift_indices": drift_indices.tolist(), "summary": summary, "cluster_report_path": cluster_report_path}



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
    X_train, y_train, X_test, y_test = split_by_families(X, y, train_families=[int(d) for d in train_f])
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

    print('Detect drifting samples in the testing set (multiclass detection + clustering)...')
    mad_threshold  = 3.5

    create_folder('reports-v2')
    create_folder('reports-v2/'+train_f)

    res = detect_and_cluster_multiclass(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        dims=cae_dims,
        margin=margin,
        mad_threshold=mad_threshold,
        best_weights_file=ae_weights_path,
        out_folder='reports-v2' + train_f,
        min_cluster_size=8,
        explain_top_k=10,
        save_training_info=True
    )

    print("Detection summary:", res["summary"])
    print("Cluster report saved to:", res.get("cluster_report_path"))

if __name__ == "__main__":
    main()
