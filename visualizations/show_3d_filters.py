import sys

sys.path.append('keras_layers/')
from keras.models import Model
import embeddings
from keras.layers import Input, Dense, BatchNormalization, Activation
import models2
import weight_logo_3d
import pipelines
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def load_PWMs(aa_features='both'):
    from rebuild_sequencepartition import build_partitions
    import pipelines
    base_path = '/specific/netapp5_2/'
    data_folder = base_path + 'iscb/wolfson/jeromet/Data/'
    pipeline_folder = base_path + \
                      'iscb/wolfson/jeromet/Data/InterfacePrediction/pipelines/'
    model_folder = base_path + 'iscb/wolfson/jeromet/Data/InterfacePrediction/models/'
    predictions_folder = base_path + \
                         'iscb/wolfson/jeromet/Data/InterfacePrediction/predictions/'

    history_folder = base_path + 'iscb/wolfson/jeromet/Data/InterfacePrediction/history/'

    initial_values_folder = base_path + \
                            'iscb/wolfson/jeromet/Data/InterfacePrediction/initial_values/'

    plot_folder = 'plots/'
    subset_indexes = None
    nchunks = 40
    with_aa = True  # True
    with_atom = True
    Lmax = 1024
    Lmax_atom = 9216
    atom_features = 'id'
    atom_features_embedding = True
    frame_orientation = 'sidechain'
    scaled_pwm = False
    method = 'Heavy4'
    Beff = 500
    padded = False
    return_index = True

    all_resolutions = ['S', '95', '90', '70', 'CATH', 'CAT']
    representatives = all_resolutions[5]
    reweighting = all_resolutions[5]
    aggregated = all_resolutions[1]
    if aggregated == 'S':
        aggregated = False
    hierarchical_weights = True

    partitions, partition_names, dockground_metadata, all_weights, all_weights_repeated = build_partitions(
        subset_indexes, nchunks, Lmax=None, exclude_antibodies=True, base_path=base_path,
        representatives=representatives,
        weights=reweighting, hierarchical_weights=hierarchical_weights)

    format_model_name = '{algo}_Lmax:{Lmax}_Lmaxatom:{Lmax_atom}_frameorientation:{frame_orientation}_Beff:{Beff}_rep:{representatives}_agg:{aggregated}_weighted:{weighted}_hierarchical:{hierarchical_weights}_method:{method}'

    train_mask, validation_mask, test_mask = partitions['Set']
    train_weights = all_weights[train_mask]
    validation_weights = all_weights[validation_mask]
    test_weights = all_weights[test_mask]

    pipeline = pipelines.ResidueAtomPointCloudPipeline2(
        'dockground',
        with_aa=True,
        with_atom=True,
        Lmax_aa=1024,
        aa_features=aa_features,
        atom_features='valency',
        Beff=500,
        method='Heavy4',
        aggregated='95',
        padded=False)

    inputs, outputs = pipeline.load_processed_dataset(
        fresh=False, save=True)
    return np.concatenate(inputs[1][train_mask], axis=0)


def get_submodel(model):
    activation = model.kwargs['activation']
    nfeatures = model.kwargs['nfeatures_aa']
    nembedding_aa = model.kwargs['nembedding_aa']
    if model.kwargs['with_atom']:
        nfilters = model.kwargs['nfilters_aa'] // 2
        filter_layer = 'spatiochemical_filters_input_aa1'
    else:
        nfilters = model.kwargs['nfilters_aa']
        filter_layer = 'filter_input_aa'

    nGaussianKernels = model.kwargs['N_aa']
    print(nfilters, nGaussianKernels, nfeatures, nembedding_aa)
    print(model.model.get_layer(filter_layer).get_weights()[0].shape)
    print(model.model.get_layer(filter_layer).get_weights()[1].shape)

    I = Input(shape=(nfeatures,))
    H = embeddings.OuterProduct(nembedding_aa, use_single1=True, use_single2=False, use_bias=False,
                                non_negative=False, name='embedded_attributes_aa')([I, I])
    H = models2.addNonLinearity(H, activation,
                                name='embedded_attributes_aa_normalized')

    out = Dense(nGaussianKernels * nfilters, use_bias=True, name=filter_layer)(H)
    submodel = Model(inputs=I, output=out)
    for layer in submodel.layers[1:-1]:
        layer.set_weights(model.model.get_layer(layer.name).get_weights())

    submodel.get_layer(filter_layer).set_weights(
        [np.swapaxes(model.model.get_layer(filter_layer).get_weights()[0], 0, 1).reshape(
            [nembedding_aa, nGaussianKernels * nfilters]),
            model.model.get_layer(filter_layer).get_weights()[1].reshape(nGaussianKernels * nfilters)]
    )
    return submodel


def get_top_PWMs(model, minimal_entropy=0):
    if model.kwargs['nfeatures_aa'] == 20:
        aa_features = 'sequence'
    if model.kwargs['nfeatures_aa'] == 21:
        aa_features = 'pwm'
    else:
        aa_features = 'both'

    PWMs = load_PWMs(aa_features=aa_features)
    #     PWMs = np.random.rand(100,41)**10
    #     PWMs[:,:21] /= PWMs[:,:21].sum(-1,keepdims=True)
    submodel = get_submodel(model)
    nGaussianKernels = model.kwargs['N_aa']
    if model.kwargs['with_atom']:
        nfilters = model.kwargs['nfilters_aa'] // 2
    else:
        nfilters = model.kwargs['nfilters_aa']

    activations = submodel.predict(PWMs).reshape([len(PWMs), nGaussianKernels, nfilters])

    if minimal_entropy > 0:
        entropy = -(np.log(PWMs + 1e-6) * (PWMs + 1e-6)).sum(-1)
        index_max = np.argmax(activations + 100 * (entropy > minimal_entropy)[:, np.newaxis, np.newaxis], axis=0)
        index_min = np.argmin(activations - 100 * (entropy > minimal_entropy)[:, np.newaxis, np.newaxis], axis=0)
    else:
        index_max = np.argmax(activations, axis=0)
        index_min = np.argmin(activations, axis=0)
    PWM_max = PWMs[index_max]
    PWM_min = PWMs[index_min]
    value_max = np.zeros([nGaussianKernels, nfilters])
    value_min = np.zeros([nGaussianKernels, nfilters])
    for mu in range(nGaussianKernels):
        for nu in range(nfilters):
            value_max[mu, nu] = activations[index_max[mu, nu], mu, nu]
            value_min[mu, nu] = activations[index_min[mu, nu], mu, nu]
    return PWM_max, value_max, PWM_min, value_min


def plot_atomic_filters(model, index, threshold1=0.33, threshold2=0.25, sg=None, atom_features='type',
                        camera_position=None,version=4):
    if camera_position is None:
        camera_position = [0.8, 0.5, 0.8]
    if version>=4:
        atom_embedding_weights = model.model.get_layer('embedded_attributes_atom').get_weights()[0]
        atom_filter_weights = model.model.get_layer('SCAN_filter_input_atom').get_weights()[0]
        atom_filter_offsets = model.model.get_layer('SCAN_filter_input_atom').get_weights()[1]

    else:
        atom_embedding_weights = model.model.get_layer('embedded_attributes_atom').get_weights()[0]
        atom_filter_weights = model.model.get_layer('filters_input_atom').get_weights()[0]
        atom_filter_offsets = model.model.get_layer('filters_input_atom').get_weights()[1]

    atom_gaussian_weights = np.dot(atom_embedding_weights, atom_filter_weights)
    atom_gaussian_weights = atom_gaussian_weights[1:] + atom_filter_offsets[np.newaxis]

    gaussiankernel_layer = model.model.get_layer('embedded_local_coordinates_atom')
    if gaussiankernel_layer.covariance_type == 'full':
        centers, sqrt_precisions = gaussiankernel_layer.get_weights()
        covariances = [0.1 * np.eye(3) + np.linalg.inv(np.dot(sqrt_precisions[:, :, u].T, sqrt_precisions[:, :, u])) for
                       u in range(sqrt_precisions.shape[-1])]
    else:
        centers, widths = gaussiankernel_layer.get_weights()
        covariances = [0.1 * np.eye(3) + np.diag(widths[:, u]) ** 2 for u in range(widths.shape[-1])]

    atom_axis = 0
    atom_gaussian_weights = atom_gaussian_weights[:, :, index]

    important_gaussians = \
    np.nonzero(
        np.abs(atom_gaussian_weights).max(atom_axis) > max(threshold2,threshold1 * np.abs(atom_gaussian_weights).max() ) )[0]

    logo_length_pos = np.maximum(atom_gaussian_weights[:, important_gaussians], 0).sum(atom_axis)
    logo_length_neg = np.maximum(-atom_gaussian_weights[:, important_gaussians], 0).sum(atom_axis)
    for k,u in enumerate(important_gaussians):
        if (atom_gaussian_weights[:,u]<-threshold2).sum()>=10:
            logo_length_pos[k] /= atom_gaussian_weights.shape[0]
            logo_length_neg[k] /= atom_gaussian_weights.shape[0]


    ymax = np.maximum(logo_length_pos,logo_length_neg).max()


    list_ellipsoids = [(centers[:, u], covariances[u]) for u in important_gaussians]
    if atom_features == 'type':
        weight_logo = weight_logo_3d.weight_logo_atom
        list_atoms = weight_logo_3d.list_atoms
        atom_colors = weight_logo_3d.atom_colors
    elif atom_features == 'valency':
        weight_logo = weight_logo_3d.weight_logo_valency
        list_atoms = weight_logo_3d.list_atom_valencies
        atom_colors = weight_logo_3d.valency_colors

    list_figures = [weight_logo(atom_gaussian_weights[:, u], ymax=ymax, threshold=threshold2) for u
                    in important_gaussians]
    list_colors = []
    for u in important_gaussians:
        if atom_gaussian_weights[:, u].max() > threshold2:
            color = atom_colors[list_atoms[np.argmax(atom_gaussian_weights[:, u])]]
        else:
            color = 'gray'
        list_colors.append(color)

    return weight_logo_3d.show_ellipsoids(list_ellipsoids=list_ellipsoids,
                                          list_colors=list_colors,
                                          list_figures=list_figures,
                                          level=1.0, sg=sg,
                                          wireframe=True,
                                          show_frame=True,
                                          fs=1.,
                                          scale=2.0,
                                          offset=[0, 0, 0],
                                          camera_position=camera_position,
                                          key_light_position=[0.5, 1, 0],
                                          maxi=5)


def plot_evolutionary_filters(model, index, top_PWM_pos, top_PWM_neg, top_value_pos, top_value_neg, threshold1=0.33,
                              threshold2=0.05, sg=None, camera_position=None):
    if camera_position is None:
        camera_position = [0.8, 0.5, 0.8]
    gaussiankernel_layer = model.model.get_layer('embedded_local_coordinates_aa')
    if gaussiankernel_layer.covariance_type == 'full':
        centers, sqrt_precisions = gaussiankernel_layer.get_weights()
        covariances = [0.1 * np.eye(3) + np.linalg.inv(np.dot(sqrt_precisions[:, :, u].T, sqrt_precisions[:, :, u])) for
                       u in range(sqrt_precisions.shape[-1])]
    else:
        centers, widths = gaussiankernel_layer.get_weights()
        covariances = [0.1 * np.eye(3) + np.diag(widths[:, u]) ** 2 for u in range(widths.shape[-1])]

    top_PWM_pos = top_PWM_pos[:, index][:, -21:]
    top_PWM_neg = top_PWM_neg[:, index][:, -21:]
    top_value_pos = top_value_pos[:, index]
    top_value_neg = top_value_neg[:, index]

    contrast = np.maximum(np.abs(top_value_pos), np.abs(top_value_neg))
    ymax = contrast.max()
    important_gaussians = np.nonzero(contrast > contrast.max() * threshold1)[0]

    list_ellipsoids = [(centers[:, u], covariances[u]) for u in important_gaussians]
    list_figures = [weight_logo_3d.weight_logo_aa(top_PWM_pos[u], top_value_pos[u], PWM_neg=top_PWM_neg[u],
                                                  value_neg=top_value_neg[u], ymax=ymax, threshold=threshold2) for u in
                    important_gaussians]
    list_colors = [weight_logo_3d.aa_colors[weight_logo_3d.list_aa[np.argmax(top_PWM_pos[u])]] for u in
                   important_gaussians]

    return weight_logo_3d.show_ellipsoids(list_ellipsoids=list_ellipsoids,
                                          list_colors=list_colors,
                                          list_figures=list_figures,
                                          level=1.0, sg=sg,
                                          wireframe=True,
                                          show_frame=True,
                                          fs=1.,
                                          scale=20.,
                                          offset=[0, 0, 0],
                                          camera_position=camera_position,
                                          key_light_position=[0.5, 1, 0],
                                          maxi=12)




def plot_evolutionary_filters2(model, index, threshold1=0.33,
                              threshold2=0.05, sg=None, camera_position=None):
    if camera_position is None:
        camera_position = [0.8, 0.5, 0.8]
    gaussiankernel_layer = model.model.get_layer('embedded_local_coordinates_aa')
    if gaussiankernel_layer.covariance_type == 'full':
        centers, sqrt_precisions = gaussiankernel_layer.get_weights()
        covariances = [0.1 * np.eye(3) + np.linalg.inv(np.dot(sqrt_precisions[:, :, u].T, sqrt_precisions[:, :, u])) for
                       u in range(sqrt_precisions.shape[-1])]
    else:
        centers, widths = gaussiankernel_layer.get_weights()
        covariances = [0.1 * np.eye(3) + np.diag(widths[:, u]) ** 2 for u in range(widths.shape[-1])]

    W = model.model.get_layer('spatiochemical_filters_input_aa1').get_weights()[0][:,:,index]
    contrast = np.maximum(np.maximum(W,0).sum(1), np.abs(np.minimum(W,0).sum(1)) )
    ymax = contrast.max()
    important_gaussians = np.nonzero(contrast > contrast.max() * threshold1)[0]

    list_ellipsoids = [(centers[:, u], covariances[u]) for u in important_gaussians]
    list_figures = [weight_logo_3d.weight_logo_aa2(W[u],ymax=ymax, threshold=threshold2) for u in
                    important_gaussians]
    list_colors = [weight_logo_3d.aa_colors[weight_logo_3d.list_aa[np.argmax(W[u])]] for u in
                   important_gaussians]

    return weight_logo_3d.show_ellipsoids(list_ellipsoids=list_ellipsoids,
                                          list_colors=list_colors,
                                          list_figures=list_figures,
                                          level=1.0, sg=sg,
                                          wireframe=True,
                                          show_frame=True,
                                          fs=1.,
                                          scale=4.,
                                          offset=[0, 0, 0],
                                          camera_position=camera_position,
                                          key_light_position=[0.5, 1, 0],
                                          maxi=12)



def plot_atomic_neighborhood(neighbors, atom_id, index=None,sg=None, camera_position=None):
    if camera_position is None:
        camera_position = [0.8, 0.5, 0.8]
    Kmax = neighbors.shape[0]
    if atom_id.max() > 4:
        atom_features = 'valency'
    else:
        atom_features = 'atom'

    if atom_features == 'type':
        weight_logo = weight_logo_3d.weight_logo_atom
        list_atoms = weight_logo_3d.list_atoms
        atom_colors = weight_logo_3d.atom_colors
    elif atom_features == 'valency':
        weight_logo = weight_logo_3d.weight_logo_valency
        list_atoms = weight_logo_3d.list_atom_valencies
        atom_colors = weight_logo_3d.valency_colors

    list_ellipsoids = [(neighbors[k], np.eye(3) * 0.2 ** 2) for k in range(Kmax)]
    list_colors = [atom_colors[list_atoms[atom_id[k] - 1]] for k in range(Kmax)]
    list_figures = []
    for k in range(Kmax):
        weights = np.zeros(len(list_atoms))
        weights[atom_id[k] - 1] = 4.
        list_figures.append(weight_logo(weights, ymax=4, threshold=0.5, bar=False))

    distances = np.sqrt(((neighbors[:, np.newaxis] - neighbors[np.newaxis, :]) ** 2).sum(-1))
    if index is not None:
        res_distances = np.abs( index[:,np.newaxis,0] - index[np.newaxis,:,0] )
        is_c_or_n =   np.array([id_-1 in [0,1,7,8] for id_ in atom_id])
        mask = (res_distances ==0 ) | ( (res_distances==1) & (is_c_or_n[:,np.newaxis]) & (is_c_or_n[np.newaxis,:]) )
        distances += (1-mask) * 10

    segment_starts,segment_ends =  np.nonzero( (distances > 0) & (distances < 1.7) )
    list_segments = [ [list(neighbors[segment_start]), list(neighbors[segment_end]) ] for segment_start,segment_end in zip(segment_starts,segment_ends) if segment_end>segment_start ]


    return weight_logo_3d.show_ellipsoids(list_ellipsoids=list_ellipsoids,
                                          list_colors=list_colors,
                                          list_figures=list_figures,
                                          list_segments=list_segments,
                                          level=1.0, sg=sg,
                                          wireframe=False,
                                          show_frame=True,
                                          fs=1.,
                                          scale=1.0,
                                          offset=[0., 0.25, 0.0],
                                          camera_position=camera_position,
                                          key_light_position=[0.5, 1, 0],
                                          opacity=0.6,
                                          maxi=5)



def plot_gaussians(params,sg=None,maxi=5,camera_position=None):

    covariance_type = 'full' if params[1].ndim>2 else 'diag'
    if covariance_type == 'full':
        centers, sqrt_precisions = params
        covariances = [0.1 * np.eye(3) + np.linalg.inv(np.dot(sqrt_precisions[:, :, u].T, sqrt_precisions[:, :, u])) for
                       u in range(sqrt_precisions.shape[-1])]
    else:
        centers, widths = params
        covariances = [0.1 * np.eye(3) + np.diag(widths[:, u]) ** 2 for u in range(widths.shape[-1])]

    K = centers.shape[1]
    list_ellipsoids = [(centers[:, u], covariances[u]) for u in range(K)]
    list_colors = [matplotlib.colors.to_rgb('C%s'%k) for k in range(K)]

    return weight_logo_3d.show_ellipsoids(list_ellipsoids=list_ellipsoids,
                                          list_colors=list_colors,
                                          level=1.0, sg=sg,
                                          wireframe=False,
                                          show_frame=True,
                                          fs=1.,
                                          scale=0.75,
                                          camera_position=camera_position,
                                          key_light_position=[0.5, 1, 0],
                                          opacity=0.3,
                                          maxi=maxi)




def plot_complex_filter(model, index,
                        value_pos,
                        aa_probability,
                        value_neg=None,
                        aa_probability_neg=None,
                        hb_probability=None,
                        hb_probability_neg=None,
                        ss_probability=None,
                        ss_probability_neg=None,
                        asa_probability=None,
                        asa_probability_neg = None,
                        threshold1=0.33,
                        scale=5.,
                        sg=None, camera_position=None):
    if camera_position is None:
        camera_position = [0.8, 0.5, 0.8]


    gaussiankernel_layer = model.model.get_layer('embedded_local_coordinates_aa')
    if gaussiankernel_layer.covariance_type == 'full':
        centers, sqrt_precisions = gaussiankernel_layer.get_weights()
        covariances = [0.1 * np.eye(3) + np.linalg.inv(np.dot(sqrt_precisions[:, :, u].T, sqrt_precisions[:, :, u])) for
                       u in range(sqrt_precisions.shape[-1])]
    else:
        centers, widths = gaussiankernel_layer.get_weights()
        covariances = [0.1 * np.eye(3) + np.diag(widths[:, u]) ** 2 for u in range(widths.shape[-1])]

    ngaussians = centers.shape[-1]
    nfilters = len(value_pos)
    if aa_probability_neg is None:
        aa_probability_neg = [[None for _ in range(ngaussians)] for _ in range(nfilters)]
    if hb_probability is None:
        hb_probability = [[None for _ in range(ngaussians)] for _ in range(nfilters)]
    if hb_probability_neg is None:
        hb_probability_neg = [[None for _ in range(ngaussians)] for _ in range(nfilters)]
    if ss_probability is None:
        ss_probability = [[None for _ in range(ngaussians)] for _ in range(nfilters)]
    if ss_probability_neg is None:
        ss_probability_neg = [[None for _ in range(ngaussians)] for _ in range(nfilters)]
    if asa_probability is None:
        asa_probability = [[None for _ in range(ngaussians)] for _ in range(nfilters)]
    if asa_probability_neg is None:
        asa_probability_neg =[[None for _ in range(ngaussians)] for _ in range(nfilters)]

    if value_neg is None:
        value_neg = value_pos * 0

    maximum = np.maximum(np.abs(value_pos[index]).max(),np.abs(value_neg[index]).max() )
    scaling = value_pos[index]/maximum
    scaling_neg = np.abs(value_neg[index]/maximum)
    important_gaussians = np.nonzero( (scaling > threshold1)  | (scaling_neg > threshold1) )[0]

    list_ellipsoids = [(centers[:, u], covariances[u]) for u in important_gaussians]

    list_figures = [weight_logo_3d.complex_filter_logo(
        aa_probability[index][u],
        aa_probability_neg=aa_probability_neg[index][u],
        hb_probability=hb_probability[index][u],
        hb_probability_neg=hb_probability_neg[index][u],
        ss_probability=ss_probability[index][u],
        ss_probability_neg=ss_probability_neg[index][u],
        asa_probability=asa_probability[index][u],
        asa_probability_neg = asa_probability_neg[index][u],
        scaling_='none',
        scaling=scaling[u],
        scaling_neg=scaling_neg[u]) for u in important_gaussians]

    list_colors = [weight_logo_3d.aa_colors[weight_logo_3d.list_aa[np.argmax(aa_probability[index][u])]] for u in important_gaussians]

    return weight_logo_3d.show_ellipsoids(list_ellipsoids=list_ellipsoids,
                                          list_colors=list_colors,
                                          list_figures=list_figures,
                                          level=1.0, sg=sg,
                                          wireframe=True,
                                          show_frame=True,
                                          fs=1.,
                                          scale=scale,
                                          offset=[0, 2.5, 0],
                                          camera_position=camera_position,
                                          key_light_position=[0.5, 1, 0],
                                          maxi=12,dpi=600,crop=False)

def plot_aminoacid_neighborhood(neighbors, pwm, index=None, sg=None, camera_position=None):
    if camera_position is None:
        camera_position = [0.8, 0.5, 0.8]
    Kmax = neighbors.shape[0]
    conservation = np.log(21) + (np.log(pwm + 1e-10) * (pwm+1e-10) ).sum(-1)

    list_ellipsoids = [(neighbors[k], np.eye(3) * 0.1 ** 2) for k in range(Kmax)]
    list_colors = [weight_logo_3d.aa_colors[ weight_logo_3d.list_aa[np.argmax(pwm[k])]] for k in range(Kmax) ]
    # list_figures = [weight_logo_3d.weight_logo_aa(pwm[k],conservation[k],ymax=np.log(21)) for k in range(Kmax)]
    list_figures = [weight_logo_3d.weight_logo_aa(pwm[k], 1.0, ymax=1.0 ) for k in range(Kmax)]

    if index is not None:
        segment_starts,segment_ends = np.nonzero( index[:,np.newaxis] == (index[np.newaxis,:]+1) )
        list_segments = [ [list(neighbors[segment_start]), list(neighbors[segment_end]) ] for segment_start,segment_end in zip(segment_starts,segment_ends) ]
    else:
        list_segments = None

    return weight_logo_3d.show_ellipsoids(list_ellipsoids=list_ellipsoids,
                                          list_colors=list_colors,
                                          list_figures=list_figures,
                                          list_segments=list_segments,
                                          level=1.0, sg=sg,
                                          wireframe=False,
                                          show_frame=True,
                                          fs=1.,
                                          scale=1.5,
                                          offset=[0., 1.0, 0.0],
                                          camera_position=camera_position,
                                          key_light_position=[0.5, 1, 0],
                                          opacity=0.6,
                                          maxi=12,dpi=300,crop=True)