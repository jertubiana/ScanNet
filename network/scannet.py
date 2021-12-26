from keras.models import Model
from keras.layers import Input, Dense, Masking, TimeDistributed, Concatenate, Activation, Embedding, Dropout, Lambda
from keras.initializers import Zeros, Ones,RandomUniform
import numpy as np
import keras.regularizers
from . import neighborhoods, attention, embeddings, utils
from functools import partial
import keras.backend as K
import tensorflow as tf
from utilities import wrappers,io_utils
from preprocessing import PDB_processing


def l1_regularization(W, l1):
    return l1 * K.sum(K.abs(W))


def l12_regularization(W, l12, ndims=2, order='gaussian_feature_filter'):
    if order == 'filter_gaussian_feature':  # Order of the tensor indices.
        if ndims == 2: # For gaussian-dependent bias
            return l12 / 2 * tf.cast(K.shape(W)[1], tf.float32) * K.sum(K.square(K.mean(K.abs(W), axis=1)))
        elif ndims == 3:
            return l12 / 2 * tf.cast(K.shape(W)[1] * K.shape(W)[2], tf.float32) * K.sum(
                K.square(K.mean(K.abs(W), axis=(1, 2))))
    elif order == 'gaussian_feature_filter':  # Default order the tensor indices.
        if ndims == 2:
            return l12 / 2 * tf.cast(K.shape(W)[0], tf.float32) * K.sum(K.square(K.mean(K.abs(W), axis=0)))
        elif ndims == 3:
            return l12 / 2 * tf.cast(K.shape(W)[0] * K.shape(W)[1], tf.float32) * K.sum(
                K.square(K.mean(K.abs(W), axis=(0, 1))))


def l12group_regularization(W, l12group, ndims=2, order='gaussian_feature_filter'):
    if ndims == 2: # For gaussian-dependent bias, Same as l12
        return l12_regularization(W, l12group, ndims=ndims, order=order)
    elif ndims == 3:
        if order == 'filter_gaussian_feature': # Order of the tensor indices.
            return l12group / 2 * tf.cast(K.shape(W)[1] * K.shape(W)[2], tf.float32) * K.sum(
                K.square(K.mean(K.sqrt(K.mean(K.square(W), axis=-1)), axis=-1)))
    elif order == 'gaussian_feature_filter': # Order of the tensor indices.
        return l12group / 2 * tf.cast(K.shape(W)[0] * K.shape(W)[1], tf.float32) * K.sum(
            K.square(K.mean(K.sqrt(K.mean(K.square(W), axis=1)), axis=0)))


def addNonLinearity(input_layer, activation, name=None):
    if name is None:
        name = 'activity_nonlinear'

    if activation == 'tanh':
        center = True
        scale = True
    elif 'multitanh' in activation: # e.g. 'multitanh5'
        center = False
        scale = False
    elif activation == 'relu':
        center = True
        scale = False
    elif activation == 'elu':
        center = True
        scale = True
    elif activation in [None,'linear']:
        center = False
        scale = False
    else:
        center = True
        scale = True

    input_layer_normalized = embeddings.MaskedBatchNormalization(
        epsilon=1e-3, axis=-1, center=center, scale=scale, name=name + '_normalization')(input_layer) # Custom batch norm layer that takes into account the mask.

    if 'multitanh' in activation:
        output_layer = embeddings.MultiTanh(int(activation[-1]), use_bias=True, name=name)(
            input_layer_normalized)
    else:
        output_layer = TimeDistributed(Activation(
            activation), name=name)(input_layer_normalized)
    return output_layer


def attribute_embedding(attribute_layer,output_dim, activation,name=None):
    if name is None:
        name = 'attribute_embedding'

    if output_dim is None: # Does nothing. Just passes through mask-preserving identity layer to change name.
        return TimeDistributed(Activation('linear'),name=name)(attribute_layer)
    else:
        projected_attribute_layer = TimeDistributed(Dense(output_dim, use_bias=False, activation=None),
                                                 name=name+'_projection')(attribute_layer)
        embedded_attribute_layer = addNonLinearity(projected_attribute_layer, activation,
                                                  name=name)
        return embedded_attribute_layer

def attribute_normalizer(attribute_layer,name=None,normalizer=1.0):
    if name is None:
        name = 'attribute_normalizer'
    return TimeDistributed(Lambda(lambda x: normalizer * x), name=name)(attribute_layer)


def neighborhood_embedding(
        point_clouds,
        frame_indices,
        attributes,
        sequence_indices=None,
        order_frame='2',
        dipole_frame=False,
        Kmax=16,
        coordinates=['euclidian'],
        index_distance_max = 8,
        nrotations=1,
        Ngaussians=32,
        covariance_type = 'full',
        initial_gaussian_values = None,
        l1=0.,
        l12=0.,
        l12group=0.,
        nfilters=64,
        activation = 'relu',
        scale='atom',
        return_frames=False):
    # Build frames.
    frames = neighborhoods.FrameBuilder(name='frames_%s'%scale,
                                        order=order_frame, # The permutation to build the frame (for order ='2', the zaxis is defined from X_1 -> X_3).
                                        dipole=dipole_frame # Only used if using a quadruplet of indices. Then, frame has a 4rth additional direction (the dipole).
                                        )([point_clouds,frame_indices])
    # Compute local neighborhoods.
    if 'index_distance' in coordinates: # Add sequence position distance.
        assert sequence_indices is not None
        input2localneighborhood = [frames, sequence_indices, attributes]
    else:
        input2localneighborhood = [frames, attributes]

    local_coordinates, local_attributes = neighborhoods.LocalNeighborhood(Kmax=Kmax,
                                                                          coordinates=coordinates,
                                                                          index_distance_max=index_distance_max,
                                                                          nrotations=nrotations,
                                                                          name='neighborhood_%s'%scale)(input2localneighborhood)

    # Gaussian embedding of local coordinates.
    embedded_local_coordinates = embeddings.GaussianKernel(Ngaussians,
                                                           initial_values=initial_gaussian_values,
                                                           covariance_type=covariance_type,
                                                           name='embedded_local_coordinates_%s'%scale)(local_coordinates)

    # Apply Spatio-chemical filters.
    if l1 > 0:
        kernel_regularizer = partial(l1_regularization, l1=l1)
        single1_regularizer = kernel_regularizer
        fixednorm = np.sqrt(Ngaussians / Kmax)
    elif l12 > 0:
        kernel_regularizer = partial(l12_regularization, l12=l12, ndims=3)
        single1_regularizer = partial(l12_regularization, l12=l12, ndims=2)
        fixednorm = np.sqrt(Ngaussians / Kmax)
    elif l12group >0:
        kernel_regularizer = partial(l12group_regularization, l12group=l12group, ndims=3)
        single1_regularizer = partial(l12group_regularization, l12group=l12group, ndims=2)
        fixednorm = np.sqrt(Ngaussians / Kmax)

    else:
        kernel_regularizer = None
        single1_regularizer = None
        fixednorm = None

    spatiochemical_filters_input = embeddings.OuterProduct(nfilters,
                                                           use_single1=True, use_single2=False,
                                                            use_bias=False,
                                                            kernel_regularizer=kernel_regularizer,
                                                            single1_regularizer=single1_regularizer,
                                                            fixednorm=fixednorm,
                                                            non_negative=False,
                                                           non_negative_initial = False,
                                                           sum_axis=2,
                                                            name='SCAN_filter_input_%s'%scale)(
        [embedded_local_coordinates, local_attributes])

    if nrotations>1:
        spatiochemical_filters_activity = addNonLinearity(spatiochemical_filters_input, activation,
                                                          name='SCAN_filter_activity_prepooling_%s' % scale)  # Add non-linearity.

        spatiochemical_filters_activity = utils.MaxAbsPooling(axis=-2, name='SCAN_filter_activity_%s' % scale)(spatiochemical_filters_input)
    else:
        spatiochemical_filters_activity = addNonLinearity(spatiochemical_filters_input, activation,
                                                          name='SCAN_filter_activity_%s' % scale)  # Add non-linearity.

    if return_frames:
        return spatiochemical_filters_activity,frames
    else:
        return spatiochemical_filters_activity






def ScanNet(
        Lmax_aa=800,
        Lmax_atom=None,
        Lmax_aa_points=None,
        Lmax_atom_points=None,
        with_atom=True,
        K_aa=32,
        K_atom=32,
        K_graph=32,
        N_aa=32,
        N_atom=16,
        N_graph=16,
        nfeatures_atom=4,
        nfeatures_aa=20,
        nembedding_atom=4,
        nembedding_aa=None,
        nembedding_graph=1,
        nfilters_atom=16,
        nfilters_aa=64,
        nfilters_graph=2,
        dense_pooling=None,
        nattentionheads_pooling=1,
        filter_MLP=[],
        nattentionheads_graph=1,
        initial_values={'GaussianKernel_atom': None, 'GaussianKernel_aa': None, 'GaussianKernel_graph': None,
                        'dense_graph': None},
        covariance_type_atom='full',
        covariance_type_aa='full',
        covariance_type_graph='diag',
        activation='multitanh5',
        coordinates_atom=['euclidian'],
        coordinates_aa=['euclidian'],
        frame_aa='triplet_backbone',
        coordinates_graph=['distance', 'ZdotZ', 'ZdotDelta', 'index_distance'],
        index_distance_max_graph=16,
        nrotations =1,
        l1_aa=0.,
        l12_aa=0.,
        l12group_aa=0.,
        l1_atom=0.,
        l12_atom=0.,
        l12group_atom=0.,
        l1_pool=0.,
        l12_pool=0.,
        l12group_pool=0.,
        dropout=0.,
        optimizer='adam'):


    if frame_aa == 'triplet_backbone':
        order_aa = '3'
        dipole_aa = False
    elif frame_aa in ['triplet_sidechain', 'triplet_cbeta']:
        order_aa = '2'
        dipole_aa = False
    elif frame_aa == 'quadruplet':
        order_aa = '3'
        dipole_aa = True
    else:
        print('Incorrect frame_aa')
        return

    frame_atom = 'covalent'
    order_atom = '2'
    dipole_atom = False


    # Check initial values for GaussianKernel, and initialize to random uniform if needed.
    for component in ['aa', 'atom', 'graph']:
        try:
            assert (initial_values['GaussianKernel_%s' % component] is not None)
        except:
            print('Initial values for %s GaussianKernel not found, random initialization...' % component)

            coordinates = locals()['coordinates_%s' % component]
            xlims = []
            for coordinate in coordinates:
                if coordinate == 'euclidian':
                    mini = -8
                    maxi = 8
                    ndims = 3
                elif coordinate == 'dipole_spherical':
                    mini = - np.pi
                    maxi = np.pi
                    ndims = 2
                elif coordinate == 'ZdotDelta':
                    mini = -1
                    maxi = 1
                    ndims = 2
                else:
                    mini = -1
                    maxi = 1
                    ndims = 1
                for _ in range(ndims):
                    xlims.append([mini, maxi])

            N = locals()['N_%s' % component]
            covariance_type = locals()['covariance_type_%s' % component]
            key = 'GaussianKernel_%s' % component
            print(N, covariance_type, key)
            initial_values[key] = embeddings.initialize_GaussianKernelRandom(xlims, N, covariance_type)

    try:
        assert (initial_values['dense_graph'] is not None)
    except:
        print('Initial values for graph dense not found, random initialization...')
        initial_values['dense_graph'] = [np.random.rand(N_graph, nembedding_graph)]

    # Given maximum number of amino acids, define other maximum values.
    if Lmax_atom is None:
        Lmax_atom = 9 * Lmax_aa
    if Lmax_aa_points is None:
        if frame_aa == 'triplet_backbone':
            Lmax_aa_points = Lmax_aa + 2
        elif frame_aa == 'triplet_sidechain':
            Lmax_aa_points = 2 * Lmax_aa + 1
        elif frame_aa == 'quadruplet':
            Lmax_aa_points = 2 * Lmax_aa + 2
    if Lmax_atom_points is None:
        Lmax_atom_points = 11 * Lmax_aa

    ### Input layers.
    if frame_aa == 'quadruplet':
        nindex = 4
    else:
        nindex = 3


    # The 4 inputs arrays at amino acid scale.
    frame_indices_aa = Input(shape=[Lmax_aa, nindex], name='frame_indices_aa', dtype="int32") # List of triplet/quadruplet of the indices of the points for building the frames.
    attributes_aa = Input(shape=[Lmax_aa, nfeatures_aa], name='attributes_aa', dtype="float32") # The attributes of the amino acids (PWM/ one-hot encoded sequence).
    sequence_indices_aa = Input(shape=[Lmax_aa, 1], name='sequence_indices_aa', dtype="int32") # The position of each amino acid on the sequence.
    point_clouds_aa = Input(shape=[Lmax_aa_points, 3], name='point_clouds_aa', dtype="float32") # 3D coordinates of the amino acids point clouds. (Calpha coordinates, side chain center of mass,...).

    # Same, after masking.
    masked_frame_indices_aa = Masking(mask_value=-1, name='masked_frame_indices_aa')(frame_indices_aa)
    masked_attributes_aa = Masking(mask_value=0.0, name='masked_attributes_aa')(attributes_aa)
    masked_sequence_indices_aa = Masking(mask_value=-1, name='masked_sequence_indices_aa')(sequence_indices_aa)
    masked_point_clouds_aa = Masking(mask_value=0.0, name='masked_point_clouds_aa')(point_clouds_aa)

    # Same at atomic scale.
    if with_atom:
        frame_indices_atom = Input(shape=[Lmax_atom, 3], name='frame_indices_atom', dtype="int32")
        attributes_atom = Input(shape=[Lmax_atom], name='attributes_atom', dtype="int32")
        sequence_indices_atom = Input(shape=[Lmax_atom, 1], name='sequence_indices_atom', dtype="int32")
        point_clouds_atom = Input(shape=[Lmax_atom_points, 3], name='point_clouds_atom', dtype="float32")

        masked_frame_indices_atom = Masking(mask_value=-1, name='masked_frame_indices_atom')(frame_indices_atom)
        masked_sequence_indices_atom = Masking(mask_value=-1, name='masked_sequence_indices_atom')(sequence_indices_atom)
        masked_point_clouds_atom = Masking(mask_value=0.0, name='masked_point_clouds_atom')(point_clouds_atom)


    ## Embed attributes.

    if nembedding_aa is not None: # Apply point-wise dense embedding.
        embedded_attributes_aa = attribute_embedding(masked_attributes_aa,nembedding_aa,activation,name='embedded_attributes_aa')
    else: # Normalize such that that variance is approximately one.
        normalizer = np.sqrt(nfeatures_aa)
        embedded_attributes_aa = TimeDistributed(Lambda(lambda x: normalizer * x), name='embedded_attributes_aa')(masked_attributes_aa)
        nembedding_aa = nfeatures_aa

    if with_atom:
        if nfeatures_atom == nembedding_atom:
            trainable = False  # Same number of categories as output dimension. Use one-hot encoding.
        else:
            trainable = True
        embedded_attributes_atom = Embedding(
            nfeatures_atom + 1, nembedding_atom, mask_zero=True, embeddings_initializer=utils.embeddings_initializer,
            embeddings_constraint=utils.FixedNorm(axis=0, value=np.sqrt(nfeatures_atom)), trainable=trainable,
            name='embedded_attributes_atom')(attributes_atom)

        ## If atomic coordinates are included, compute embeddings of atomic neighborhoods.
        SCAN_filters_atom = neighborhood_embedding(
            masked_point_clouds_atom,
            masked_frame_indices_atom,
            embedded_attributes_atom,
            sequence_indices=masked_sequence_indices_atom,
            order_frame=order_atom,
            dipole_frame=dipole_atom,
            Kmax=K_atom,
            coordinates=coordinates_atom,
            Ngaussians=N_atom,
            covariance_type=covariance_type_atom,
            initial_gaussian_values=initial_values['GaussianKernel_atom'],
            l1=l1_atom,
            l12=l12_atom,
            l12group=l12group_atom,
            nfilters=nfilters_atom,
            activation=activation,
            scale='atom')


        # Pool at amino acid level by attention-pooling.
        if dense_pooling is None:
            dense_pooling = nfilters_atom

        if l12_pool > 0:
            kernel_regularizer = partial(l12_regularization, l12=l12_pool)
            kernel_constraint = utils.FixedNorm(1.0,axis=0)
        else:
            kernel_regularizer = None
            kernel_constraint = None

        # Compute attention coefficients (before softmax) for each atom.
        pooling_attention = TimeDistributed(Dense(nattentionheads_pooling, use_bias=False,
                                                  kernel_regularizer=kernel_regularizer,
                                                  kernel_initializer=Zeros()),
                                            name='attention_pooling_coefficients_atom')(SCAN_filters_atom)

        # Compute output coefficients (before averaging) for each atom.
        pooling_features = TimeDistributed(Dense(dense_pooling, activation=None, use_bias=False,
                                           kernel_regularizer=kernel_regularizer,
                                           kernel_constraint=kernel_constraint,
                                           #kernel_initializer= RandomUniform(minval=0,maxval=np.sqrt(3/nfilters_atom))
                                                 ),
                                           name='attention_pooling_features_atom')(SCAN_filters_atom)


        # Build a binary bipartite graph from amino acid to atoms.
        # For each amino acid we look for the 14 closest atoms in terms of sequence distance.
        pooling_mask, pooling_attention_local, pooling_features_local = neighborhoods.LocalNeighborhood(Kmax=14,
                                                                                                        coordinates=[
                                                                                                            'index_distance'],
                                                                                                        self_neighborhood=False,
                                                                                                        index_distance_max=1,
                                                                                                        name='pooling_intermediate_atom')(
            [masked_sequence_indices_aa, masked_sequence_indices_atom, pooling_attention, pooling_features])
        # Here pooling_mask is a binary matrix of size [L_aa, L_atom], with M_{ij} =0 iff atom j belongs to amino acid i.


        pooling_mask = Lambda(lambda x: 1 - x, name='pooling_edges_atom')(
            pooling_mask)
        # Reverse the pooling mask such that M_{ij} =1 iff atom j belongs to amino acid i.


        SCAN_filters_atom_aggregated_input, _ = attention.AttentionLayer(self_attention=False, beta=False,
                                                        name='SCAN_filters_atom_aggregated_input')(
            [pooling_attention_local, pooling_features_local, pooling_mask])
        # Attention-based aggregation of atom features to amino acid scale.

        SCAN_filters_atom_aggregated_activity = addNonLinearity(SCAN_filters_atom_aggregated_input, activation,name='SCAN_filters_atom_aggregated_activity')
        # Add final non-linearity.


    if with_atom:
        all_embedded_attributes_aa = Concatenate(name='all_embedded_attributes_aa', axis=-1)([embedded_attributes_aa, SCAN_filters_atom_aggregated_activity] )
    else:
        all_embedded_attributes_aa = Activation('linear',name='all_embedded_attributes_aa')(embedded_attributes_aa)



    # Compute embeddings of amino acid neighborhoods.
    SCAN_filters_aa,frames_aa = neighborhood_embedding(
        masked_point_clouds_aa,
        masked_frame_indices_aa,
        all_embedded_attributes_aa,
        sequence_indices=masked_sequence_indices_aa,
        order_frame=order_aa,
        dipole_frame=dipole_aa,
        Kmax=K_aa,
        coordinates=coordinates_aa,
        nrotations=nrotations,
        Ngaussians=N_aa,
        covariance_type=covariance_type_aa,
        initial_gaussian_values=initial_values['GaussianKernel_aa'],
        l1=l1_aa,
        l12=l12_aa,
        l12group=l12group_aa,
        nfilters=nfilters_aa,
        activation=activation,
        scale='aa',
        return_frames=True)

    if dropout > 0:
        SCAN_filters_aa = Dropout(dropout, noise_shape=(
            None, 1, None), name='dropout')(SCAN_filters_aa)

    embedded_filter = SCAN_filters_aa
    if filter_MLP:
        for k, n_feature_filter in enumerate(filter_MLP):
            embedded_filter = attribute_embedding(embedded_filter,n_feature_filter,activation,name='SCAN_filters_aa_embedded_%s'%(k+1))



    # Final graph attention layer. Propagates label information from "hotspots" to passengers to obtain spatially consistent labels.

    beta = TimeDistributed(Dense(nembedding_graph * nattentionheads_graph, use_bias=True, bias_initializer=Ones(),
                                 kernel_initializer=Zeros(), activation='relu'), name='beta')(embedded_filter)

    self_attention = TimeDistributed(
        Dense(nembedding_graph * nattentionheads_graph, use_bias=True, bias_initializer=Zeros(),
              kernel_initializer=Zeros()), name='self_attention')(embedded_filter)

    cross_attention = TimeDistributed(Dense(nembedding_graph * nattentionheads_graph, use_bias=False,
                                            kernel_initializer=Zeros()), name='cross_attention')(embedded_filter)

    if nfilters_graph > 2:
        node_features_activation = 'relu'
    else:
        node_features_activation = None

    node_features = TimeDistributed(Dense(
        nattentionheads_graph * nfilters_graph, activation=node_features_activation, use_bias=True),
        name='node_output')(embedded_filter)

    graph_weights, attention_local, node_features_local = neighborhoods.LocalNeighborhood(
        Kmax=K_graph, coordinates=coordinates_graph,
        index_distance_max=index_distance_max_graph, name='neighborhood_graph')(
        [frames_aa, masked_sequence_indices_aa, cross_attention, node_features])

    embedded_graph_weights = embeddings.GaussianKernel(N_graph, initial_values=initial_values['GaussianKernel_graph']
                                                       , covariance_type=covariance_type_graph,
                                                       name='embedded_local_coordinates_graph')(graph_weights)

    embedded_graph_weights = TimeDistributed(Dense(
        nembedding_graph, use_bias=False), name='edges_graph')(embedded_graph_weights)

    graph_attention_output, attention_coefficients = attention.AttentionLayer(name='attention_layer')(
        [beta, self_attention, attention_local, node_features_local, embedded_graph_weights])

    if nattentionheads_graph * nfilters_graph > 2:
        classifier_output = TimeDistributed(Dense(
            2, use_bias=True, activation='softmax'), name='classifier_output')(graph_attention_output)
    else:
        classifier_output = TimeDistributed(Activation(
            'softmax'), name='classifier_output')(graph_attention_output)

    inputs = [frame_indices_aa, attributes_aa, sequence_indices_aa, point_clouds_aa]
    if with_atom:
        inputs += [frame_indices_atom, attributes_atom, sequence_indices_atom, point_clouds_atom]


    model = Model(inputs=inputs,
                  outputs=classifier_output)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[
        'categorical_crossentropy', 'categorical_accuracy'])
    model.get_layer('edges_graph').set_weights(
        initial_values['dense_graph'])
    model.summary()
    return model



def initialize_ScanNet(
        inputs,
        outputs,
        with_atom=True,
        Lmax_aa=1024,
        K_aa=16,
        K_atom=16,
        K_graph=32,
        Dmax_aa=13.,
        Dmax_atom=4.,
        Dmax_graph=13.,
        N_aa=32,
        N_atom=32,
        N_graph=32,
        nfeatures_aa=21,
        nfeatures_atom=12,
        nembedding_atom=12,
        nembedding_aa=16,
        nembedding_graph=1,
        dense_pooling=None,
        nattentionheads_pooling=1,
        nfilters_atom=16,
        nfilters_aa=64,
        nfilters_graph=2,
        nattentionheads_graph=1,
        filter_MLP=[],
        covariance_type_atom='full',
        covariance_type_aa='full',
        covariance_type_graph='full',
        activation='relu',
        coordinates_atom=['euclidian'],
        coordinates_aa=['euclidian'],
        frame_aa='triplet_sidechain',
        coordinates_graph=['distance', 'ZdotZ', 'ZdotDelta', 'index_distance'],
        index_distance_max_graph=16,
        nrotations=1,
        l1_aa=0.,
        l12_aa=0.,
        l12group_aa=0.,
        l1_atom=0.,
        l12_atom=0.,
        l12group_atom=0.,
        l1_pool=0.,
        l12_pool=0.,
        l12group_pool=0.,
        dropout=0.,
        optimizer='adam',
        initial_values_folder='/specific/netapp5_2/iscb/wolfson/jeromet/Data/InterfacePrediction/initial_values/',
        fresh_initial_values=False,
        save_initial_values=True,
        n_init=10,
        epochs = 100,
        batch_size=1):


    Lmax_atom = 9 * Lmax_aa
    if frame_aa == 'triplet_backbone':
        Lmax_aa_points = Lmax_aa + 2
    elif frame_aa in ['triplet_sidechain','triplet_cbeta']:
        Lmax_aa_points = 2 * Lmax_aa + 1
    elif frame_aa == 'quadruplet':
        Lmax_aa_points = 2 * Lmax_aa + 2
    Lmax_atom_points = 11 * Lmax_aa

    initial_values = {'GaussianKernel_aa': None, 'GaussianKernel_atom': None, 'GaussianKernel_graph': None,'dense_graph': None}

    if with_atom:
        input_type = ['triplets', 'attributes', 'indices', 'points', 'triplets', 'attributes', 'indices', 'points']
        Lmaxs = [Lmax_aa, Lmax_aa, Lmax_aa, Lmax_aa_points, Lmax_atom, Lmax_atom, Lmax_atom, Lmax_atom_points]
    else:
        input_type = ['triplets', 'attributes', 'indices', 'points']
        Lmaxs = [Lmax_aa, Lmax_aa, Lmax_aa, Lmax_aa_points]

    if frame_aa == 'triplet_backbone':
        order_aa = '3'
        dipole_aa = False
    elif frame_aa in ['triplet_sidechain', 'triplet_cbeta']:
        order_aa = '2'
        dipole_aa = False
    elif frame_aa == 'quadruplet':
        order_aa = '3'
        dipole_aa = True
    else:
        print('Incorrecte frame_aa')
        return

    order_atom = '2'
    dipole_atom = False
    frame_atom = 'covalent'

    c_aa = ''.join([c[0] for c in coordinates_aa])
    location_aa = initial_values_folder + 'initial_GaussianKernel_aa_N_%s_Kmax_%s_Dmax_%s_frames_%s_coords_%s_nrotations_%s_cov_%s_tripletsorder_%s_dipole_%s.data' % (
        N_aa, K_aa, Dmax_aa, frame_aa, c_aa, nrotations, covariance_type_aa, order_aa, dipole_aa)

    try:
        assert fresh_initial_values == False
        initial_values['GaussianKernel_aa'] = io_utils.load_pickle(location_aa)[
            'GaussianKernel_aa']
    except:
        print(
            'Initializing the Gaussian kernels for the amino acid neighborhood (takes a few minutes to do it robustly, be patient!)  Reduce n_init from 10 to 1 if speed needed')
        initial_values['GaussianKernel_aa'] = neighborhoods.initialize_GaussianKernel_for_NeighborhoodEmbedding(
            [inputs[0], inputs[3]],
            N_aa,
            covariance_type_aa,
            neighborhood_params={'Kmax': K_aa,
                                 'coordinates': coordinates_aa,
                                 'nrotations': nrotations,
                                 'index_distance_max': None,
                                 'self_neighborhood': True,
                                 },
            n_samples=100,
            Dmax=Dmax_aa, padded=False, from_triplets=True, order=order_aa, dipole=dipole_aa, n_init=n_init)
        if save_initial_values:
            io_utils.save_pickle(
                {'GaussianKernel_aa': initial_values['GaussianKernel_aa']}, location_aa)

    if with_atom:
        print(
            'Initializing the Gaussian kernels for the atomic neighborhood (takes a few minutes to do it robustly, be patient!). Reduce n_init from 10 to 1 if speed needed')
        c_atom = ''.join([c[0] for c in coordinates_atom])
        location_atom = initial_values_folder + 'initial_GaussianKernel_atom_N_%s_Kmax_%s_Dmax_%s_frames_%s_coords_%s_nrotations_%s_cov_%s_tripletsorder_%s_dipole_%s.data' % (
            N_atom, K_atom, Dmax_atom, frame_atom, c_atom, 1, covariance_type_atom, order_atom, dipole_atom)

        try:
            assert fresh_initial_values == False
            initial_values['GaussianKernel_atom'] = io_utils.load_pickle(location_atom)[
                'GaussianKernel_atom']
        except:
            if 'index_distance' in coordinates_atom:
                inputs_ = [inputs[4], inputs[7], inputs[6]]
            else:
                inputs_ = [inputs[4], inputs[7]]
            initial_values['GaussianKernel_atom'] = neighborhoods.initialize_GaussianKernel_for_NeighborhoodEmbedding(
                inputs_,
                N_atom,
                covariance_type_atom,
                neighborhood_params={'Kmax': K_atom,
                                     'coordinates': coordinates_atom,
                                     'index_distance_max': None,
                                     'self_neighborhood': True
                                     },
                n_samples=100,
                Dmax=Dmax_atom, padded=False, from_triplets=True, order=order_atom, dipole=dipole_atom, n_init=n_init)
            if save_initial_values:
                io_utils.save_pickle(
                    {'GaussianKernel_atom': initial_values['GaussianKernel_atom']}, location_atom)

    c_graph = ''.join([c[0] for c in coordinates_graph])
    location_graph = initial_values_folder + 'initial_GaussianKernel_graph_N_%s_%s_Kmax_%s_Dmax_%s_coords_%s_indexmax_%s_cov_%s_tripletsorder_%s_dipole_%s.data' % (
        N_graph, nembedding_graph, K_graph, Dmax_graph, c_graph, index_distance_max_graph, covariance_type_graph,
        order_aa, dipole_aa)

    try:
        assert fresh_initial_values == False
        initial_values_graph = io_utils.load_pickle(location_graph)
    except:
        print(
            'Initializing the Gaussian and dense kernels for the graph neighborhood (takes a few minutes to do it robustly, be patient!)  Reduce n_init from 10 to 1 if speed needed')
        initial_values_graph = neighborhoods.initialize_Embedding_for_NeighborhoodAttention(
            [inputs[0], inputs[3], inputs[2]], outputs,
            neighborhood_params={'Kmax': K_graph,
                                 'coordinates': coordinates_graph,
                                 'index_distance_max': index_distance_max_graph,
                                 'self_neighborhood': True
                                 },
            N=N_graph, dense=nembedding_graph, covariance_type=covariance_type_graph,
            Dmax=Dmax_graph,
            epochs=10,
            padded=False,
            from_triplets=True,
            n_samples=100, order=order_aa, dipole=dipole_aa, n_init=n_init)
        if save_initial_values:
            io_utils.save_pickle(initial_values_graph, location_graph)
    initial_values['GaussianKernel_graph'] = initial_values_graph['graph_embedding_GaussianKernel']
    initial_values['dense_graph'] = initial_values_graph['graph_embedding_dense']

    model = wrappers.grouped_Predictor_wrapper(ScanNet,
                                               with_atom=with_atom,
                                               Lmax_aa=Lmax_aa,
                                               Lmax_atom=Lmax_atom,
                                               Lmax_aa_points=Lmax_aa_points,
                                               Lmax_atom_points=Lmax_atom_points,
                                               K_aa=K_aa,
                                               K_atom=K_atom,
                                               K_graph=K_graph,
                                               N_aa=N_aa,
                                               N_atom=N_atom,
                                               N_graph=N_graph,
                                               nfeatures_aa=nfeatures_aa,
                                               nfeatures_atom=nfeatures_atom,
                                               nembedding_atom=nembedding_atom,
                                               nembedding_aa=nembedding_aa,
                                               nembedding_graph=nembedding_graph,
                                               dense_pooling=dense_pooling,
                                               nattentionheads_pooling=nattentionheads_pooling,
                                               nfilters_aa=nfilters_aa,
                                               nfilters_atom=nfilters_atom,
                                               nfilters_graph=nfilters_graph,
                                               nattentionheads_graph=nattentionheads_graph,
                                               filter_MLP=filter_MLP,
                                               initial_values=initial_values,
                                               covariance_type_aa=covariance_type_aa,
                                               covariance_type_atom=covariance_type_atom,
                                               covariance_type_graph=covariance_type_graph,
                                               activation=activation,
                                               frame_aa=frame_aa,
                                               coordinates_aa=coordinates_aa,
                                               coordinates_atom=coordinates_atom,
                                               coordinates_graph=coordinates_graph,
                                               index_distance_max_graph=index_distance_max_graph,
                                               l1_aa=l1_aa,
                                               l12_aa=l12_aa,
                                               l12group_aa=l12group_aa,
                                               l1_atom=l1_atom,
                                               l12_atom=l12_atom,
                                               l12group_atom=l12group_atom,
                                               l1_pool=l1_pool,
                                               l12_pool=l12_pool,
                                               l12group_pool=l12group_pool,
                                               nrotations=nrotations,
                                               dropout=dropout,
                                               optimizer=optimizer,
                                               input_type=input_type,
                                               Lmaxs=Lmaxs,
                                               multi_inputs=True,
                                               multi_outputs=False,
                                               )
    extra_params = {'epochs': epochs, 'batch_size': batch_size}
    return model, extra_params





if __name__ == '__main__':
    import matplotlib
    matplotlib.use('module://backend_interagg')
    import matplotlib.pyplot as plt
    import Bio.PDB
    from preprocessing import PDBio,pipelines
    import numpy as np

    with_atom = True
    frames_aa = 'triplet_sidechain'

    pipeline = pipelines.ScanNetPipeline(
                                                        with_aa=True,
                                                        with_atom=with_atom,
                                                        aa_features='sequence',
                                                        atom_features='valency',
                                                        aa_frames=frames_aa,
                                                        )

    PDB_folder = '/Users/jerometubiana/PDB/'
    label_file = '/Users/jerometubiana/Downloads/interface_labels_train.txt'

    # PDB_folder = '/home/iscb/wolfson/jeromet/Data/PDB_files/'
    # label_file = '/home/iscb/wolfson/jeromet/Data/PPI_binding_database/CATH/interface_labels_train.txt'

    pdblist = Bio.PDB.PDBList()
    nmax = 11
    list_origins, list_sequences, list_resids, list_labels = io_utils.read_labels(label_file, nmax=nmax,
                                                                                           label_type='int')

    inputs = []
    for origin in list_origins:
        pdb = origin[:4]
        chain = origin.split('_')[-1]
        name = pdblist.retrieve_pdb_file(pdb, pdir=PDB_folder)
        struct, chains = PDBio.load_chains(pdb_id=pdb, chain_ids=[(0, chain)], file=PDB_folder + '%s.cif' % pdb)
        inputs.append(pipeline.process_example(chains))
    inputs = [np.array([input[k] for input in inputs])
              for k in range(len(inputs[0]))]
    outputs = [np.stack([label < 5, label >= 5], axis=-1) for label in list_labels]

    Lmax_aa = max([len(o) for o in outputs])
    K_aa = 8
    K_atom = 8
    K_graph = 16
    Dmax_aa = 13.
    Dmax_atom = 4.
    Dmax_graph = 13.
    N_aa = 16
    N_atom = 16
    N_graph = 16
    nfeatures_aa = 20
    nfeatures_atom = 12
    nembedding_atom = 4
    nembedding_aa = 8
    nembedding_graph = 1
    dense_pooling = None
    nattentionheads_pooling = 2
    nfilters_atom = 32
    nfilters_aa = 32
    nfilters_graph = 2
    nattentionheads_graph = 1
    filter_MLP = []
    covariance_type_atom = 'full'
    covariance_type_aa = 'full'
    covariance_type_graph = 'full'
    activation = 'relu'
    coordinates_atom = ['euclidian']
    coordinates_aa = ['euclidian']  # ,'dipole_spherical']
    frames_aa = frames_aa
    coordinates_graph = ['distance', 'ZdotZ', 'ZdotDelta', 'index_distance']
    index_distance_max_graph = 16
    l1_aa = 0.
    l12_aa = 0.
    l12group_aa = 1e-3
    l1_atom = 0.
    l12_atom = 1e-3
    l12group_atom = 0.
    nrotations = 1
    dropout = 0.
    optimizer = 'adam'

    model, extra_params = initialize_ScanNet(
        inputs,
        outputs,
        with_atom=with_atom,
        Lmax_aa=Lmax_aa,
        K_aa=K_aa,
        K_atom=K_atom,
        K_graph=K_graph,
        Dmax_aa=Dmax_aa,
        Dmax_atom=Dmax_atom,
        Dmax_graph=Dmax_graph,
        N_aa=N_aa,
        N_atom=N_atom,
        N_graph=N_graph,
        nfeatures_aa=nfeatures_aa,
        nfeatures_atom=nfeatures_atom,
        nembedding_atom=nembedding_atom,
        nembedding_aa=nembedding_aa,
        nembedding_graph=nembedding_graph,
        dense_pooling=dense_pooling,
        nattentionheads_pooling=nattentionheads_pooling,
        nfilters_atom=nfilters_atom,
        nfilters_aa=nfilters_aa,
        nfilters_graph=nfilters_graph,
        nattentionheads_graph=nattentionheads_graph,
        filter_MLP=filter_MLP,
        covariance_type_atom=covariance_type_atom,
        covariance_type_aa=covariance_type_aa,
        covariance_type_graph=covariance_type_graph,
        activation=activation,
        coordinates_atom=coordinates_atom,
        coordinates_aa=coordinates_aa,
        frame_aa=frames_aa,
        coordinates_graph=coordinates_graph,
        index_distance_max_graph=index_distance_max_graph,
        l1_aa=l1_aa,
        l12_aa=l12_aa,
        l12group_aa=l12group_aa,
        l1_atom=l1_atom,
        l12_atom=l12_atom,
        l12group_atom=l12group_atom,
        nrotations=nrotations,
        dropout=dropout,
        optimizer=optimizer, batch_size=1,
        fresh_initial_values=True,
        save_initial_values=False, n_init=1
    )

    model.fit(inputs, outputs, batch_size=1, epochs=10)
