from keras.engine.base_layer import Layer
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.layers import Input, Masking, Dense
from keras.models import Model, Sequential
from utilities import wrappers
from preprocessing.pipelines import padd_matrix
from network.embeddings import GaussianKernel,initialize_GaussianKernel
from preprocessing import PDB_processing


def distance_pcs(cloud1,cloud2,squared=False):
    distance = (tf.expand_dims(cloud1[...,0],axis=-1) - tf.expand_dims(cloud2[...,0],axis=-2) )**2
    distance += (tf.expand_dims(cloud1[...,1],axis=-1) - tf.expand_dims(cloud2[...,1],axis=-2) )**2
    distance += (tf.expand_dims(cloud1[...,2],axis=-1) - tf.expand_dims(cloud2[...,2],axis=-2) )**2
    if not squared:
        distance = tf.sqrt(distance)
    return distance



class FrameBuilder(Layer):
    def __init__(self, order='1', dipole=False, **kwargs):
        super(FrameBuilder, self).__init__(**kwargs)
        self.support_masking = True
        self.epsilon = tf.constant(1e-6)
        self.order = order
        self.dipole = dipole
        return

    def build(self, input_shape):
        self.xaxis = tf.constant(np.array([[1, 0, 0]], dtype=np.float32), name='xaxis')
        self.yaxis = tf.constant(np.array([[0, 1, 0]], dtype=np.float32), name='yaxis')
        self.zaxis = tf.constant(np.array([[0, 0, 1]], dtype=np.float32), name='zaxis')

        super(FrameBuilder, self).build(input_shape)

    def call(self, inputs, mask=None):

        points, triplets = inputs
        '''
        For each atom, four cases should be distinguished.
        Case 1: both neighbors exist (i.e. at least two covalent bonds). Construct the frame as usual using Schmidt orthonormalization.
        Case 2: The first neighbor does not exist (i.e. the next atom along the protein tree. Example: Alanine, A_i =Cbeta A_{i-1} = Calpha, A_{i+1} = Cgamma does not exists). 
        The solution is to place a virtual atom such that (A_{i-2}, A_{i-1}, A_{i}, A_{i+1,virtual}) is a parallelogram. For alanine, (N,Calpha,Cbeta,Cgamma_virt) is a parallelogram.
        Case 3: The second neighbor does not exist (i.e. the previous atom along the protein tree.
        Example: N-terminal N along the backbone, or missing residues). Similarly, we build a parallelogram
        (A_{i-1,virtual}, A_i, A_{i+1},A_{i+2}).
        Case 4: None exist (missing atoms). Use the default cartesian frame.
        '''

        triplets = tf.clip_by_value(triplets, 0, tf.shape(points)[-2]-1)


        delta_10 = tf.gather_nd(points, triplets[:, :, 1:2], batch_dims=1) - tf.gather_nd(points, triplets[:, :, 0:1],
                                                                                          batch_dims=1)
        delta_20 = tf.gather_nd(points, triplets[:, :, 2:3], batch_dims=1) - tf.gather_nd(points, triplets[:, :, 0:1],
                                                                                          batch_dims=1)
        if self.order in ['2','3']: # Order 1: the second point is on the zaxis and the third in the xz plane. Order 2: the third point is on the zaxis and the second in the xz plane.
            delta_10,delta_20 = delta_20,delta_10

        centers = tf.gather_nd(points, triplets[:, :, 0:1], batch_dims=1)
        zaxis = (delta_10 + self.epsilon * tf.reshape(self.zaxis,[1,1,3])) / (tf.sqrt(tf.reduce_sum(delta_10 ** 2, axis=-1, keepdims=True) ) + self.epsilon)

        yaxis = tf.linalg.cross(zaxis, delta_20)
        yaxis = (yaxis + self.epsilon * tf.reshape(self.yaxis,[1,1,3]) ) / (tf.sqrt(tf.reduce_sum(yaxis ** 2, axis=-1, keepdims=True) ) + self.epsilon)

        xaxis = tf.linalg.cross(yaxis, zaxis)
        xaxis = (xaxis + self.epsilon * tf.reshape(self.xaxis,[1,1,3]) ) / (tf.sqrt(tf.reduce_sum(xaxis ** 2, axis=-1, keepdims=True)) + self.epsilon)

        if self.order == '3':
            xaxis,yaxis,zaxis = zaxis,xaxis,yaxis

        if self.dipole:
            dipole = tf.gather_nd(points, triplets[:, :, 3:4], batch_dims=1) - tf.gather_nd(points, triplets[:, :, 0:1],batch_dims=1)
            dipole = (dipole + self.epsilon * tf.reshape(self.zaxis,[1,1,3])) / (tf.sqrt(tf.reduce_sum(dipole ** 2, axis=-1, keepdims=True) ) + self.epsilon)
            frames = tf.stack([centers,xaxis,yaxis,zaxis,dipole],axis=-2)
        else:
            frames = tf.stack([centers,xaxis,yaxis,zaxis],axis=-2)

        if mask not in [None,[None,None]]:
            frames *= tf.expand_dims(tf.expand_dims(tf.cast(mask[-1],tf.float32),axis=-1),axis=-1)
        return frames

    def compute_output_shape(self, input_shape):
        if self.dipole:
            num_vectors = 5
        else:
            num_vectors = 4
        output_shape = (input_shape[1][0], input_shape[1][1], num_vectors, 3)
        return output_shape

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        if self.dipole:
            num_vectors = 5
        else:
            num_vectors = 4
        if mask not in [None, [None, None]]:
            return tf.tile(tf.expand_dims(mask[1], axis=-1),[1,1,num_vectors])
        else:
            return mask


def distance(coordinates1,coordinates2,squared=False,ndims=3):
    D = (tf.expand_dims(coordinates1[...,0],axis=-1) - tf.expand_dims(coordinates2[...,0],axis=-2) )**2
    for n in range(1,ndims):
        D += (tf.expand_dims(coordinates1[..., n], axis=-1) - tf.expand_dims(coordinates2[..., n], axis=-2)) ** 2
    if not squared:
        D = tf.sqrt(D)
    return D

def euclidian_to_spherical(x,return_r=True,cut='2pi',eps=1e-8):
    r = tf.sqrt( tf.reduce_sum(x**2,axis=-1) )
    theta = tf.acos(x[...,-1]/(r+eps) )
    phi = tf.atan2( x[...,1],x[...,0]+eps)
    if cut == '2pi':
        phi = phi + tf.cast(tf.greater(0.,phi), tf.float32) * (2 * np.pi)
    if return_r:
        return tf.stack([r,theta,phi],axis=-1)
    else:
        return tf.stack([theta, phi], axis=-1)


class LocalNeighborhood(Layer):
    def __init__(self, Kmax=10, coordinates=['euclidian'],self_neighborhood=True,index_distance_max=None, nrotations = 1,**kwargs):
        super(LocalNeighborhood, self).__init__(**kwargs)
        self.Kmax = Kmax
        self.coordinates = coordinates
        self.self_neighborhood = self_neighborhood
        self.support_masking = True

        for coordinate in self.coordinates:
            assert coordinate in ['distance','index_distance',
                                  'euclidian','ZdotZ','ZdotDelta','dipole_spherical'
                                  ]

        self.first_format = []
        self.second_format = []
        if ('euclidian' in self.coordinates) | ('ZdotZ' in self.coordinates) | ('ZdotDelta' in self.coordinates):
            self.first_format.append('frame')
            if self.self_neighborhood | ('ZdotZ' in self.coordinates) | ('ZdotDelta' in self.coordinates):
                self.second_format.append('frame')
            else:
                self.second_format.append('point')

        elif 'distance' in self.coordinates:
            self.first_format.append('point')
            self.second_format.append('point')
        if 'index_distance' in self.coordinates:
            self.first_format.append('index')
            self.second_format.append('index')

        coordinates_dimension = 0
        for coordinate in coordinates:
            if coordinate == 'euclidian':
                coordinates_dimension+=3
            elif coordinate =='dipole_spherical':
                coordinates_dimension += 2
            elif coordinate == 'ZdotDelta':
                coordinates_dimension+=2
            else:
                coordinates_dimension+=1

        self.coordinates_dimension = coordinates_dimension

        self.index_distance_max = index_distance_max
        self.epsilon = tf.constant(1e-10)
        self.big_distance = tf.constant(1000.)
        self.nrotations = nrotations
        if self.nrotations > 1:
            assert self.coordinates == ['euclidian'], 'Rotations only work with Euclidian coordinates'
        return

    def build(self, input_shape):
        self.nattributes = len(input_shape) - len(self.first_format) - (1-1*self.self_neighborhood) * len(self.second_format)
        if self.nrotations > 1:
            phis = np.arange(self.nrotations) / self.nrotations * 2 * np.pi
            rotations = np.zeros([self.nrotations, 3, 3], dtype=np.float32)
            rotations[:, 0, 0] = np.cos(phis)
            rotations[:, 1, 1] = np.cos(phis)
            rotations[:, 1, 0] = np.sin(phis)
            rotations[:, 0, 1] = -np.sin(phis)
            rotations[:, 2, 2] = 1
            self.rotations = tf.constant(rotations)
        super(LocalNeighborhood, self).build(input_shape)


    def call(self, inputs, mask=None):

        if mask is None:
            mask = [None for _ in inputs]

        if 'frame' in self.first_format:
            first_frame = inputs[self.first_format.index('frame')]
        else:
            first_frame = None
        if 'frame' in self.second_format:
            if self.self_neighborhood:
                second_frame = first_frame
            else:
                second_frame = inputs[len(self.first_format)+self.second_format.index('frame')]
        else:
            second_frame = None

        if 'point' in self.first_format:
            first_point = inputs[self.first_format.index('point')]
        else:
            first_point = None
        if 'point' in self.second_format:
            if self.self_neighborhood:
                second_point = first_point
            else:
                second_point = inputs[len(self.first_format)+self.second_format.index('point')]
        else:
            second_point = None

        if 'index' in self.first_format:
            first_index = inputs[self.first_format.index('index')]
        else:
            first_index = None
        if 'index' in self.second_format:
            if self.self_neighborhood:
                second_index = first_index
            else:
                second_index = inputs[len(self.first_format)+self.second_format.index('index')]
        else:
            second_index = None

        second_attributes = inputs[-self.nattributes:]

        first_mask = mask[0]
        if (first_mask is not None) & (first_frame is not None):
            first_mask = first_mask[:,:,1]
        if self.self_neighborhood:
            second_mask = first_mask
        else:
            second_mask = mask[len(self.first_format)]
            if (second_mask is not None) & (second_frame is not None):
                second_mask = second_mask[:, :, 1]
        if second_mask is not None:
            irrelevant_seconds = tf.expand_dims(1 - tf.cast(second_mask, tf.float32), axis=1)
        else:
            irrelevant_seconds = None

        if first_frame is not None:
            first_center = first_frame[:,:,0]
            ndims = 3
        elif first_point is not None:
            first_center = first_point
            ndims = 3
        else:
            first_center = tf.cast(first_index,dtype='float32')
            ndims = 1

        if second_frame is not None:
            second_center = second_frame[:,:,0]
        elif second_point is not None:
            second_center = second_point
        else:
            second_center = tf.cast(second_index,dtype='float32')

        distance_square = distance(first_center, second_center, squared=True,ndims=ndims)
        if irrelevant_seconds is not None:
            distance_square += irrelevant_seconds * self.big_distance

        neighbors = tf.expand_dims(tf.argsort(distance_square)[:, :, :self.Kmax], axis=-1)

        neighbors_attributes = [tf.gather_nd(
            attribute, neighbors, batch_dims=1) for attribute in second_attributes]

        neighbor_coordinates = []

        if 'euclidian' in self.coordinates:
            euclidian_coordinates = tf.reduce_sum(tf.expand_dims(
                # B x Lmax x Kmax x 3
                tf.gather_nd(second_center, neighbors, batch_dims=1)
                - tf.expand_dims(first_center, axis=-2),  # B X Lmax X 1 X 3,
                axis=-2)  # B X Lmax X Kmax X 1 X 3 \
                * tf.expand_dims(
                first_frame[:,:,1:4],
                axis=-3)  # B X Lmax X 1 X 3 X 3
                , axis=-1)  # B X Lmax X Kmax X 3
            if self.nrotations>1:
                euclidian_coordinates = K.dot(
                euclidian_coordinates, self.rotations)

                neighbors_attributes = [tf.expand_dims(
                    neighbors_attribute, axis=-2) for neighbors_attribute in neighbors_attributes]

            neighbor_coordinates.append(euclidian_coordinates)

        if 'dipole_spherical' in self.coordinates:
            dipole_euclidian_coordinates = tf.reduce_sum(tf.expand_dims(
                # B x Lmax x Kmax x 3
                tf.gather_nd(second_frame[:,:,-1], neighbors, batch_dims=1),
                axis=-2)  # B X Lmax X Kmax X 1 X 3 \
                * tf.expand_dims(
                first_frame[:,:,1:4],
                axis=-3)  # B X Lmax X 1 X 3 X 3
                , axis=-1)  # B X Lmax X Kmax X 3
            dipole_spherical_coordinates = euclidian_to_spherical(dipole_euclidian_coordinates,return_r=False)
            neighbor_coordinates.append(dipole_spherical_coordinates)

        if 'distance' in self.coordinates:
            distance_neighbors = tf.expand_dims(tf.sqrt(tf.gather_nd(
                distance_square, neighbors, batch_dims=2) ), axis=-1)
            neighbor_coordinates.append(distance_neighbors)

        if 'ZdotZ' in self.coordinates:
            first_zdirection = first_frame[:,:,-1]
            second_zdirection = second_frame[:, :, -1]

            ZdotZ_neighbors = tf.reduce_sum(tf.expand_dims(
                first_zdirection, axis=-2) * tf.gather_nd(second_zdirection, neighbors, batch_dims=1), axis=-1, keepdims=True)
            neighbor_coordinates.append(ZdotZ_neighbors)

        if 'ZdotDelta' in self.coordinates:
            first_zdirection = first_frame[:,:,-1]
            second_zdirection = second_frame[:, :, -1]

            DeltaCenter_neighbors = (tf.gather_nd(
                second_center, neighbors, batch_dims=1) - tf.expand_dims(first_center, axis=-2)) / (distance_neighbors + self.epsilon)
            ZdotDelta_neighbors = tf.reduce_sum(tf.expand_dims(
                first_zdirection, axis=-2) * DeltaCenter_neighbors, axis=-1, keepdims=True)
            DeltadotZ_neighbors = tf.reduce_sum(DeltaCenter_neighbors * tf.gather_nd(
                second_zdirection, neighbors, batch_dims=1), axis=-1, keepdims=True)
            neighbor_coordinates.append(DeltadotZ_neighbors)
            neighbor_coordinates.append(ZdotDelta_neighbors)

        if 'index_distance' in self.coordinates:
            index_distance = tf.abs(tf.cast(
                tf.expand_dims(first_index,axis=-2) - tf.gather_nd(second_index, neighbors, batch_dims=1),tf.float32) )

            if self.index_distance_max is not None:
                index_distance = tf.clip_by_value(index_distance, 0, self.index_distance_max)

            neighbor_coordinates.append(index_distance)


        neighbor_coordinates = tf.concat(neighbor_coordinates, axis=-1)

        if first_mask is not None:
            if (self.nrotations > 1):
                neighbor_coordinates *= tf.expand_dims(tf.expand_dims(tf.expand_dims(
                    tf.cast(first_mask, tf.float32), axis=-1),axis=-1),axis=-1)

                for neighbors_attribute in neighbors_attributes:

                    neighbors_attribute *= tf.expand_dims(tf.expand_dims(tf.expand_dims(
                        tf.cast(first_mask, tf.float32), axis=-1),axis=-1),axis=-1)
            else:
                neighbor_coordinates *= tf.expand_dims(tf.expand_dims(
                    tf.cast(first_mask, tf.float32), axis=-1),axis=-1)

                for neighbors_attribute in neighbors_attributes:

                    neighbors_attribute *= tf.expand_dims(tf.expand_dims(
                        tf.cast(first_mask, tf.float32), axis=-1),axis=-1)

        output = [neighbor_coordinates] + neighbors_attributes
        return output

    def compute_output_shape(self, input_shape):
        B = input_shape[0][0]
        Lmax = input_shape[0][1]
        dim_attributes = [shape[-1] for shape in input_shape[-self.nattributes:]]
        if self.nrotations>1:
            output_shape = [(B, Lmax, self.Kmax, self.nrotations,self.coordinates_dimension)]
        else:
            output_shape = [(B, Lmax, self.Kmax, self.coordinates_dimension)]
        for dim_attribute in dim_attributes:
            if self.nrotations>1:
                output_shape.append(
                    (B, Lmax, self.Kmax, 1, dim_attribute)
                )
            else:
                output_shape.append(
                    (B, Lmax, self.Kmax, dim_attribute)
                )
        return output_shape

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        if mask not in [None, [None for _ in inputs]]:
            first_mask = mask[0]
            if 'frame' in self.first_format:
                first_mask = first_mask[:, :, 1]

            if self.nrotations>1:
                return [tf.expand_dims(tf.expand_dims(first_mask,axis=-1),axis=-1) ]* (1+ self.nattributes)
            else:
                return [tf.expand_dims(first_mask,axis=-1) ] * (1+self.nattributes)
        else:
            return mask

    def get_config(self):
        config = {'Kmax': self.Kmax,
                  'coordinates': self.coordinates,
                  'index_distance_max': self.index_distance_max,
                  'self_neighborhood':self.self_neighborhood,
                  'coordinates_dimension': self.coordinates_dimension,
                  'nrotations':self.nrotations
                  }
        base_config = super(
            LocalNeighborhood, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




def get_LocalNeighborhood(inputs,neighborhood_params,flat=False,n_samples=100,padded=False,attributes=None):
    B = len(inputs[0])
    if n_samples is not None:
        b = min(n_samples, B)
    else:
        b = B

    if padded:
        Lmaxs = [inputs_.shape[1] for inputs_ in inputs]
        inputs = [inputs_[:b] for inputs_ in inputs]
        if attributes is not None:
            attributes = attributes[:b]
    else:
        Lmaxs = [ max([len(input_) for input_ in inputs_[:b]] ) for inputs_ in  inputs ]
        inputs = [
        np.stack(
        [padd_matrix(input,Lmax=Lmax,padding_value=0) for input in inputs_[:b]],
        axis =0
        )
        for Lmax,inputs_ in zip(Lmaxs,inputs)

        ]
        if attributes is not None:
            attributes = np.stack([padd_matrix(attribute,Lmax=Lmaxs[0],padding_value=0.) for attribute in attributes[:b]],
                                  axis=0)
    if attributes is not None:
        inputs.append(attributes)
    else:
        inputs.append(  np.ones([b, Lmaxs[0], 1], dtype=np.float32) )

    keras_inputs  = [Input(shape=inputs_.shape[1:]) for inputs_ in inputs]
    masked_keras_inputs = [Masking()(keras_inputs_) for keras_inputs_ in keras_inputs]
    local_coordinates, local_attributes = LocalNeighborhood(
        **neighborhood_params)(masked_keras_inputs)

    first_layer = Model(
        inputs=keras_inputs, outputs=[local_coordinates,local_attributes])

    local_coordinates,local_attributes = first_layer.predict(inputs, batch_size=10)

    if flat:
        d = local_coordinates.shape[-1]
        nattributes = local_attributes.shape[-1]
        local_coordinates = local_coordinates[local_coordinates.max(
            -1).max(-1) > 0].reshape([-1, d])
        local_attributes = local_attributes[local_attributes.max(-1)>0].reshape([-1,nattributes])
    if attributes is not None:
        return local_coordinates,local_attributes
    else:
        return local_coordinates


def get_Frames(inputs,n_samples=None,padded=False,order='1',dipole=False,Lmax=None):
    B = len(inputs[0])
    if n_samples is not None:
        b = min(n_samples, B)
    else:
        b = B

    nindices = inputs[0][0].shape[-1]

    if padded:
        triplets_ = inputs[0][:b]
        clouds_ = inputs[1][:b]
        Lmax = inputs[0].shape[1]
        Lmax2 = inputs[1].shape[1]
    else:
        if Lmax is not None:
            Lmax = min(max([len(input_) for input_ in inputs[0][:b]] ), Lmax)
        else:
            Lmax = max([len(input_) for input_ in inputs[0][:b]] )
        Lmax2 = max([len(input_) for input_ in inputs[1][:b]] )
        Ls = [len(x) for x in inputs[0][:b]]
        triplets_ = np.zeros([b,Lmax, nindices ],dtype=np.int32)
        clouds_ = np.zeros([b,Lmax2,3],dtype=np.float32)

        for b_ in range(b):
            padd_matrix(inputs[0][b_], padded_matrix=triplets_[b_], padding_value=-1)
            padd_matrix(inputs[1][b_], padded_matrix=clouds_[b_], padding_value=0)

    inputs_ = [triplets_,clouds_]

    triplets = Input(shape=[Lmax, nindices], dtype="int32",name='triplets')
    clouds = Input(shape=[Lmax2, 3], dtype="float32",name='clouds')
    masked_triplets = Masking(mask_value=-1, name='masked_triplets')(triplets)
    masked_clouds = Masking(mask_value=0.0, name='masked_clouds')(clouds)
    frames = FrameBuilder(name='frames',order=order,dipole=dipole)([masked_clouds,masked_triplets])
    first_layer = Model(
        inputs=[triplets,clouds], outputs=frames)
    frames_ = first_layer.predict(inputs_)
    if not padded:
        frames_ = wrappers.truncate_list_of_arrays(frames_,Ls)
    return frames_



def initialize_GaussianKernel_for_NeighborhoodEmbedding(
        inputs, N,
        covariance_type='diag',
        neighborhood_params = {'Kmax':10,'coordinates':['euclidian'],'nrotations':1,'index_distance_max':None,'self_neighborhood':True},
        from_triplets = False, Dmax = None, n_samples=None,padded=True,order='1',dipole=False,n_init=10):

    if from_triplets:
        frames = get_Frames(inputs[:2],n_samples=n_samples,padded=padded,order=order,dipole=dipole)
        inputs = [frames] + inputs[2:]

    local_coordinates = get_LocalNeighborhood(inputs, neighborhood_params, flat=True, n_samples=n_samples,padded=padded)

    if Dmax is not None:
        if 'euclidian' in neighborhood_params['coordinates']:
            d = np.sqrt((local_coordinates[:,:3]**2).sum(-1))
        else:
            d = local_coordinates[:,0]
        local_coordinates = local_coordinates[d <= Dmax]
    if 'index_distance' in neighborhood_params['coordinates']:
        reg_covar = 1e0
    else:
        reg_covar= 1e-2
    return initialize_GaussianKernel(local_coordinates, N,covariance_type=covariance_type,reg_covar=reg_covar,n_init=n_init)



def initialize_Embedding_for_NeighborhoodAttention(
        inputs, labels,N=16,covariance_type='full',dense=None,
            nsamples=100,
            epochs=10,
        neighborhood_params={
            'Kmax': 32,
            'coordinates': ['distance', 'ZdotZ', 'ZdotDelta', 'index_distance'],
            'nrotations': 1,
            'index_distance_max': 16,
            'self_neighborhood': True},
        from_triplets=False, n_samples=None, padded=True,Dmax=None,order='1',dipole=False,n_init=10):
    '''
    labels in binary format.
    '''
    if nsamples is not None:
        inputs = [input[:nsamples] for input in inputs]
        labels = labels[:nsamples]

    Ls = [label.shape[0] for label in labels]
    if padded:
        inputs = wrappers.truncate_list_of_arrays(inputs, Ls)
        labels = wrappers.truncate_list_of_arrays(labels, Ls)


    if from_triplets:
        frames = get_Frames(inputs[:2],order=order,dipole=dipole, n_samples=n_samples, padded=False)
        frames = wrappers.truncate_list_of_arrays(frames, Ls)
        inputs = [frames] + inputs[2:]

    mu_labels = np.concatenate(labels,axis=0)[:,-1].mean()
    local_coordinates,local_attributes = get_LocalNeighborhood(inputs,neighborhood_params,flat=False,padded=False,attributes=labels)
    local_coordinates = np.concatenate(wrappers.truncate_list_of_arrays(local_coordinates, Ls), axis=0)
    local_attributes = np.concatenate(wrappers.truncate_list_of_arrays(local_attributes, Ls),axis=0)
    features = local_coordinates.reshape([-1,local_coordinates.shape[-1]] )
    target = ( (local_attributes[:,:1,-1] * local_attributes[:,:,-1]).flatten() - mu_labels**2)/(mu_labels - mu_labels**2)
    if Dmax is not None:
        mask = features[:,0] < Dmax
        features = features[mask]
        target = target[mask]
    initial_values = initialize_GaussianKernel(features, N, covariance_type=covariance_type,n_init=n_init)

    model = Sequential()
    model.add(GaussianKernel(N, covariance_type=covariance_type,
                      initial_values=initial_values, name='graph_embedding_GaussianKernel'))
    if dense is not None:
        model.add(Dense(dense, activation='tanh',
                        name='graph_embedding_dense', use_bias=False))
        model.add(Dense(1, activation=None,use_bias=False, name='graph_embedding_dense_final'))
    else:
        model.add(Dense(1, activation=None, use_bias=False, name='graph_embedding_dense'))

    model.compile(loss='MSE', optimizer='adam')
    model.fit(features, target, epochs=epochs, batch_size=1024)
    model_params = dict([(layer.name, layer.get_weights())
                         for layer in model.layers])
    return model_params






if __name__ == '__main__':
# %%
    import matplotlib
    matplotlib.use('module://backend_interagg')
    import matplotlib.pyplot as plt
    import PDB_utils2
    import Bio.PDB
    import pipelines
    import numpy as np
    import wrappers
    import format_dockground



    with_atom = True
    aa_frames = 'quadruplet'
    order = '3'
    dipole = True

    pipeline = pipelines.ScanNetPipeline(
                                                        with_aa=True,
                                                        with_atom=with_atom,
                                                        aa_features='sequence',
                                                        atom_features='type',
                                                        aa_frames=aa_frames,
    )


    PDB_folder = '/Users/jerometubiana/PDB/'
    pdblist = Bio.PDB.PDBList()
    # list_pdbs = ['11as_A',
    #              '137l_B',
    #              '13gs_A',
    #              '1a05_A',
    #              '1a09_A',
    #              '1a0d_A',
    #              '1a0e_A',
    #              '1a0f_A',
    #              '1a0g_B',
    #              '1a0o_B']

    nmax = 10

    list_origins, list_sequences,list_resids,list_labels = format_dockground.read_labels('/Users/jerometubiana/Downloads/interface_labels_train.txt',nmax=nmax,label_type='int')

    inputs = []
    for origin in list_origins:
        pdb = origin[:4]
        chain = origin.split('_')[-1]
        name = pdblist.retrieve_pdb_file(pdb, pdir=PDB_folder)
        struct, chains = PDB_processing.load_chains(pdb_id=pdb, chain_ids=[(0, chain)], file=PDB_folder + '%s.cif' % pdb)
        inputs.append(pipeline.process_example(chains))
    inputs = [np.array([input[k] for input in inputs])
              for k in range(len(inputs[0]))]
    outputs = [ np.stack([label <5,label >=5],axis = -1) for label in list_labels]


    frames = get_Frames([inputs[0],inputs[3]],order=order,dipole=dipole) # Valide 24/01/2021

    plt.plot(frames[0][:,0,:]); plt.show() # Check centers.
    plt.plot(frames[0][:, 1, :]); plt.show()  # Check unit vectors.
    for i in range(3): # Check orthonormality.
        for j in range(3):
            print('Dot product',i,j , np.abs( (frames[0][:, 1+i, :] * frames[0][:,1+j,:]).sum(-1)  ).max() )

    local_coordinates = get_LocalNeighborhood([frames],padded=False,neighborhood_params={
        'coordinates': ['euclidian','dipole_spherical'],
        'Kmax':10
    })

    plt.hist( local_coordinates.flatten(),bins=100 ); plt.show() # Check scales.
    plt.matshow(np.sqrt( (local_coordinates[0]**2).sum(-1) ) ,aspect='auto'); plt.colorbar(); plt.show() # Check order and padding.
#%%
    local_coordinates = get_LocalNeighborhood([frames,inputs[2]],padded=False,neighborhood_params={
        'coordinates': ['distance', 'ZdotZ', 'ZdotDelta', 'index_distance'],
        'Kmax':10,
        'index_distance_max': 16,
    })

    plt.matshow(local_coordinates[0,:,:,0],aspect='auto'); plt.colorbar(); plt.show()
    for i in range(local_coordinates.shape[-1]):
        plt.hist(local_coordinates[...,i].flatten(),bins=100); plt.show()

    plt.matshow(local_coordinates[0,:,:,-1],aspect='auto'); plt.colorbar(); plt.show()

#%%

    params = initialize_GaussianKernel_for_NeighborhoodEmbedding(
        [inputs[0],inputs[3]], 16,
        covariance_type='diag',
        neighborhood_params = {
            'Kmax':10,
            'coordinates':['euclidian','dipole_spherical'],'nrotations':1,'index_distance_max':None,'self_neighborhood':True},
        from_triplets = True,
        dipole=dipole,
        padded=False)

    plt.plot(params[0].T); plt.show()
    plt.hist(params[1].flatten(), bins=20); plt.show()

#%%

    params = initialize_GaussianKernel_for_NeighborhoodEmbedding(
        [inputs[0], inputs[3],inputs[2]], 16,
        covariance_type='diag',
        neighborhood_params={
            'Kmax': 10,
            'coordinates': ['distance', 'ZdotZ', 'ZdotDelta', 'index_distance'], 'nrotations': 1, 'index_distance_max':  8, 'self_neighborhood':True},
        from_triplets=True,
        padded=False)

    plt.plot(params[0].T)
    plt.show()
    plt.hist(params[1].flatten(), bins=20)
    plt.show()

#%%

    params = initialize_GaussianKernel_for_NeighborhoodEmbedding(
        [inputs[2]], 16,
        covariance_type='diag',
        neighborhood_params = {
            'Kmax':10,
            'coordinates':['index_distance'],'nrotations':1,'index_distance_max':16,'self_neighborhood':True},
        from_triplets = False,
        padded=False)


#%%
    model_params = initialize_Embedding_for_NeighborhoodAttention([inputs[0], inputs[3],inputs[2]],outputs,padded=False,from_triplets=True)

    from keras.models import Sequential
    import embeddings
    model = Sequential()
    model.add(GaussianKernel(16, covariance_type='full',
                      initial_values=model_params['graph_embedding_GaussianKernel'], name='graph_embedding_GaussianKernel',input_shape=(5,)))
    model.add(Dense(1, activation='linear',use_bias=False, name='graph_embedding_dense'))
    model.layers[-1].set_weights(model_params['graph_embedding_dense'])

    local_coordinates_flat = local_coordinates[local_coordinates.max(-1).max(-1)>0]
    graph_value = model.predict(local_coordinates_flat.reshape([-1,5]))
    plt.scatter(local_coordinates_flat[:,:,0].flatten(),graph_value[:,0],s=1,c=graph_value[:,-1]); plt.show()




    #%% Check consistency with wrapper.
    import wrappers
    import numpy as np


    all_triplets = []
    all_clouds = []

    B = 100
    for b in range(B):
        L = np.random.randint(5,high=21)
        random_direction1 = np.random.randn(3)
        random_direction2 = np.random.randn(3)
        points = np.random.randn(L,3)
        cloud = np.concatenate([
            points,
            points + random_direction1[np.newaxis],
            points + random_direction2[np.newaxis]
        ], axis=0)
        triplet = np.stack([
            np.arange(L),
            np.arange(L)+L,
            np.arange(L)+2*L], axis=-1)
        all_clouds.append(cloud)
        all_triplets.append(triplet)


    def keras_frames(Lmax=20):
        from keras.engine.base_layer import Layer
        from keras import backend as K
        import tensorflow as tf
        import numpy as np
        from keras.layers import Input, Masking, Dense
        from keras.models import Model
        cloud = Input(shape=(3*Lmax,3),dtype="float32")
        triplet = Input(shape=(Lmax, 3), dtype="int32")
        masked_cloud = Masking(mask_value=0.0, name='masked_indices_atom')(cloud)
        masked_triplets = Masking(mask_value=-1, name='masked_triplets_atom')(triplet)
        frames = FrameBuilder()([cloud,triplet])
        model = Model(inputs=[triplet,cloud],outputs=frames)
        return model


    model = wrappers.grouped_Predictor_wrapper(keras_frames,
                                               Lmax=20,
                                               multi_inputs=True,
                                               input_type=['triplets','points'],
                                               Lmaxs=[20,3*20])


    all_frames = model.predict([all_triplets,all_clouds],return_all=True,batch_size=1)

    for i,frame in enumerate(all_frames):
        print(i,(frame[:,1:].max(0)-frame[:,1:].min(0) ).max()  )
