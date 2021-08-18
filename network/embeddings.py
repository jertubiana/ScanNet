from keras.engine.base_layer import Layer
from keras import backend as K
from keras.constraints import NonNeg
from keras.initializers import Zeros,TruncatedNormal,Ones,RandomUniform
import numpy as np
from sklearn.mixture import GaussianMixture
import keras
from network.utils import Init2Value,FixedNorm,Symmetric,ConstraintBetween
import tensorflow as tf
from keras.layers import BatchNormalization

class GaussianKernel(Layer):
    def __init__(self, N, initial_values, covariance_type='diag', eps=1e-1, **kwargs):
        super(GaussianKernel, self).__init__(**kwargs)
        self.support_masking = True
        self.eps = eps
        self.N = N
        self.initial_values = initial_values
        self.covariance_type = covariance_type
        assert self.covariance_type in ['diag', 'full']

    def build(self, input_shape):
        self.nbatch_dim = len(input_shape) - 1
        self.d = input_shape[-1]

        self.center_shape = [self.d, self.N]

        self.centers = self.add_weight(shape=self.center_shape, name='centers',
                                       initializer=Init2Value(
                                           self.initial_values[0]),
                                       regularizer=None,
                                       constraint=None)

        if self.covariance_type == 'diag':
            self.width_shape = [self.d, self.N]

            self.widths = self.add_weight(shape=self.width_shape,
                                          name='widths',
                                          initializer=Init2Value(
                                              self.initial_values[1]),
                                          regularizer=None,
                                          constraint=NonNeg())

        elif self.covariance_type == 'full':
            self.sqrt_precision_shape = [self.d, self.d, self.N]

            self.sqrt_precision = self.add_weight(shape=self.sqrt_precision_shape,
                                                  name='sqrt_precision',
                                                  initializer=Init2Value(
                                                      self.initial_values[1]),
                                                  regularizer=None,
                                                  constraint=ConstraintBetween(-1/self.eps,1/self.eps))

        # Set input spec
        super(GaussianKernel, self).build(input_shape)

    def call(self, inputs, mask=None):
        if self.covariance_type == 'diag':
            activity = K.exp(- 0.5 * K.sum(
                (
                    (
                        K.expand_dims(inputs, axis=-1)
                        - K.reshape(self.centers,
                                    [1 for _ in range(self.nbatch_dim)] + self.center_shape)
                    ) / K.reshape(self.eps + self.widths, [1 for _ in range(self.nbatch_dim)] + self.width_shape)
                )**2, axis=-2))

        elif self.covariance_type == 'full':
            intermediate = K.expand_dims(inputs, axis=-1) - K.reshape(self.centers,
                                                                      [1 for _ in range(self.nbatch_dim)] + self.center_shape)  # B X d X n_centers

            intermediate2 = K.sum(
                K.expand_dims(intermediate, axis=-3) *
                K.expand_dims(self.sqrt_precision, axis=0),
                axis=- 2)

            activity = K.exp(- 0.5 * K.sum(intermediate2**2, axis=-2))

        return activity

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape[:-1]) + [self.N]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    def get_config(self):
        config = {'N': self.N, 'initial_values': self.initial_values,
                  'covariance_type': self.covariance_type}
        base_config = super(
            GaussianKernel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def inv_root_matrix(H):
    lam, v = np.linalg.eigh(H)
    return np.dot(v,  1 / np.sqrt(lam)[:, np.newaxis] * v.T)


def initialize_GaussianKernel(points, N,covariance_type='diag',reg_covar=1e-1,n_init=10):
    GMM = GaussianMixture(n_components=N, covariance_type=covariance_type, verbose=1,
                          reg_covar=reg_covar,n_init=n_init)
    GMM.fit(points)
    centers = GMM.means_
    covariances = GMM.covariances_
    probas = GMM.weights_
    order = np.argsort(probas)[::-1]
    centers = centers[order]
    covariances = covariances[order]
    probas = probas[order]
    if covariance_type == 'diag':
        widths = np.sqrt(covariances)
    elif covariance_type == 'full':
        sqrt_precision_matrix = np.array(
            [inv_root_matrix(covariance) for covariance in covariances])
    if covariance_type == 'diag':
        return centers.T, widths.T
    elif covariance_type == 'full':
        return centers.T, sqrt_precision_matrix.T


def initialize_GaussianKernelRandom(xlims, N, covariance_type):
    xlims = np.array(xlims,dtype=np.float32)
    coordinates_dimension = xlims.shape[0]

    centers = np.random.rand(coordinates_dimension, N).astype(np.float32)
    centers = centers * (xlims[:,1]-xlims[:,0])[:,np.newaxis] + xlims[:,0][:,np.newaxis]

    widths = np.ones([coordinates_dimension, N], dtype=np.float32)
    widths = widths * (xlims[:, 1] - xlims[:, 0])[:, np.newaxis] / (N / 4)

    if covariance_type == 'diag':
        initial_values = [centers,widths]
    else:
        sqrt_precision_matrix = np.stack([np.diag( 1.0/(1e-4+widths[:,n]) ).astype(np.float32) for n in range(N)],axis=-1)
        initial_values = [centers,sqrt_precision_matrix]
    return initial_values





class OuterProduct(Layer):
    def __init__(self, n_filters, use_single1=True, use_single2=True, use_bias=True, non_negative=False, unitnorm=False, fixednorm=None,
    symmetric=False, diagonal = False, non_negative_initial=False,
     kernel_regularizer=None, single1_regularizer=None, single2_regularizer=None, sum_axis=None, **kwargs):
        super(OuterProduct, self).__init__(**kwargs)
        self.support_masking = True
        self.n_filters = n_filters
        self.use_single1 = use_single1
        self.use_single2 = use_single2
        self.use_bias = use_bias
        self.non_negative = non_negative
        self.kernel_regularizer = kernel_regularizer
        self.single1_regularizer = single1_regularizer
        self.single2_regularizer = single2_regularizer
        if unitnorm:  # for retro-compatibility...
            fixednorm = 1.0
        self.fixednorm = fixednorm
        self.symmetric = symmetric
        self.diagonal = diagonal
        self.sum_axis = sum_axis
        self.non_negative_initial = non_negative_initial


    def build(self, input_shape):
        if self.non_negative:
            constraint = NonNeg()
        else:
            constraint = None

        if self.fixednorm is not None:
            constraint_kernel = FixedNorm(value=self.fixednorm, axis=[0, 1])
        else:
            constraint_kernel = constraint

        if self.symmetric:
            constraint_kernel = Symmetric()

        self.n1 = input_shape[0][-1]
        self.n2 = input_shape[1][-1]
        if self.fixednorm is not None:
            stddev = self.fixednorm / np.sqrt(self.n1 * self.n2)
        else:
            if self.diagonal:
                stddev = 1.0 / np.sqrt(self.n1)
            else:
                stddev = 1.0 / np.sqrt(self.n1 * self.n2)



        if self.non_negative_initial:
            initializer = RandomUniform(
                        minval=0, maxval=stddev * np.sqrt(3) ) # such that < w^2 > = stddev exactly.

        else:
            initializer = TruncatedNormal(
                        mean=0., stddev=stddev)


        if self.diagonal:
            self.kernel12 = self.add_weight(
                shape=[self.n1, self.n_filters],
                name='kernel12',
                initializer=initializer,
            )
        else:
            self.kernel12 = self.add_weight(
                shape=[self.n1, self.n2, self.n_filters],
                name='kernel12',
                initializer=TruncatedNormal(
                    mean=0., stddev=stddev),
                constraint=constraint_kernel,
                regularizer=self.kernel_regularizer
            )

        if self.use_single1:
            stddev = 1.0 / np.sqrt(self.n1)
            if self.non_negative_initial:
                initializer = RandomUniform(
                    minval=0, maxval=stddev * np.sqrt(3)) # such that < w^2 > = stddev exactly.

            else:
                initializer = TruncatedNormal(
                    mean=0., stddev=stddev)

            self.kernel1 = self.add_weight(
                shape=[self.n1, self.n_filters],
                name='kernel1',
                initializer=initializer,
                constraint=constraint,
                regularizer = self.single1_regularizer
            )
        if self.use_single2:
            stddev = 1.0 / np.sqrt(self.n2)
            if self.non_negative_initial:
                initializer = RandomUniform(
                    minval=0, maxval=stddev * np.sqrt(3)) # such that < w^2 > = stddev exactly.

            else:
                initializer = TruncatedNormal(
                    mean=0., stddev=stddev)

            if self.symmetric:
                self.kernel2 = self.kernel1
            else:
                self.kernel2 = self.add_weight(
                shape=[self.n2, self.n_filters],
                name='kernel2',
                initializer=initializer,
                constraint=constraint,
                regularizer = self.single2_regularizer
            )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=[self.n_filters, ],
                name='bias',
                initializer=Zeros(),
                constraint=None
            )

        # Set input spec
        super(OuterProduct, self).build(input_shape)

    def call(self, inputs, mask=None):
        first_input = inputs[0]
        second_input = inputs[1]
        bias_shape = [1 for _ in first_input.shape[:-1]] + [self.n_filters]
        if self.sum_axis is not None:
            del bias_shape[self.sum_axis]

        if self.diagonal:
            activity = K.dot(first_input * second_input, self.kernel12)
        else:
            if self.sum_axis is not None:
                outer_product = tf.reduce_sum(tf.expand_dims(
                    first_input, axis=-1) * tf.expand_dims(second_input, axis=-2), axis=self.sum_axis)
            else:
                outer_product = tf.expand_dims(
                    first_input, axis=-1) * tf.expand_dims(second_input, axis=-2)

            activity = tf.tensordot(
                outer_product, self.kernel12, [[-2, -1], [0, 1]])

        if self.use_single1:
            if self.sum_axis is not None:
                activity += K.dot(tf.reduce_sum(first_input,
                                                axis=self.sum_axis), self.kernel1)
            else:
                activity += K.dot(first_input, self.kernel1)
        if self.use_single2:
            if self.sum_axis is not None:
                activity += K.dot(tf.reduce_sum(second_input,
                                                axis=self.sum_axis), self.kernel2)
            else:
                activity += K.dot(second_input, self.kernel2)
        if self.use_bias:
            activity += K.reshape(self.bias, bias_shape)
        return activity

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0][0]] + [max(shape1, shape2) for shape1, shape2 in zip(input_shape[0][1:-1],
                                                                                            input_shape[1][1:-1])] + [self.n_filters]
        if self.sum_axis is not None:
            del output_shape[self.sum_axis]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        if self.sum_axis is not None:
            return mask[0][..., 0]
        else:
            return mask[0]

    def get_config(self):
        config = {'n_filters': self.n_filters,
                  'use_single1': self.use_single1,
                  'use_single2': self.use_single2,
                  'use_bias': self.use_bias,
                  'non_negative': self.non_negative,
                  'fixednorm': self.fixednorm,
                  'symmetric':self.symmetric,
                  'diagonal':self.diagonal,
                  'sum_axis': self.sum_axis,
                  }
        base_config = super(
            OuterProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MultiTanh(Layer):
    def __init__(self,ntanh,use_bias=True,**kwargs):

        super(MultiTanh, self).__init__(**kwargs)
        self.supports_masking = True
        self.ntanh = ntanh
        self.use_bias = use_bias

    def build(self, input_shape):
        param_shape = (input_shape[-1],self.ntanh)
        param_shape2 = (input_shape[-1],)
        self.broadcasted_param_shape = [1 for _ in range(
            len(input_shape)-1 )] + [input_shape[-1],self.ntanh]
        self.broadcasted_param_shape2 = [1 for _ in range(
            len(input_shape)-1 )] + [input_shape[-1] ]


        self.widths = self.add_weight(shape=param_shape,
                                   name='widths',
                                   initializer=keras.initializers.Constant(1),
                                   regularizer=None,
                                   constraint=NonNeg())

        self.slopes = self.add_weight(shape=param_shape,
                                    name='slopes',
                                    initializer=keras.initializers.Constant(1),
                                    constraint=NonNeg())

        initial_offsets = np.zeros([input_shape[-1],self.ntanh])
        if self.ntanh>1:
            initial_offsets += (np.arange(self.ntanh)/(self.ntanh-1) * (3 - (-3)) - 3 )[np.newaxis]


        self.offsets = self.add_weight(shape=param_shape,
                                     name='offsets',
                                     initializer=Init2Value( initial_offsets  )
                                     )

        if self.use_bias:
            self.biases = self.add_weight(shape=param_shape2,
                                       name='biases',
                                        initializer ='zeros'
                                       )

        super(MultiTanh, self).build(input_shape)

    def call(self, inputs, mask=None):
        widths = K.reshape(self.widths, self.broadcasted_param_shape)
        slopes = K.reshape(self.slopes, self.broadcasted_param_shape)
        offsets = K.reshape(self.offsets, self.broadcasted_param_shape)
        if self.use_bias:
            biases = K.reshape(self.biases, self.broadcasted_param_shape2)

        output= K.sum(slopes * K.tanh( (K.expand_dims(inputs,-1) - offsets )/(widths + 1e-4 ) ) ,axis =-1)
        if self.use_bias:
            output+= biases
        return output


    def get_config(self):
        config = {'ntanh': self.ntanh,'use_bias':self.use_bias}
        base_config = super(MultiTanh, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape






def moments_masked(
        x, mask,
        axes,
        shift=None,  # pylint: disable=unused-argument
        name=None,
        keep_dims=None,
        keepdims=None):
    if keep_dims is None:
        keep_dims = False
    with tf.name_scope(name, "moments", [x, axes]):
        # The dynamic range of fp16 is too limited to support the collection of
        # sufficient statistics. As a workaround we simply perform the operations
        # on 32-bit floats before converting the mean and variance back to fp16
        y = tf.cast(x, tf.dtypes.float32) if x.dtype == tf.dtypes.float16 else x
        mask = tf.expand_dims(tf.cast(mask, tf.dtypes.float32), axis=-1)
        # Compute true mean while keeping the dims for proper broadcasting.
        sum_mask = tf.maximum( tf.reduce_sum(mask, axes, keepdims=True, name="mask_sum"), 1)
        mean = tf.reduce_sum(y * mask, axes, keepdims=True,
                             name="sum") / sum_mask
        # sample variance, not unbiased variance
        # Note: stop_gradient does not change the gradient that gets
        #       backpropagated to the mean from the variance calculation,
        #       because that gradient is zero
        variance = tf.reduce_sum(
            tf.squared_difference(y, tf.stop_gradient(mean)) * mask,
            axes,
            keepdims=True,
            name="variance") / sum_mask
        if not keep_dims:
            mean = tf.squeeze(mean, axes)
            variance = tf.squeeze(variance, axes)
        if x.dtype == tf.dtypes.float16:
            return (tf.cast(mean, tf.dtypes.float16),
                    tf.cast(variance, tf.dtypes.float16))
        else:
            return (mean, variance)


def normalize_batch_in_training_masking(x, mask, gamma, beta,
                                        reduction_axes, epsilon=1e-3):
    """Only works for Batch size X Time X features or Batch size X features"""
    mean, var = moments_masked(x, mask, reduction_axes,
                               None, None, False)
    normed = tf.nn.batch_normalization(x, mean, var,
                                       beta, gamma,
                                       epsilon)
    return normed, mean, var


class MaskedBatchNormalization(BatchNormalization):
    def call(self, inputs, training=None, mask=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    axis=self.axis,
                    epsilon=self.epsilon)
            else:
                return K.batch_normalization(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    axis=self.axis,
                    epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if (training in {0, False}) | (self.trainable == False):
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        if mask is None:
            normed_training, mean, variance = K.normalize_batch_in_training(
                inputs, self.gamma, self.beta, reduction_axes,
                epsilon=self.epsilon)
        else:
            normed_training, mean, variance = normalize_batch_in_training_masking(
                inputs, mask, self.gamma, self.beta, reduction_axes,
                epsilon=self.epsilon)

        if K.backend() != 'cntk':
            if mask is None:
                sample_size = K.prod([K.shape(inputs)[axis]
                                      for axis in reduction_axes])
                sample_size = K.cast(sample_size, dtype=K.dtype(inputs))
                if K.backend() == 'tensorflow' and sample_size.dtype != 'float32':
                    sample_size = K.cast(sample_size, dtype='float32')
            else:
                sample_size = K.maximum(K.sum(K.cast(mask, dtype='float32')), 2)

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask



class Bias(Layer):
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)
        self.support_masking = True

    def build(self, input_shape):
        self.ndim = input_shape[-1]
        self.bias_dynamical_shape = [
                                        1 for _ in range(len(input_shape) - 1)] + [self.ndim]
        self.bias = self.add_weight(
            shape=[self.ndim, ],
            name='bias',
            initializer=Zeros(),
            constraint=None)
        super(Bias, self).build(input_shape)

    def call(self, inputs):
        return inputs + tf.reshape(self.bias, self.bias_dynamical_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, input, mask=None):
        return mask



class Slope(Layer):
    def __init__(self, **kwargs):
        super(Slope, self).__init__(**kwargs)
        self.support_masking = True

    def build(self, input_shape):
        self.ndim = input_shape[-1]
        self.slope_dynamical_shape = [
                                        1 for _ in range(len(input_shape) - 1)] + [self.ndim]
        self.slope = self.add_weight(
            shape=[self.ndim, ],
            name='slope',
            initializer=Ones(),
            constraint=None)
        super(Slope, self).build(input_shape)

    def call(self, inputs):
        return inputs * tf.reshape(self.slope, self.slope_dynamical_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, input, mask=None):
        return mask

