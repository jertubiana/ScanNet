from keras.engine.base_layer import Layer
from keras import backend as K
from keras.constraints import Constraint
from keras.initializers import Initializer
import tensorflow as tf
import numpy as np





class Init2Value(Initializer):
    def __init__(self, value):
        self.value = value

    def __call__(self, shape, dtype=None):
        assert self.value.shape == self.value.shape
        return self.value.astype(np.float32)




class ConstraintBetween(Constraint):
    def __init__(self, minimum=-1, maximum=+1):
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, w):
        return K.clip(w, self.minimum, self.maximum)



class FixedNorm(Constraint):
    def __init__(self, value=1.0, axis=0):
        self.axis = axis
        self.value = value

    def __call__(self, w):
        return w * self.value / (
            K.epsilon() + K.sqrt(
                tf.reduce_sum(
                    tf.square(w), axis=self.axis, keepdims=True)))

    def get_config(self):
        return {'axis': self.axis, 'value': self.value}

class Symmetric(Constraint):
    def __call__(self, w):
        return (w + tf.transpose(w, [1, 0, 2]) )/2



def masked_categorical_cross_entropy(y_true, y_pred):
    # Inspired from https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/.
    # mask = K.max(K.cast(K.equal(y_true, 1), K.floatx()), axis=-1)
    # Modification for when y_true is float (class reweighting)
    mask = K.max(K.cast(K.greater(y_true, 0), K.floatx()), axis=-1)
    return K.mean(K.categorical_crossentropy(y_true, y_pred) * mask, axis=-1) / K.mean(mask, axis=-1)


def masked_categorical_accuracy(y_true, y_pred):
    # Modified from https://github.com/keras-team/keras/blob/master/keras/metrics.py
    # mask = K.max(K.cast(K.equal(y_true, 1), K.floatx()), axis=-1)
    # Modification for when y_true is float (class reweighting)
    mask = K.max(K.cast(K.greater(y_true, 0), K.floatx()), axis=-1)
    error = K.cast(K.equal(K.argmax(y_true, axis=-1),
                           K.argmax(y_pred, axis=-1)),
                   K.floatx())
    return K.mean(error * mask) / K.mean(mask)


def masked_MSE(y_true, y_pred):
    mask = 1 - K.cast(K.equal(y_true, -1), K.floatx())
    return K.mean(K.square(y_pred - y_true) * mask, axis=-1) / K.mean(mask, axis=-1)




class MaxAbsPooling(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(MaxAbsPooling, self).__init__(**kwargs)
        self.support_masking = True
        self.axis = axis

    def build(self, input_shape):
        self.ndim = len(input_shape)
        super(MaxAbsPooling, self).build(input_shape)

    def call(self, inputs):
        if self.axis < 0:
            axis = self.axis + self.ndim
        else:
            axis = self.axis
        if (axis != self.ndim - 1):
            identity = np.arange(self.ndim)
            permutation = np.arange(self.ndim)
            permutation[axis:-1] = identity[axis + 1:]
            permutation[-1] = axis
            inputs = tf.transpose(inputs, perm=permutation,
                                  conjugate=False, name='transpose')

        indices = tf.expand_dims(tf.argmax(tf.abs(inputs), axis=-1), axis=-1)
        out = tf.gather_nd(inputs, indices, batch_dims=self.ndim - 1)
        return out

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        del output_shape[self.axis]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        if mask is not None:
            return mask[..., 0]
        else:
            return mask

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(MaxAbsPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




def embeddings_initializer(shape, dtype=None):
    if shape[0] == shape[1] + 1:
        init = np.zeros(shape, dtype=np.float32)
        for i in range(1, shape[0]):
            init[i, i - 1] = np.sqrt(shape[1])
    else:
        init = np.random.randn(*shape).astype(np.float32)
        init[0, :] *= 0
        init[1:, :] /= np.sqrt((init[1:, :]**2).mean(0))[np.newaxis, :]
    return tf.constant(init,dtype=dtype)


class PoolbyIndices(Layer):
    def __init__(self,**kwargs):
        self.support_masking = True
        super(PoolbyIndices,self).__init__(**kwargs)

    def call(self,inputs,mask=[None,None,None]):
        index_target,index_source, array_source = inputs
        mapping = tf.equal(tf.expand_dims(
            index_target[:,:,0], axis = -1),
                           tf.expand_dims(
            index_source[:,:,0], axis=-2)
                           )
        if mask[0] is not None:
            mapping = tf.logical_and(mapping,tf.expand_dims(mask[0],axis=-1) )
        if mask[1] is not None:
            mapping = tf.logical_and(mapping,tf.expand_dims(mask[1],axis=-2) )
        mapping = tf.cast(mapping,tf.float32)
        array_target = tf.matmul(tf.cast(mapping,tf.float32), array_source)/(1e-10+tf.reduce_sum(mapping,axis=-1,keep_dims=True) )
        return array_target

    def compute_output_shape(self, input_shape):
        index_target_shape,index_source_shape,array_shape = input_shape
        output_shape = list(index_target_shape[:-1]) + [array_shape[-1]]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=[None, None,None]):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        if mask != [None, None,None]:
            return mask[0]
        else:
            return mask




class MaxAbsPooling(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(MaxAbsPooling, self).__init__(**kwargs)
        self.support_masking = True
        self.axis = axis

    def build(self, input_shape):
        self.ndim = len(input_shape)
        super(MaxAbsPooling, self).build(input_shape)

    def call(self, inputs):
        if self.axis < 0:
            axis = self.axis + self.ndim
        else:
            axis = self.axis
        if (axis != self.ndim - 1):
            identity = np.arange(self.ndim)
            permutation = np.arange(self.ndim)
            permutation[axis:-1] = identity[axis + 1:]
            permutation[-1] = axis
            inputs = tf.transpose(inputs, perm=permutation,
                                  conjugate=False, name='transpose')

        indices = tf.expand_dims(tf.argmax(tf.abs(inputs), axis=-1), axis=-1)
        out = tf.gather_nd(inputs, indices, batch_dims=self.ndim - 1)
        return out

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        del output_shape[self.axis]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        if mask is not None:
            return mask[..., 0]
        else:
            return mask

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(MaxAbsPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
