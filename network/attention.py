from keras.engine.base_layer import Layer
from keras import backend as K
import tensorflow as tf





class AttentionLayer(Layer):
    def __init__(self, self_attention=True,beta=True,**kwargs):
        self.support_masking = True
        self.self_attention = self_attention
        self.beta = beta
        self.epsilon = tf.constant(1e-6)
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.beta & self.self_attention:
            beta_shape, self_attention_shape, attention_coefficient_shape, node_activity_shape, graph_weights_shape = input_shape
        elif self.beta & (~self.self_attention):
            beta_shape, attention_coefficient_shape, node_activity_shape, graph_weights_shape = input_shape
        elif (~self.beta) & self.self_attention:
            self_attention_shape, attention_coefficient_shape, node_activity_shape, graph_weights_shape = input_shape
        else:
            attention_coefficient_shape, node_activity_shape, graph_weights_shape = input_shape

        self.Lmax = graph_weights_shape[1]
        self.Kmax = graph_weights_shape[2]
        self.nfeatures_graph = graph_weights_shape[-1]
        self.nheads = attention_coefficient_shape[-1] // self.nfeatures_graph
        self.nfeatures_output = node_activity_shape[-1] // self.nheads
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs,mask=None):
        if self.beta & self.self_attention:
            beta, self_attention, attention_coefficients, node_outputs, graph_weights = inputs
        elif self.beta & (~self.self_attention):
            beta, attention_coefficients, node_outputs, graph_weights = inputs
        elif (~self.beta) & self.self_attention:
            self_attention, attention_coefficients, node_outputs, graph_weights = inputs
        else:
            attention_coefficients, node_outputs, graph_weights = inputs

        if self.beta:
            beta = tf.reshape(
                beta, [-1, self.Lmax, self.nfeatures_graph, self.nheads])
        if self.self_attention:
            self_attention = tf.reshape(
                self_attention, [-1, self.Lmax, self.nfeatures_graph, self.nheads])
        attention_coefficients = tf.reshape(
            attention_coefficients, [-1, self.Lmax, self.Kmax, self.nfeatures_graph, self.nheads])
        node_outputs = tf.reshape(
            node_outputs, [-1, self.Lmax, self.Kmax, self.nfeatures_output, self.nheads])

        # Add self-attention coefficient.
        if self.self_attention:
            attention_coefficients_self, attention_coefficient_others = tf.split(
                attention_coefficients, [1, self.Kmax - 1], axis=2)
            attention_coefficients_self += tf.expand_dims(self_attention, axis=2)
            attention_coefficients = tf.concat(
                [attention_coefficients_self, attention_coefficient_others], axis=2)
        # Multiply by inverse temperature beta.
        if self.beta:
            attention_coefficients *= tf.expand_dims(beta + self.epsilon, axis=2)

        ##
        attention_coefficients -= tf.reduce_max(
            attention_coefficients, axis=[-3, -2], keep_dims=True)
        attention_coefficients_final = tf.reduce_sum(tf.expand_dims(
            graph_weights, axis=-1) * K.exp(attention_coefficients), axis=-2)
        attention_coefficients_final /= tf.reduce_sum(
            tf.abs(attention_coefficients_final), axis=-2, keep_dims=True) + self.epsilon
        output_final = tf.reshape(tf.reduce_sum(node_outputs * tf.expand_dims(
            attention_coefficients_final, axis=-2), axis=2), [-1, self.Lmax, self.nfeatures_output * self.nheads])
        return [output_final, attention_coefficients_final]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.Lmax, self.nfeatures_output * self.nheads),
                (input_shape[0][0], self.Lmax, self.Kmax, self.nheads)]

    def compute_mask(self, input, mask=None):
        if mask not in [None,[None for _ in input]]:
            if self.beta | self.self_attention:
                return [mask[0], None]
            else:
                return [None,None]
        else:
            return mask




