import tensorflow as tf
from keras import Input, Sequential
from keras.layers import Dense


@tf.keras.utils.register_keras_serializable(package="MyGroupedSoftmaxDenseLayers")
class GroupedSoftmaxDenseLayer(tf.keras.layers.Layer):
    def __init__(
      self, input_width, intermediate_col_sizes, num_layers_per_intermediate_classifier, intermediate_classifier_width, name=None):
        super(GroupedSoftmaxDenseLayer, self).__init__(name=name)

        self.input_width = input_width
        self.intermediate_col_sizes = intermediate_col_sizes
        self.num_layers_per_intermediate_classifier = num_layers_per_intermediate_classifier
        self.intermediate_classifier_width = intermediate_classifier_width

        self.layers = []
        for col_size in intermediate_col_sizes:
            cur_int_classifier = Sequential()
            cur_int_classifier.add(Input(shape=((input_width,))))
            for i in range(num_layers_per_intermediate_classifier):
                cur_int_classifier.add(
                    Dense(intermediate_classifier_width, activation="relu"))
            cur_int_classifier.add(Dense(col_size, activation="softmax"))
            self.layers.append(cur_int_classifier)

    def call(self, inputs):
        outputs = []
        for layer in self.layers:
            output = layer(inputs)
            outputs.append(output)
        return outputs

    def get_config(self):
        return {
            'input_width': self.input_width,
            'intermediate_col_sizes': self.intermediate_col_sizes,
            'num_layers_per_intermediate_classifier': self.num_layers_per_intermediate_classifier,
            'intermediate_classifier_width': self.intermediate_classifier_width,
            "name": self.name
        }