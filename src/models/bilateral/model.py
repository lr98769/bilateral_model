import tensorflow as tf
from tensorflow import Variable
from keras import Sequential, Input
from keras.layers import Dense, Multiply
import numpy as np
from src.models.bilateral.groupedSoftmaxDenseLayer import GroupedSoftmaxDenseLayer
from src.models.bilateral.regularizer import SparseRegularizer
from src.configs.config import num_input_cols

tf.keras.utils.get_custom_objects().clear()
class BilateralModel(tf.keras.Model):
    def __init__(
        self,
        input_cols, intermediate_col_dict, output_cols, 
        le_label, re_label,
        ae_width, intermediate_classifier_width, final_classifier_width, 
        rho=0.1, beta=1., **kwargs, 
    ):
        super().__init__(**kwargs)
        
        
        self.input_cols = Variable(input_cols, trainable=False) # For each eye
        intermediate_cols = []
        intermediate_col_groups = []
        intermediate_col_sizes = []
        for key, col_list in intermediate_col_dict.items():
            for col in col_list:
                intermediate_cols.append(f"{key}_{col}")
            intermediate_col_groups.append(key)
            intermediate_col_sizes.append(len(col_list))
        intermediate_col_sizes=np.array(intermediate_col_sizes)
        # self.intermediate_col_dict = Variable(intermediate_col_dict, trainable=False) # For each eye
        self.intermediate_cols = Variable(intermediate_cols, trainable=False) # For each eye
        self.intermediate_col_groups = Variable(intermediate_col_groups, trainable=False)
        self.intermediate_col_sizes = Variable(intermediate_col_sizes, trainable=False)
        self.output_cols = Variable(output_cols, trainable=False) # For each eye

        self.le_label = Variable(le_label, trainable=False) 
        self.re_label = Variable(re_label, trainable=False) 

        self.num_input_cols = Variable(len(input_cols), trainable=False)
        self.num_intermediate_cols = Variable(len(intermediate_cols), trainable=False)
        self.num_output_cols = Variable(len(output_cols), trainable=False)

        self.ae_width = Variable(ae_width, trainable=False)
        self.intermediate_classifier_width = Variable(intermediate_classifier_width, trainable=False)
        self.final_classifier_width = Variable(final_classifier_width, trainable=False)

        self.rho = Variable(rho, trainable=False)
        self.beta = Variable(beta, trainable=False)

        num_input_cols = self.num_input_cols.numpy()
        num_intermediate_cols = self.num_intermediate_cols.numpy()
        num_output_cols = self.num_output_cols.numpy()
        
        # Autoencoder (Learns structure of data in an unsupervised manner)
        self.encoder = Sequential([
            Input(shape=(num_input_cols,)),
            Dense(ae_width, activation="relu"),
            Dense(ae_width, activation="relu"),
            Dense(ae_width, activation="relu", 
                  activity_regularizer=SparseRegularizer(rho = rho, beta = beta)), # Induce sparsity on this layer
        ], name="encoder")
        self.decoder = Sequential([
            Input(shape=(ae_width,)),
            Dense(ae_width, activation="relu"),
            Dense(ae_width, activation="relu"),
            Dense(num_input_cols)
        ], name="decoder")

        # Intermediate Classifier Layer (E.g. Maps from encoded features to diagnosis for each eye)
        self.intermediate_classifier = GroupedSoftmaxDenseLayer(
            input_width = ae_width,
            intermediate_col_sizes=intermediate_col_sizes, 
            num_layers_per_intermediate_classifier=3, 
            intermediate_classifier_width=intermediate_classifier_width,
            name="intermediate_classifier"
        )
        
        # concatenate softmax outputs
        self.concatenate = tf.keras.layers.Concatenate()

        # Final Classifier Layer (E.g. Maps from diagnosis for each eye to TCU for this eye)
        self.final_classifier = Sequential([
            Input(shape=((num_intermediate_cols,))),
            Dense(final_classifier_width, activation="relu"),
            Dense(final_classifier_width, activation="relu"),
            Dense(num_output_cols, activation="sigmoid"), 
        ], name="final_classifier")
        # Citation: https://ieeexplore.ieee.org.remotexs.ntu.edu.sg/stamp/stamp.jsp?tp=&arnumber=4633963
        # Output class should be processed like this:
        # - Most severe: [0, 0, 0], Least severe: [1, 1, 1]

        # E.g. Get diagnosis of both eyes:
        # [0, 0, 0] * [1, 1, 1] = [0, 0, 0] (the TCU of the most severe eye)
        self.multiply = Multiply(name="and")

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, num_input_cols], dtype=tf.float32)])
    def predict_ae_each(self, x):
        # Encode
        output = self.encoder(x)
        # Decode
        output = self.decoder(output)
        return output

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, num_input_cols], dtype=tf.float32)])
    def predict_intermediate_classifier_each(self, x):
        # Encode
        output = self.encoder(x)
        # Classify intermediate output
        output = self.intermediate_classifier(output)
        return output

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, num_input_cols], dtype=tf.float32)])
    def predict_final_classifier_each(self, x):
        # Encode
        output = self.encoder(x)
        # Classify intermediate output
        output = self.intermediate_classifier(output)
        # Concatenate individual softmax classifiers
        output = self.concatenate(output)
        # Classify final output
        output = self.final_classifier(output)
        return output

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, num_input_cols], dtype=tf.float32),
        tf.TensorSpec(shape=[None, num_input_cols], dtype=tf.float32)]
    )
    def predict_final_classifier_both(self, x1, x2):
        # Assume x1 and x2 are aligned
        # For x1
        # - Encode
        output1 = self.encoder(x1)
        
        # - Classify intermediate output
        output1 = self.intermediate_classifier(output1)
        # Concatenate individual softmax classifiers
        output1 = self.concatenate(output1)
        # - Classify final output
        output1 = self.final_classifier(output1)

        # For x2
        # - Encode
        output2 = self.encoder(x2)
        # - Classify intermediate output
        output2 = self.intermediate_classifier(output2)
        # Concatenate individual softmax classifiers
        output2 = self.concatenate(output2)
        # - Classify final output
        output2 = self.final_classifier(output2)

        # Get final class
        output = self.multiply([output1, output2])
        return output

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, num_input_cols], dtype=tf.float32),
        tf.TensorSpec(shape=[None, num_input_cols], dtype=tf.float32)]
    )
    def predict_everything(self, x1, x2):
        # Assume x1 and x2 are aligned
        # For x1
        # - Encode
        encoder1 = self.encoder(x1)
        # - Decode
        decoder1 = self.decoder(encoder1)
        # - Classify intermediate output
        ic1 = self.intermediate_classifier(encoder1)
        # Concatenate individual softmax classifiers
        ic1 = self.concatenate(ic1)
        # - Classify final output
        fc1 = self.final_classifier(ic1)

        # For x2
        # - Encode
        encoder2 = self.encoder(x2)
        # - Decode
        decoder2 = self.decoder(encoder2)
        # - Classify intermediate output
        ic2 = self.intermediate_classifier(encoder2)
        # Concatenate individual softmax classifiers
        ic2 = self.concatenate(ic2)
        # - Classify final output
        fc2 = self.final_classifier(ic2)

        # Get final class
        output = self.multiply([fc1, fc2])

        predict_dict = dict(
            encoder1=encoder1, encoder2=encoder2,
            decoder1=decoder1, decoder2=decoder2,
            ic1=ic1, ic2=ic2,
            fc1=fc1, fc2=fc2,
            output=output
        )
        return predict_dict
    
    def call(self, x1, x2):
        return self.predict_final_classifier_both(x1, x2)
        
