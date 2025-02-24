import tensorflow as tf

# For Sparsity, from: https://stackoverflow.com/questions/36913281/how-do-i-correctly-implement-a-custom-activity-regularizer-in-keras
# https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
# https://ieeexplore.ieee.org.remotexs.ntu.edu.sg/abstract/document/7280364
# https://serp.ai/sparse-autoencoder/ (Why KL Divergence)
@tf.keras.utils.register_keras_serializable(package="MyRegularizers")
class SparseRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, rho = 0.01,beta = 1):
        """
        rho  : Desired average activation of the hidden units
        beta : Weight of sparsity penalty term
        """
        self.rho = rho
        self.beta = beta
        
    def __call__(self, activation):
        # sigmoid because we need the probability distributions
        activation = tf.nn.sigmoid(activation)
        # average over the batch samples (Proportion of samples that have this node activated)
        rho_bar = tf.keras.backend.mean(activation, axis=0)
        # Avoid division by 0
        rho_bar = tf.keras.backend.maximum(rho_bar, 1e-10) 
        # KL divergence with bernoulli distribution where p = rho 
        KLs = self.rho*tf.keras.backend.log(self.rho/rho_bar) + (1-self.rho)*tf.keras.backend.log((1-self.rho)/(1-rho_bar))
        # sum over the layer units
        return self.beta * tf.keras.backend.sum(KLs) 
        
    def get_config(self):
        return {
            'rho': self.rho,
            'beta': self.beta
        }
