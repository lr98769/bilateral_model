import tensorflow as tf
from src.models.bilateral.groupedSoftmaxDenseLayer import GroupedSoftmaxDenseLayer


def get_ae_model(model):
    input = tf.keras.Input(shape=(len(model.input_cols.numpy()),))
    output = model.encoder(input)
    output = model.decoder(output)
    return tf.keras.Model(inputs=input, outputs=output, name="ae")

def get_intermediate_classifier_model(model):
    input = tf.keras.Input(shape=(len(model.input_cols.numpy()),))
    output = model.encoder(input)
    output = model.intermediate_classifier(output)
    return tf.keras.Model(inputs=input, outputs=output, name="intermediate_classifier")

def get_individual_final_classifier(model):
    input1 = tf.keras.Input(shape=(len(model.input_cols.numpy()),))
    output1 = model.encoder(input1)
    output1 = model.intermediate_classifier(output1)
    output1 = model.concatenate(output1)
    output1 = model.final_classifier(output1)
    return tf.keras.Model(inputs=[input1], outputs=output1, name="individual_final_classifier")

def get_final_classifier_model(model):
    input1 = tf.keras.Input(shape=(len(model.input_cols.numpy()),))
    input2 = tf.keras.Input(shape=(len(model.input_cols.numpy()),))
    output1 = model.encoder(input1)
    output1 = model.intermediate_classifier(output1)
    output1 = model.concatenate(output1)
    output1 = model.final_classifier(output1)
    output2 = model.encoder(input2)
    output2 = model.intermediate_classifier(output2)
    output2 = model.concatenate(output2)
    output2 = model.final_classifier(output2)
    output = model.multiply([output1, output2])
    return tf.keras.Model(inputs=[input1, input2], outputs=output, name="final_classifier")