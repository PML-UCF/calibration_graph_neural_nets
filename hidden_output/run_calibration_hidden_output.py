# ______          _           _     _ _ _     _   _
# | ___ \        | |         | |   (_) (_)   | | (_)
# | |_/ / __ ___ | |__   __ _| |__  _| |_ ___| |_ _  ___
# |  __/ '__/ _ \| '_ \ / _` | '_ \| | | / __| __| |/ __|
# | |  | | | (_) | |_) | (_| | |_) | | | \__ \ |_| | (__
# \_|  |_|  \___/|_.__/ \__,_|_.__/|_|_|_|___/\__|_|\___|
# ___  ___          _                 _
# |  \/  |         | |               (_)
# | .  . | ___  ___| |__   __ _ _ __  _  ___ ___
# | |\/| |/ _ \/ __| '_ \ / _` | '_ \| |/ __/ __|
# | |  | |  __/ (__| | | | (_| | | | | | (__\__ \
# \_|  |_/\___|\___|_| |_|\__,_|_| |_|_|\___|___/
#  _           _                     _
# | |         | |                   | |
# | |     __ _| |__   ___  _ __ __ _| |_ ___  _ __ _   _
# | |    / _` | '_ \ / _ \| '__/ _` | __/ _ \| '__| | | |
# | |___| (_| | |_) | (_) | | | (_| | || (_) | |  | |_| |
# \_____/\__,_|_.__/ \___/|_|  \__,_|\__\___/|_|   \__, |
#                                                   __/ |
#                                                  |___/
#
# MIT License
#
# Copyright (c) 2019 Probabilistic Mechanics Laboratory
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
Calibration neural networks

Hidden output example
"""

import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow import compat
from tensorflow.keras.callbacks import ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.layers import Input, Lambda, Dense, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops

from pinn.layers import inputsSelection, TableInterpolation, getScalingDenseLayer


# TensorFlow 2.0.0 Disable Eager
compat.v1.disable_eager_execution()


# Layers

class StorageModulus(Layer):
    """
    w = x[:,0]
    T = x[:,1]

    y = tt1[0]*w + tt1[1]*T + tt1[2]
    """

    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(StorageModulus, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape, **kwargs):
        self.kernel = self.add_weight("kernel",
                                      shape=[3],
                                      initializer=self.kernel_initializer,
                                      dtype=self.dtype,
                                      trainable=self.trainable,
                                      **kwargs)
        self.built = True

    def call(self, inputs):
        output = inputs[:, 1] * (self.kernel[0] * (inputs[:, 0] ** self.kernel[1]) + self.kernel[2])
        output = array_ops.reshape(output, (array_ops.shape(output)[0], 1))
        return output

    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None, 1))
        return aux_shape[:-1].concatenate(1)


class LossModulus(Layer):
    """
    w = x[:,0]
    T = x[:,1]

    y = tt3[0]*w + tt3[1]*T + tt3[2]
    """

    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(LossModulus, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape, **kwargs):
        self.kernel = self.add_weight("kernel",
                                      shape=[3],
                                      initializer=self.kernel_initializer,
                                      dtype=self.dtype,
                                      trainable=self.trainable,
                                      **kwargs)
        self.built = True

    def call(self, inputs):
        output = inputs[:, 1] * (self.kernel[0] * (inputs[:, 0] ** self.kernel[1]) + self.kernel[2])
        output = array_ops.reshape(output, (array_ops.shape(output)[0], 1))
        return output

    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None, 1))
        return aux_shape[:-1].concatenate(1)


class Stiffness(Layer):
    """
    R3 = tt2[0]**3
    L  = tt2[1]
    e  = tt2[2]
    nu = tt2[3]

    y = np.pi*R3*L*Ep/(e*(1 + nu))
    """

    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Stiffness, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape, **kwargs):
        self.kernel = self.add_weight("kernel",
                                      shape=[1],
                                      initializer=self.kernel_initializer,
                                      dtype=self.dtype,
                                      trainable=self.trainable,
                                      constraint=self.kernel_constraint,
                                      **kwargs)
        self.built = True

    def call(self, inputs):
        output = np.pi * (0.1 ** 3) * 0.04 * inputs / (0.01 * (1 + self.kernel))
        output = array_ops.reshape(output, (array_ops.shape(output)[0], 1))
        return output

    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None, 1))
        return aux_shape[:-1].concatenate(1)


class Damping(Layer):
    """
    I = tt3[0]

    Epp = EppFN(x,tt3[1:])

    y = np.sqrt(k*I)*Epp/Ep
    """

    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Damping, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape, **kwargs):
        self.kernel = self.add_weight("kernel",
                                      shape=[1],
                                      initializer=self.kernel_initializer,
                                      dtype=self.dtype,
                                      trainable=self.trainable,
                                      constraint=self.kernel_constraint,
                                      **kwargs)
        self.built = True

    def call(self, inputs):
        output = ((inputs[:, 0] * self.kernel) ** 0.5) * inputs[:, 2] / inputs[:, 1]
        output = array_ops.reshape(output, (array_ops.shape(output)[0], 1))
        return output

    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None, 1))
        return aux_shape[:-1].concatenate(1)


class FRFAmplitude(Layer):
    """
    w2 = x[:,0]**2
    c2 = c**2
    I  = tt4

    y = w2/np.sqrt((k - I*w2)**2 + c2*w2)
    """

    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FRFAmplitude, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape, **kwargs):
        self.kernel = self.add_weight("kernel",
                                      shape=[1],
                                      initializer=self.kernel_initializer,
                                      dtype=self.dtype,
                                      trainable=self.trainable,
                                      **kwargs)
        self.built = True

    def call(self, inputs):
        output = inputs[:, 0] ** 2 / (
                    (inputs[:, 1] - self.kernel * inputs[:, 0] ** 2) ** 2 + (inputs[:, 2] * inputs[:, 0]) ** 2) ** 0.5
        output = array_ops.reshape(output, (array_ops.shape(output)[0], 1))
        return output

    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None, 1))
        return aux_shape[:-1].concatenate(1)


# Models

def arrange_table(table):
    data = np.transpose(np.asarray(np.transpose(table))[1:])
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    data = np.expand_dims(data, 0)
    data = np.expand_dims(data, -1)
    space = np.asarray([np.asarray(table.iloc[:, 0]), np.asarray([float(i) for i in table.columns[1:]])])
    table_shape = data.shape
    bounds = np.asarray([[np.min(space[0]), np.min(space[1])], [np.max(space[0]), np.max(space[1])]])
    return {'data': data, 'bounds': bounds, 'table_shape': table_shape}


def stiffness_var(radius, width, thickness, poisson, storage_modulus):
    return storage_modulus * np.pi * radius ** 3 * width / (thickness * (1 + poisson))


def damping_var(inertia, stiffness, storage_modulus, loss_modulus):
    return (loss_modulus / storage_modulus) * (stiffness * inertia) ** 0.5


def modulus_fit(freq, temp):
    storage_modulus = []
    loss_modulus = []
    for t in temp:
        for f in freq:
            if t == 20.0:
                storage_modulus.append(6.55 * f ** 0.0814 * 1e6)
                loss_modulus.append((5.79 * f ** 0.0156 - 4.81) * 1e6)
            elif t == 40.0:
                storage_modulus.append(5.68 * f ** 0.0815 * 1e6)
                loss_modulus.append((5.41 * f ** 0.0163 - 4.58) * 1e6)
            elif t == 60.0:
                storage_modulus.append(5.02 * f ** 0.0820 * 1e6)
                loss_modulus.append((6.17 * f ** 0.0141 - 5.47) * 1e6)
            elif t == 80.0:
                storage_modulus.append(4.49 * f ** 0.0808 * 1e6)
                loss_modulus.append((6.07 * f ** 0.0139 - 5.49) * 1e6)
            elif t == 100.0:
                storage_modulus.append(4.27 * f ** 0.0835 * 1e6)
                loss_modulus.append((6.34 * f ** 0.0137 - 5.81) * 1e6)
    return np.transpose(np.asarray([storage_modulus, loss_modulus]))


def create_mlp(dLInputScaling, mlp_name):
    model = Sequential([
        #            dLInputScaling,
        #            Dense(40,activation = 'elu'),
        #            Dense(20,activation = 'elu'),
        Dense(10, activation='sigmoid'),
        Dense(5, activation='elu'),
        Dense(1)
    ], name=mlp_name)
    optimizer = Adam(1e-2)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def create_simulator_model(radius, length, thickness, poisson, inertia,
                           storage_data, storage_bounds, storage_table_shape,
                           loss_data, loss_bounds, loss_table_shape,
                           select_freq, select_temp, batch_input_shape, myDtype):
    batch_adjusted_shape = (batch_input_shape[1],)
    inputLayer = Input(shape=(batch_input_shape[1],))

    freqLayer = inputsSelection(batch_adjusted_shape, select_freq)(inputLayer)
    tempLayer = inputsSelection(batch_adjusted_shape, select_temp)(inputLayer)

    omegaLayer = Lambda(lambda x: 2 * np.pi * x)(freqLayer)

    moduliInputLayer = Concatenate(axis=-1)([tempLayer, freqLayer])

    storageModulusLayer = TableInterpolation(table_shape=storage_table_shape, dtype=myDtype, trainable=False)
    storageModulusLayer.build(input_shape=moduliInputLayer.shape)
    storageModulusLayer.set_weights([storage_data, storage_bounds])
    storageModulusLayer = storageModulusLayer(moduliInputLayer)

    lossModulusLayer = TableInterpolation(table_shape=loss_table_shape, dtype=myDtype, trainable=False)
    lossModulusLayer.build(input_shape=moduliInputLayer.shape)
    lossModulusLayer.set_weights([loss_data, loss_bounds])
    lossModulusLayer = lossModulusLayer(moduliInputLayer)

    stiffnessLayer = Stiffness(input_shape=storageModulusLayer.shape, dtype=myDtype, trainable=False)
    stiffnessLayer.build(input_shape=storageModulusLayer.shape)
    stiffnessLayer.set_weights([np.asarray([poisson], dtype=stiffnessLayer.dtype)])
    stiffnessLayer = stiffnessLayer(storageModulusLayer)

    dampingInputLayer = Concatenate(axis=-1)([stiffnessLayer, storageModulusLayer, lossModulusLayer])

    dampingLayer = Damping(input_shape=dampingInputLayer.shape, dtype=myDtype, trainable=False)
    dampingLayer.build(input_shape=dampingInputLayer.shape)
    dampingLayer.set_weights([np.asarray([inertia], dtype=dampingLayer.dtype)])
    dampingLayer = dampingLayer(dampingInputLayer)

    FRFAmpInputLayer = Concatenate(axis=-1)([omegaLayer, stiffnessLayer, dampingLayer])

    FRFAmpLayer = FRFAmplitude(input_shape=FRFAmpInputLayer.shape, dtype=myDtype, trainable=False)
    FRFAmpLayer.build(input_shape=FRFAmpInputLayer.shape)
    FRFAmpLayer.set_weights([np.asarray([inertia], dtype=FRFAmpLayer.dtype)])
    FRFAmpLayer = FRFAmpLayer(FRFAmpInputLayer)

    functionalModel = Model(inputs=[inputLayer], outputs=[FRFAmpLayer])

    functionalModel.compile(loss='mean_squared_error',
                            optimizer=Adam(1e-2),
                            metrics=['mean_absolute_error', 'mean_squared_error'])
    return functionalModel


def create_physics_model(radius, length, thickness, poisson, inertia,
                         storage_data, storage_bounds, storage_table_shape,
                         loss_data, loss_bounds, loss_table_shape,
                         select_freq, select_temp, batch_input_shape, myDtype):
    batch_adjusted_shape = (batch_input_shape[1],)
    inputLayer = Input(shape=(batch_input_shape[1],))

    freqLayer = inputsSelection(batch_adjusted_shape, select_freq)(inputLayer)
    tempLayer = inputsSelection(batch_adjusted_shape, select_temp)(inputLayer)

    omegaLayer = Lambda(lambda x: 2 * np.pi * x)(freqLayer)

    moduliInputLayer = Concatenate(axis=-1)([tempLayer, freqLayer])

    storageModulusLayer = TableInterpolation(table_shape=storage_table_shape, dtype=myDtype, trainable=False)
    storageModulusLayer.build(input_shape=moduliInputLayer.shape)
    storageModulusLayer.set_weights([storage_data, storage_bounds])
    storageModulusLayer = storageModulusLayer(moduliInputLayer)

    lossModulusLayer = TableInterpolation(table_shape=loss_table_shape, dtype=myDtype, trainable=False)
    lossModulusLayer.build(input_shape=moduliInputLayer.shape)
    lossModulusLayer.set_weights([loss_data, loss_bounds])
    lossModulusLayer = lossModulusLayer(moduliInputLayer)

    stiffnessLayer = Stiffness(input_shape=storageModulusLayer.shape, dtype=myDtype, trainable=False)
    stiffnessLayer.build(input_shape=storageModulusLayer.shape)
    stiffnessLayer.set_weights([np.asarray([poisson], dtype=stiffnessLayer.dtype)])
    stiffnessLayer = stiffnessLayer(storageModulusLayer)
    artificialDelta1Layer = Lambda(lambda x: x[0] - x[1] * 10 + x[0] / x[1])([freqLayer, tempLayer])
    modifiedStiffnessLayer = Lambda(lambda x: x[0] + x[1])([stiffnessLayer, artificialDelta1Layer])

    dampingInputLayer = Concatenate(axis=-1)([modifiedStiffnessLayer, storageModulusLayer, lossModulusLayer])

    dampingLayer = Damping(input_shape=dampingInputLayer.shape, dtype=myDtype, trainable=False)
    dampingLayer.build(input_shape=dampingInputLayer.shape)
    dampingLayer.set_weights([np.asarray([inertia], dtype=dampingLayer.dtype)])
    dampingLayer = dampingLayer(dampingInputLayer)
    artificialDelta2Layer = Lambda(lambda x: x[0] / 250 - x[1] / 50 + x[0] / (10 * x[1]))([freqLayer, tempLayer])
    modifiedDampingLayer = Lambda(lambda x: x[0] + x[1])([dampingLayer, artificialDelta2Layer])

    FRFAmpInputLayer = Concatenate(axis=-1)([omegaLayer, modifiedStiffnessLayer, modifiedDampingLayer])

    FRFAmpLayer = FRFAmplitude(input_shape=FRFAmpInputLayer.shape, dtype=myDtype, trainable=False)
    FRFAmpLayer.build(input_shape=FRFAmpInputLayer.shape)
    FRFAmpLayer.set_weights([np.asarray([inertia], dtype=FRFAmpLayer.dtype)])
    FRFAmpLayer = FRFAmpLayer(FRFAmpInputLayer)

    functionalModel = Model(inputs=[inputLayer], outputs=[FRFAmpLayer])

    functionalModel.compile(loss='mean_squared_error',
                            optimizer=Adam(1e-2),
                            metrics=['mean_absolute_error', 'mean_squared_error'])
    return functionalModel


def create_calibration_model(storage_coefs, loss_coefs, radius, length, thickness, poisson, inertia,
                             storage_data, storage_bounds, storage_table_shape,
                             loss_data, loss_bounds, loss_table_shape,
                             delta_stiffness_mlp, delta_damping_mlp, stiffness_low, stiffness_up, damping_low,
                             damping_up,
                             input_min, input_range, select_freq, select_temp, batch_input_shape, myDtype):
    batch_adjusted_shape = (batch_input_shape[1],)
    inputLayer = Input(shape=(batch_input_shape[1],))

    normalizedInputLayer = Lambda(lambda x, input_min=input_min, input_range=input_range:
                                  (x - input_min) / input_range)(inputLayer)

    freqLayer = inputsSelection(batch_adjusted_shape, select_freq)(inputLayer)
    tempLayer = inputsSelection(batch_adjusted_shape, select_temp)(inputLayer)

    omegaLayer = Lambda(lambda x: 2 * np.pi * x)(freqLayer)

    moduliInputLayer = Concatenate(axis=-1)([tempLayer, freqLayer])

    storageModulusLayer = TableInterpolation(table_shape=storage_table_shape, dtype=myDtype, trainable=False)
    storageModulusLayer.build(input_shape=moduliInputLayer.shape)
    storageModulusLayer.set_weights([storage_data, storage_bounds])
    storageModulusLayer = storageModulusLayer(moduliInputLayer)

    lossModulusLayer = TableInterpolation(table_shape=loss_table_shape, dtype=myDtype, trainable=False)
    lossModulusLayer.build(input_shape=moduliInputLayer.shape)
    lossModulusLayer.set_weights([loss_data, loss_bounds])
    lossModulusLayer = lossModulusLayer(moduliInputLayer)

    stiffnessLayer = Stiffness(input_shape=storageModulusLayer.shape, dtype=myDtype, trainable=True,
                               kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=0.5, rate=1.0))
    stiffnessLayer.build(input_shape=storageModulusLayer.shape)
    stiffnessLayer.set_weights([np.asarray([poisson], dtype=stiffnessLayer.dtype)])
    stiffnessLayer = stiffnessLayer(storageModulusLayer)

    deltaStiffnessLayer = delta_stiffness_mlp(normalizedInputLayer)
    scaledDeltaStiffnessLayer = Lambda(lambda x, stiffness_low=stiffness_low, stiffness_up=stiffness_up:
                                       x * (stiffness_up - stiffness_low) + stiffness_low)(deltaStiffnessLayer)
    correctedStiffnessLayer = Lambda(lambda x: x[0] + x[1])([stiffnessLayer, scaledDeltaStiffnessLayer])

    dampingInputLayer = Concatenate(axis=-1)([correctedStiffnessLayer, storageModulusLayer, lossModulusLayer])

    dampingLayer = Damping(input_shape=dampingInputLayer.shape, dtype=myDtype, trainable=False)
    dampingLayer.build(input_shape=dampingInputLayer.shape)
    dampingLayer.set_weights([np.asarray([inertia], dtype=dampingLayer.dtype)])
    dampingLayer = dampingLayer(dampingInputLayer)

    deltaDampingLayer = delta_damping_mlp(normalizedInputLayer)
    scaledDeltaDampingLayer = Lambda(lambda x, damping_low=damping_low, damping_up=damping_up:
                                     x * (damping_up - damping_low) + damping_low)(deltaDampingLayer)
    correctedDampingLayer = Lambda(lambda x: x[0] + x[1])([dampingLayer, scaledDeltaDampingLayer])

    FRFAmpInputLayer = Concatenate(axis=-1)([omegaLayer, correctedStiffnessLayer, correctedDampingLayer])

    FRFAmpLayer = FRFAmplitude(input_shape=FRFAmpInputLayer.shape, dtype=myDtype, trainable=False)
    FRFAmpLayer.build(input_shape=FRFAmpInputLayer.shape)
    FRFAmpLayer.set_weights([np.asarray([inertia], dtype=FRFAmpLayer.dtype)])
    FRFAmpLayer = FRFAmpLayer(FRFAmpInputLayer)

    functionalModel = Model(inputs=[inputLayer], outputs=[FRFAmpLayer])

    functionalModel.compile(loss='mean_squared_error',
                            optimizer=Adam(5e-2),
                            metrics=['mean_absolute_error', 'mean_squared_error'])
    return functionalModel


if __name__ == "__main__":

    myDtype = 'float32'
    radius = 0.1
    length = 0.04
    thickness = 0.01
    poisson = 0.5
    inertia = 0.03

    MLP_EPOCHS = 1000

    storage_coefs = np.asarray([5e6, 1e-2, 0])
    loss_coefs = np.asarray([6e6, 1e-2, 5e6])

    x = np.load('./data/x.npy', allow_pickle=True)
    xOBS = np.load('./data/x_obs.npy', allow_pickle=True)

    modulus_array = modulus_fit(x[:, 0], x[:, 1])
    stiffness_array = stiffness_var(radius, length, thickness, poisson, modulus_array[:, 0])
    damping_array = damping_var(inertia, stiffness_array, modulus_array[:, 0], modulus_array[:, 1])
    stiffness_low = stiffness_array.min()
    stiffness_up = stiffness_array.max()
    damping_low = damping_array.min()
    damping_up = damping_array.max()

    input_min = x.min(axis=0)
    input_range = x.max(axis=0) - input_min
    input_scaling = getScalingDenseLayer(input_min, input_range)

    delta_stiffness_mlp = create_mlp(input_scaling, 'delta_stiffness')
    delta_damping_mlp = create_mlp(input_scaling, 'delta_damping')

    df = pd.read_csv('data/storage_modulus.csv')
    storageModulus = arrange_table(df)
    df = pd.read_csv('data/loss_modulus.csv')
    lossModulus = arrange_table(df)

    batch_input_shape = x.shape

    select_freq = [0]
    select_temp = [1]

    # Calculate outputs
    physics_model = create_physics_model(radius, length, thickness, poisson, inertia,
                                         storageModulus['data'], storageModulus['bounds'],
                                         storageModulus['table_shape'],
                                         lossModulus['data'], lossModulus['bounds'], lossModulus['table_shape'],
                                         select_freq, select_temp, batch_input_shape, myDtype)

    yOBS = physics_model.predict(xOBS)
    yHF_nongrid = np.asarray(physics_model.predict(x))

    calibrated_lf_model = create_simulator_model(radius, length, thickness, poisson, inertia,
                                                 storageModulus['data'], storageModulus['bounds'],
                                                 storageModulus['table_shape'],
                                                 lossModulus['data'], lossModulus['bounds'],
                                                 lossModulus['table_shape'],
                                                 select_freq, select_temp, batch_input_shape, myDtype)

    yLF_nongrid = np.asarray(calibrated_lf_model.predict(x))

    # Create the calibration model
    calibration_model = create_calibration_model(storage_coefs, loss_coefs, radius, length, thickness, poisson,
                                                 inertia,
                                                 storageModulus['data'], storageModulus['bounds'],
                                                 storageModulus['table_shape'],
                                                 lossModulus['data'], lossModulus['bounds'],
                                                 lossModulus['table_shape'],
                                                 delta_stiffness_mlp, delta_damping_mlp, stiffness_low,
                                                 stiffness_up, damping_low, damping_up,
                                                 input_min, input_range, select_freq, select_temp,
                                                 batch_input_shape, myDtype)

    # Fit callbacks
    ReduceLR = ReduceLROnPlateau(monitor='loss', factor=0.75, min_lr=1e-15, patience=5, verbose=1, mode='min')
    ToNaN = TerminateOnNaN()
    callbacks_list = [ReduceLR, ToNaN]

    # Model training
    history = calibration_model.fit(xOBS, yOBS, epochs=MLP_EPOCHS, verbose=1, callbacks=callbacks_list)

    # Make predictions
    yCalibrated_nongrid = np.asarray(calibration_model.predict(x))

    # Plot the results
    matplotlib.rc('font', size=14)
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(yHF_nongrid, yLF_nongrid, color='orange', label='calibrated simulator')
    plt.scatter(yHF_nongrid, yCalibrated_nongrid, color='black', label='adjusted model')
    plt.plot([0, 250], [0, 250], '--k')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.xlim(0, 250)
    plt.ylim(0, 250)
    plt.grid(which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()
