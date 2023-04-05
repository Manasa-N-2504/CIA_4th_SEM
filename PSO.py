import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data = pd.read_csv(r"D:\4TH SEMESTER\NN\Bank_Personal_Loan_Modelling.csv").drop(['ID'], axis=1)
x = data.drop(['Personal Loan'], axis=1).values
y = to_categorical(data['Personal Loan'].values)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.25)

from keras import layers
class NN(tf.keras.Model):
    def __init__(self,input_lay,hidden_lay1,hidden_lay2,output_lay):
        super().__init__()
        self.nnlayers = keras.Sequential([
            layers.Dense(hidden_lay1, activation='relu', input_shape=(input_lay,)),
            layers.Dense(hidden_lay2, activation='relu'),
            layers.Dense(output_lay, activation='sigmoid')
        ])

    def call(self, inputs):
        return self.nnlayers(inputs)

model = NN(9,6,3,1)
loss_function = keras.losses.MeanSquaredError()

class PSO:
    def __init__(self, model, w, c1, c2, num_of_particles, decay , inputs, labels):
        self.model, self.w, self.c1, self.c2, self.num_of_particles, self.inputs, self.labels = model, w, c1, c2, num_of_particles, inputs, labels
        self.positions, self.velocity, self.personal_best = (10 * np.random.rand(self.num_of_particles, sum(p.numel() for p in self.model.weights)) - 0.5), (np.random.rand(self.num_of_particles, sum(p.numel() for p in self.model.weights)) - 0.5), self.positions
        self.global_best, self.decay = np.inf, decay

    def find_personal_best(self):
        self.personal_best = np.where(self.fitness(self.positions) > self.fitness(self.personal_best), self.personal_best, self.positions)

    def find_global_best(self):
        self.global_best = np.where(self.fitness(self.positions) < self.fitness(self.global_best), self.positions, self.global_best)

    def new_velocity(self):
        self.velocity = self.w * self.velocity + self.c1 * np.random.rand(self.num_of_particles, self.velocity.shape[1]) * (self.personal_best - self.positions) + self.c2 * np.random.rand(self.num_of_particles, self.velocity.shape[1]) * (self.global_best - self.positions)

    def new_position(self):
        self.positions += self.velocity

    def fitness(self, weights):
        self.model.set_weights(weights.reshape(self.model.get_weights()[0].shape, self.model.get_weights()[1].shape))
        outputs = self.model(self.inputs)
        loss = tf.keras.losses.binary_crossentropy(self.labels, outputs)
        return loss.numpy()

    def update_weights(self):
        self.find_personal_best()
        self.find_global_best()
        self.new_velocity()
        self.new_position()
        fitness_scores = [self.fitness(weights) for weights in self.positions]
        best_index = np.argmin(fitness_scores)
        best_weights = self.positions[best_index]
        self.model.set_weights(best_weights.reshape(self.model.get_weights()[0].shape, self.model.get_weights()[1].shape))

    def decay_w(self):
        self.w = self.w - (self.w * self.decay)

model = keras.Sequential([layers.Dense(1, activation='sigmoid', input_shape=(x_train.shape[1],))])

particleswarmOptimizer = PSO(model, w=0.8, c1=0.1, c2=0.1, num_of_particles=20, decay=0.05, inputs=x_train, labels=y_train)

def optimize(num_epochs):
    loss_list = []
    for epoch in range(num_epochs):
        particleswarmOptimizer.update_weights()
        outputs = model(x_train.float())
        loss = tf.keras.losses.binary_crossentropy(y_train.reshape([len(x_train), 1]).float(), outputs.float())
        loss_list.append(loss.numpy().mean())
        if epoch % 10 == 0:
            print("Epoch", epoch, ": ", loss.numpy().mean())
            particleswarmOptimizer.decay_w()
    return loss_list
