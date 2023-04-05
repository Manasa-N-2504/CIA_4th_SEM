import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras import layers

from sklearn.model_selection import train_test_split

data = pd.read_csv(r"D:\4TH SEMESTER\NN\Bank_Personal_Loan_Modelling.csv").drop(['ID'], axis=1)
x = data.drop(['Personal Loan'], axis=1).values
y = data['Personal Loan'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.25)

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
loss_function = tf.keras.losses.MeanSquaredError()

class GA:
    def __init__(self, model, population_size, mutation_rate, decay_rate, x, y):
        self.model = model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.primary_population()
        self.decay_rate = decay_rate
        self.x = x
        self.y = y

    def primary_population(self):
        return [self.model.get_weights() for _ in range(self.population_size)]

    def select(self, scores):
        return np.searchsorted(np.cumsum(scores), np.random.uniform(0, np.sum(scores)))

    def cross(self, male, female):
        pt = np.random.randint(1, len(male))
        return male[:pt] + female[pt:], female[:pt] + male[pt:]

    def decay_mutation_rate(self):
        self.mutation_rate -= self.decay_rate * self.mutation_rate

    def mutate(self, child):
        return [weight + np.random.normal(0, 0.1, weight.shape) if np.random.uniform(0, 1) < self.mutation_rate else weight for weight in child]

    def generate_offspring(self, scores):
        self.population = [self.mutate(child1) + self.mutate(child2)
                           for child1, child2 in [self.cross(self.population[self.select(scores)],
                                                                  self.population[self.select(scores)])
                                                 for i in range(self.population_size)]]

    def update_weights(self):
        best_weights = self.population[np.argmax([self.fitness(weights) for weights in self.population])]
        self.model.set_weights(best_weights)

    def fitness(self, weights):
        self.model.set_weights(weights)
        loss = self.model.evaluate(self.x, self.y, verbose=0)
        return 1 / (loss + 1e-6)


genetic_optimizer = GA(model, population_size=20, mutation_rate=0.3, decay_rate=0.05, x=x_train, y=y_train)

def optimize(self, epochs):
    loss_list = []
    for epoch in range(epochs):
        self.generate_offspring([])
        self.update_weights()
        y_pred = self.model.predict(self.x)
        loss = tf.keras.losses.mean_squared_error(self.y.reshape([len(self.x), 1]).astype('float32'), y_pred)
        loss_list.append(loss.numpy().item())
        gradients = tf.gradients(loss, self.model.trainable_variables)
        self.generate_offspring([])
        self.update_weights()
        for var, grad in zip(self.model.trainable_variables, gradients):
            var.assign_sub(grad * 0.01)
        if epoch % 10 == 0:
            print("Epoch", epoch, ":", loss.numpy().item())
            self.decay_mutation_rate()
    return loss_list
