from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import neat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Fetch dataset
credit_approval = fetch_ucirepo(id=27)

# Data
X = credit_approval.data.features
y = credit_approval.data.targets

data = pd.concat([X, y], axis=1)
data = data.dropna()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

y = y.values.ravel()
y = np.where(y == '+', 1.0, 0.0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0.0

        for xi, yi in zip(X_train, y_train):
            output = net.activate(xi)[0]
            prediction = 1.0 if output > 0.5 else 0.0

            if prediction == yi:
                genome.fitness += 1

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'C:/Users/adof1/Desktop/Ux/Quinto semestre/Redes Neuronales/Neat/config.txt'
)

p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

winner = p.run(eval_genomes, 100)
net = neat.nn.FeedForwardNetwork.create(winner, config)

correct = 0
for xi, yi in zip(X_test, y_test):
    output = net.activate(xi)[0]
    prediction = 1.0 if output > 0.5 else 0.0
    if prediction == yi:
        correct += 1

accuracy = correct / len(y_test)
print(f"\nAccuracy en test: {accuracy:.2f}")
