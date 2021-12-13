import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix
from NeuralNetwork3 import MultiLayerPerceptron

def get_accuracy():

    predictions = pd.read_csv('test_predictions.csv').values
    true_values = pd.read_csv('test_label.csv').values


    print(accuracy_score(true_values, predictions))
    print(precision_score(true_values, predictions, average='micro'))
    # print(recall_score(true_values, predictions, average='micro'))
    # print(confusion_matrix(predictions, true_values))
    # N = true_values.shape[0]
    # accuracy = (true_values == predictions).sum() / N
    # TP = ((predictions == 1) & (true_values == 1)).sum()
    # FP = ((predictions == 1) & (true_values == 0)).sum()
    # precision = TP / (TP+FP)
    # print(accuracy)
    # print(precision)



for bs in [16,32,64,128]:
    for lr in [0.1, 0.5, 0.01, 0.05]:
        for ep in [50, 60, 70, 80, 90]:
            print('{} {} {}'.format(bs, lr, ep) )
            nn = MultiLayerPerceptron(batch_size=bs, learning_rate=lr, num_epochs=ep)
            trainX, onehotY, testX = nn.start()
            nn.train(trainX, onehotY)
            out = nn.forward_pass(testX)
            output = out['A3']
            pred = np.argmax(output, axis=0)
            pd.DataFrame(pred).to_csv('test_predictions.csv', header=None, index=None)
            get_accuracy()