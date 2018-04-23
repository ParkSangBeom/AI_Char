import tensorflow as tf
import numpy as np
import Network as nt
import DataConverter as dc

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

NAME = "CHAR_CNN"
BATCH_SIZE = 100

def main(_):
    with tf.Session() as sess:
        dataConverter = dc.DataConverter()
        network = nt.Network(sess, NAME, dataConverter)

        for epoch in range(20):
            Train(epoch, network, dataConverter)
            Accuracy(network, dataConverter)

def Train(epoch, network, dataConverter):
    print("==============", epoch, "===============")
    dataConverter.ResetBatchTrainData()
    while True:
        opinions = []
        scores = []
        train_data = dataConverter.GetTrainDataInfo(BATCH_SIZE)
        if len(train_data) == 0:
            break;

        for data in train_data:
            idx = dataConverter.GetDataCharIndex(data)
            opinions.append(idx)
            scores.append([data.GetScore()])

        network.Train(opinions, scores)

def Accuracy(network, dataConverter):
    print("=== 측정 시작 ===")
    dataConverter.ResetBatchTestData()
    accuracy = []
    while True:         
        testOpnion = []
        testScore = []
        test_data = dataConverter.GetTestDataInfo(100)
        if len(test_data) == 0:
            print(np.mean(accuracy))
            break;
                
        for data in test_data:
            idx = dataConverter.GetDataCharIndex(data)
            testOpnion.append(idx)
            testScore.append([data.GetScore()])

        result = network.Accuracy(testOpnion, testScore)
        accuracy.extend(result)

if __name__ == "__main__":
    tf.app.run()
