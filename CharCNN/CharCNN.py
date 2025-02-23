import tensorflow as tf
import numpy as np
import Network as nt
import DataManager as dm

NAME = "CHAR_CNN"
EPOCH_SIZE = 10000
BATCH_SIZE = 100
GLOBAL_STEP = 0

def main(_):
    with tf.Session() as sess:
        data_mgr = dm.DataManager()
        network = nt.Network(sess, NAME)
        global GLOBAL_STEP
        GLOBAL_STEP = 0

        for epoch in range(EPOCH_SIZE):
            Train(epoch, network, data_mgr)
            Accuracy(epoch, network, data_mgr)

def Train(epoch, network, data_mgr):
    print("==============", epoch, "===============")

    global GLOBAL_STEP
    data_mgr.ResetBatchTrainData()
    while True:
        datas, labels = data_mgr.GetTrainDataAndLabel(BATCH_SIZE)
        if len(datas) == 0:
            break

        network.Train(datas, labels, GLOBAL_STEP)
        GLOBAL_STEP += 1

def Accuracy(poch, network, data_mgr):
    print("=== 측정 시작 ===")

    data_mgr.ResetBatchTrainData()
    accuracy = []
    while True:
        datas, labels = data_mgr.GetTrainDataAndLabel(BATCH_SIZE)
        if len(datas) == 0:
            break

        result = network.Accuracy(datas, labels)
        result = np.reshape(result, [-1])
        accuracy.extend(result)

    print("Train :", np.mean(accuracy))



    data_mgr.ResetBatchTestData() 
    accuracy = []
    while True:
        datas, labels = data_mgr.GetTestDataAndLabel(BATCH_SIZE)
        if len(datas) == 0:
            break

        result = network.Accuracy(datas, labels)
        result = np.reshape(result, [-1])
        accuracy.extend(result)

    print("Test :",np.mean(accuracy))

if __name__ == "__main__":
    tf.app.run()
