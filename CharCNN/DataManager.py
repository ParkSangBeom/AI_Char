import numpy as np
from CharParser_Kor import String_To_OneHot

class DataInfo:
    _max_text_length = 400

    def __init__(self, data, label):
        self._SetData(data)
        self._SetOneHotData(data)
        self._SetLabel(label)
        
    def _SetData(self, data):
        self._data = data

    def _SetOneHotData(self, data):
        one_hot = String_To_OneHot(data)
        one_hot = self._AddZeroPad(one_hot)
        self._one_hot = one_hot

    def _SetLabel(self, label):
        self._label = [label]

    def _AddZeroPad(self, one_hot):
        zero_pad = np.zeros((self._max_text_length), dtype = np.int32)
        for i, one in enumerate(one_hot):
            if i == self._max_text_length:
                break

            zero_pad[i] = one

        return zero_pad

    def GetData(self):
        return np.array(self._data)

    def GetOneHotData(self):
        return np.array(self._one_hot)

    def GetLabel(self):
        return np.array(self._label)

class DataManager:
    _train_data_path = "./Data/ratings_train.txt"
    _test_data_path = "./Data/ratings_test.txt"

    def __init__(self):
        self._train_info = self._ReadData(self._train_data_path)
        self._test_info = self._ReadData(self._test_data_path)
        self.ResetBatchTrainData()
        self.ResetBatchTestData()

    def _ReadData(self, path):
        data_info_container = []
        index = 0
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                line = line.replace("\n", "")
                s_line = line.split('\t')

                # [0].ID, [1].Data, [2].Label
                info = DataInfo(s_line[1], s_line[2])
                data_info_container.append(info)

                #if index == 30000:
                #    break

                index += 1

        return data_info_container

    def _GetDataAndLabel(self, infos, start_batch_cnt, batch_cnt):
        data_list = []
        label_list = []
        for i in range(batch_cnt):
            index = start_batch_cnt + i
            if len(infos) <= index:
                break
            
            info = infos[index]

            # Data Add.
            one_hot = info.GetOneHotData()
            data_list.append(one_hot)

            # Label Add.
            label = info.GetLabel()
            label_list.append(label)

        return np.array(data_list), np.array(label_list)

    def GetTrainDataAndLabel(self, batch_cnt): 
        datas, labels = self._GetDataAndLabel(self._train_info, self._train_batch_cnt, batch_cnt)
        self._train_batch_cnt += batch_cnt

        return datas, labels

    def ResetBatchTrainData(self):
        self._train_batch_cnt = 0

    def GetTestDataAndLabel(self, batch_cnt):
        datas, labels = self._GetDataAndLabel(self._test_info, self._test_batch_cnt, batch_cnt)
        self._test_batch_cnt += batch_cnt

        return datas, labels

    def ResetBatchTestData(self):
        self._test_batch_cnt = 0