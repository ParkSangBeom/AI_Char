import numpy as np
import hgtk
import os
#from konlpy.tag import Komoran

class DataInfo:
    def __init__(self, data, score, process):
        self._SetData(data, process)
        self._SetScore(score)

    def _SetData(self, data, process):
        if process:
            remove_char = "~([.,!?\"':;)(]%)><"
            for r in remove_char:
                data = data.replace(r, "")

            data = hgtk.text.decompose(data)
            data = data.replace("ᴥ", "")

        self._data = data

    def _SetScore(self, score):
        self._score = score

    def GetData(self):
        return self._data

    def GetScore(self):
        return self._score

class DataConverter:
    _define_path = "./Data/Define.txt"
    _train_data_path = "./Data/ratings_train.txt"
    _test_data_path = "./Data/ratings_test.txt"
    _train_process_data_path = "./Data/Train_Procee_Data.txt"
    _test_process_data_path = "./Data/Test_Procee_Data.txt"

    _cur_train_batch = 0
    _train_data_info = dict()

    _cur_test_batch = 0
    _test_data_info = dict()

    _idx_to_char = dict()
    _char_to_idx = dict()

    def __init__(self):
        # Define.
        self._idx_to_char, self._char_to_idx = self._ReadDefineData(self._define_path)

        # Train.
        if os.path.isfile(self._train_process_data_path):
            self._train_data_info= self._ReadDataInfoTextFile(self._train_process_data_path)
        else:
            self._train_data_info = self.CreateDataInfo(self._train_data_path)
            self._CreateTextFile(self._train_process_data_path, self._train_data_info)

        # Text.
        if os.path.isfile(self._test_process_data_path):
            self._test_data_info = self._ReadDataInfoTextFile(self._test_process_data_path)
        else:
            self._test_data_info = self.CreateDataInfo(self._test_data_path)
            self._CreateTextFile(self._test_process_data_path, self._test_data_info)

    def _ReadDefineData(self, path):
        idx_to_char = dict()
        char_to_idx = dict()
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                line = line.replace("\n", "")
                lines = line.split('\t')
                idx_to_char[lines[0]] = lines[1]
                char_to_idx[lines[1]] = lines[0]

        return idx_to_char, char_to_idx

    def CreateDataInfo(self, path):
        dic = dict()
        index = 0
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                line = line.replace("\n", "")
                lines = line.split('\t')
                data = DataInfo(lines[1], lines[2], True)
                dic[index] = data
                index += 1

        return dic

    def _CreateTextFile(self, save_path, info):
        with open(save_path, 'w', encoding='UTF-8')  as f:
            for i in range(len(info)):
                text = str(info[i].GetData()) + "\t" + str(info[i].GetScore())

                if i != len(info) - 1:
                    text += "\n"

                f.write(text)

        print(save_path + " CreateTextFile!!")

    def _ReadDataInfoTextFile(self, path):
        dic = dict()
        index = 0
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                line = line.replace("\n", "")
                lines = line.split('\t')
                data = DataInfo(lines[0], lines[1], False)
                dic[index] = data
                index += 1

        return dic

    def GetTrainDataInfo(self, batch_size):
        datainfo = []
        for _ in range(batch_size):
            #if len(self._train_data_info) == self._cur_train_batch:
            #    break

            if 50000 == self._cur_train_batch:
                break

            datainfo.append(self._train_data_info[self._cur_train_batch])
            self._cur_train_batch += 1

        return datainfo;

    def ResetBatchTrainData(self):
        self._cur_train_batch = 0

    def GetTestDataInfo(self, batch_size):
        datainfo = []
        for _ in range(batch_size):
            #if len(self._test_data_info) == self._cur_test_batch:
            #    break

            if 100 == self._cur_test_batch:
                break

            datainfo.append(self._test_data_info[self._cur_test_batch])
            self._cur_test_batch += 1

        return datainfo;

    def ResetBatchTestData(self):
        self._cur_test_batch = 0

    def GetDataCharIndex(self, data):
        idx = np.zeros([625])
        chars = data.GetData()
        index = 0
        for c in chars:
            if c in self._char_to_idx:
                idx[index] = self._char_to_idx[c]
                index += 1

        return idx












    #def _CreateCharData(self):
    #    train_morpes_data = self._GetMorphChars(self.train_data)
    #    self._CreateTextFile(self._train_morphs_data_path, train_morpes_data)
        
    #    test_morpes_data = self._GetMorphChars(self.test_data)
    #    self._CreateTextFile(self._test_morphs_data_path, test_morpes_data)

    #def _GetMorphChars(self, data):
    #    print("Create Start!")

    #    komoran = Komoran()
    #    chars = []
    #    chars.append(" ")

    #    for line in data:
    #        text = data.split('\t')[1]
    #        morphs = self.GetMorphe(text, komoran)
    #        for mor in morphs:
    #            if mor not in chars:
    #                chars.append(mor)

    #    return chars

    #def GetMorphe(self, text, komoran):
    #    text = komoran.morphs(text)
    #    text = self._DeleteSpecialText(text)
    #    return text

    #def _DeleteSpecialText(self, text):
    #    remove_char = "([.,!?\"':;)(])";
    #    add_text = []
    #    for t in text:
    #        replace_text = t;
    #        for r in remove_char:
    #            replace_text = replace_text.replace(r, "")

    #        if replace_text != "":
    #            add_text.append(replace_text)

    #    return add_text

    #def _CreateTextFile(self, save_path, data):
    #    w_text = open(save_path, 'w', encoding='UTF-8') 
    #    for i in range(len(data)):
    #        text = str(i) + "\t" + data[i]

    #        if i != len(data) - 1:
    #            text += "\n"

    #        w_text.write(text)

    #    print(save_path + " CreateTextFile!!")

    #def _LoadMorpeData(self):
    #    self.train_idx2data = self._DataFab(self._train_morphs_data_path)
    #    self.train_data2idx = self._GetChar2Idx(self.train_idx2data)

    #    self.test_idx2data = self._DataFab(self._test_morphs_data_path)
    #    self.test_data2idx = self._GetChar2Idx(self.test_idx2data)

    #def _DataFab(self, path):
    #    data = dict()
    #    with open(path, 'r', encoding='utf8') as f:
    #        for line in f:
    #            line_s = line.split('\t')
    #            line_s[0] = line_s[0].replace("\n", "")
    #            line_s[1] = line_s[1].replace("\n", "")
    #            data[line_s[0]] = line_s[1]

    #    return data

    #def _GetChar2Idx(self, data):
    #    char2Idx = dict()
    #    for i in data:
    #        char2Idx[data[i]] = i

    #    return char2Idx;