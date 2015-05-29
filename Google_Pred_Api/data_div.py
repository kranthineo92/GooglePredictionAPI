__author__ = 'KranthiDhanala'

import pandas as pd
from string import ascii_uppercase
from collections import OrderedDict
def main():

    letters_data = pd.read_csv("Data/letter-recognition.data",header = None)


    alphabets = OrderedDict((ch, idx) for idx, ch in enumerate(ascii_uppercase, 1))
    letters_data[0] = letters_data[0].apply(lambda x: alphabets[x])
    length = len(letters_data)
    train_length = int(length * 0.8)
    train_data = letters_data[0:train_length]
    test_data = letters_data[train_length+1:length]
    """
    #to check how good is the split

    for i  in range(1,27):
        print len(letters_data[letters_data[0]==i])
        print len(train_data[train_data[0]==i]), "  ", len(test_data[test_data[0]==i])
    """

    #write to csv
    train_data.to_csv("train_data.csv",sep=",",header=False,index=False)
    test_data[0].to_csv("test_pred.csv",sep=",",header = False,index=False)
    test_data = test_data.drop([test_data.columns[0]], axis=1)
    test_data.to_csv("test_data.csv",sep=",",header=False,index=False)



    return

if __name__ == "__main__":
    main()
