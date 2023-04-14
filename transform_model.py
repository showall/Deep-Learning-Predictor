import datetime as dt
import glob
import os

import pandas as pd
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
from math import exp
import pickle 


PATH_DIR = "."

def transform_and_model():

    pick_the_date = dt.datetime.now().strftime("%Y%m%d-")

    all_files = glob.glob(os.path.join("mydir",pick_the_date, "*.csv"))


    class MultipleClassification(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super(MultipleClassification, self).__init__()
            self.batchnorm1d = nn.BatchNorm1d(input_size)
            self.sigmoid = nn.Sigmoid()
            self.fc1 = nn.Linear(input_size, 4*2)
            self.dropout = nn.Dropout(p=0.6)
            self.fc2 = nn.Linear(4*2, 12)
            self.fc3 = nn.Linear(12, 8)
        #  self.fc4 = nn.Linear(16, 32)        
    #         self.fc5 = nn.Linear(256*2, 512*2)
    #         self.fc6 = nn.Linear(512*2, 512*4)
    #         self.fc7 = nn.Linear(512*4, 512*8)
            self.fc8 = nn.Linear(8, output_size)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=0)
            
        def forward(self, x):
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.relu(self.fc2(x))
            x = self.dropout(x)    
            x = self.relu(self.fc3(x))
    #         x = self.relu(self.fc4(x))
        #  x = self.dropout(x)  
    #         x = self.relu(self.fc5(x))
    #         x = self.relu(self.fc6(x))
    #         x = self.relu(self.fc7(x))
            x = self.fc8(x)
            return x   

    df = pd.DataFrame({})
    check = True
    for i in all_files:
        while check == True:
            df = pd.concat([df,pd.read_csv(i, parse_dates=["Start Time"], dayfirst=True)], axis=0, ignore_index=True)
            check = False
        df =  pd.concat([df,pd.read_csv(i, parse_dates=["Start Time"], dayfirst=True)], axis=0, ignore_index=True)

    df.Result = df.Result.str.split(",")
    df = df[[len(x)==2 for x in df.Result]]
    df.Result = df.Result.apply(lambda x: x[1]).copy()
    df.Result = df.Result.str.replace("FT:","")
    df.Result = df.Result.str.strip()
    df["home_score"] = df.Result.apply(lambda x: int(x.split("-")[0]))
    df["away_score"] = df.Result.apply(lambda x: int(x.split("-")[1]))
    df["total_goals"] = df["home_score"]+df["away_score"]
    df["home_netgoals"] = df["home_score"]-df["away_score"]
    df["away_netgoals"] = -df["home_score"]+df["away_score"]

    def net_score(score_line):
        if score_line < 0.5:
            label = 0
        else:
            label = 1
        return label

    df["Ah_home_win"] = df["home_netgoals"].apply(lambda x: net_score(x))

    what_we_need = [
                    "Ah_01_Hcap",
                    "Ah_01",
                    "Ah_02_Hcap",
                    "Ah_02",
                    "Ft1X2_01",
                    "Ft1X2_02",
                    "Ft1X2_03",
                    "Htft_01",
                    "Htft_02",
                    "Htft_03",
                    "Htft_04",
                    "Htft_05",
                    "Htft_06",
                    "Htft_07",
                    "Htft_08",
                    "Htft_09",
                    "Tg_00",
                    "Tg_01",
                    "Tg_02",
                    "Tg_03",
                    "Tg_04",
                    "Tg_05",
                    "Tg_06",
                    "Tg_07",
                    "Tg_08",
                    "Tg_09",
                    "Ou_hcap",
                    "Ou_01",
                    "Ou_02",
                    "Htou_Hcap",
                    "Htou_01", 
                    "Htou_02",
                    "Bg_01",
                    "Bg_02",
                    "Httg_00",
                    "Httg_01",
                    "Httg_02",
                    "Httg_03",
                    "Htbg_01",
                    "Htbg_02"
    ]


    df = df.drop_duplicates()
    label_column = "Ah_home_win"
    df = pd.concat([df[df[label_column]==0][0-min(df[label_column].value_counts()):],df[df[label_column]==1][0-min(df[label_column].value_counts()):]], axis=0, ignore_index=True)
    sc = StandardScaler()

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')


    # USE FOR CALIBRATION
    last_seven_day = dt.datetime.now() - dt.timedelta(days=7)
    last_seven_day = dt.datetime(last_seven_day.year, last_seven_day.month, last_seven_day.day)
    last_seven_day
    df_calibrate = df[df["Start Time"] <= last_seven_day]
    df_display = df[df["Start Time"] > last_seven_day]

    test_ratio = 0.5
    test_index = torch.randint(0,len(df_calibrate), (int(test_ratio*len(df_calibrate)),))
    test_index = df_calibrate.index[list(test_index)]

    df_calibrate = df_calibrate.dropna()
    #df = df[["total_goals"]+what_we_need]
    df_calibrate = df_calibrate[["Ah_home_win"]+list(what_we_need)]
    len(df_calibrate)

    tensor_input = torch.tensor(df_calibrate.drop([label_column], axis=1).values, dtype=torch.float32)
    tensor_input.shape
    input_size = tensor_input.shape[1]
    output_size = len(set(df_calibrate[label_column].values))
    batch_size = 50
    #tensor_label =torch.tensor(df["Result"] .values,  dtype=torch.int64)
    tensor_label =torch.tensor(df_calibrate[label_column] .values,  dtype=torch.int64)

    df_display = df_display.dropna()
    #df = df[["total_goals"]+what_we_need]
    df_display = df_display[["Ah_home_win"]+list(what_we_need)]

    df_train = df_calibrate[df_calibrate.index.isin(list(test_index))==False]
    df_test = df_calibrate[df_calibrate.index.isin(list(test_index))==True]

    X_train = df_train.drop([label_column], axis=1)
    y_train = df_train[label_column].values

    X_test = df_test.drop([label_column], axis=1)
    y_test = df_test[label_column].values

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = MultipleClassification(input_size,output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.002)

    train_accuracy_list =[]
    test_accuracy_list = []
    high_test_accuracy_list = []

    # best = 0.8
    # consec = 0
    # for i in range(10000):
    #     correct = 0
    #     df_batch_count = 0
    #     loss_amount = 0
    #     for j in range(int(round(len(df_train)/batch_size,0))):
    #         train_input = torch.tensor(X_train[j*batch_size:(j+1)*batch_size], dtype=torch.float32)
    #         train_label =torch.tensor(y_train[j*batch_size:(j+1)*batch_size],  dtype=torch.int64) 
    #     # train_input = torch.tensor(df_batch.iloc[:,6:-2] .values, dtype=torch.float32)
    # #         train_input = torch.tensor(df_batch.drop([label_column], axis=1).values, dtype=torch.float32)
    # #         train_label =torch.tensor(df_batch[label_column] .values,  dtype=torch.int64)    
    #         output = model(train_input)
    #         loss = criterion(output, train_label.squeeze(0))
    #         loss_amount += loss.item() 
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         _, predicted = torch.max(output.data, 1)
    #         correct = correct + (predicted == train_label).sum().item()
    #         df_batch_count= df_batch_count + len(train_label)
    #         train_accuracy = correct / df_batch_count
    #     average_loss = loss_amount/j
    #     with torch.no_grad():
    #         model.eval()
    #         rand = torch.randint(0, len(X_test), (150,))
    #         test_input = torch.tensor(X_test[rand], dtype=torch.float32)
    # #        test_input = torch.tensor(df_test.drop([label_column], axis=1).values, dtype=torch.float32)
    #     # normalize_test_input = normalize(test_input)
    #         test_label = torch.tensor(y_test[rand],  dtype=torch.int64) 
    #         test_output = model(test_input)
    #         loss = criterion(test_output, test_label.squeeze(0))
    #         _t, test_predicted = torch.max(test_output.data, 1)
            
    #         x_list = [[exp(x[0])/(exp(x[0])+exp(x[1])),exp(x[1])/(exp(x[0])+exp(x[1]))] for x in test_output]
    #         # output
    #         x_list = np.max(np.array(x_list), axis=1)       
    #         check = float(np.sort(x_list)[::-1][int(len(x_list)*0.15)-1:int(len(x_list)*0.15)])
    #         test_correct = (test_predicted == test_label).sum().item()
    #         test_correct_9 = ((test_predicted == test_label) & (torch.tensor(x_list)>check)).sum().item()
    #         test_accuracy =  test_correct / len(test_label)
    #         try:
    #             test_accuracy_9 =  test_correct_9 / (torch.tensor(x_list)>check).sum().item()
    #         except:
    #             test_accuracy_9 = 0
    #     train_accuracy_list.append(100*train_accuracy)
    #     test_accuracy_list.append(100*test_accuracy)
    #     high_test_accuracy_list.append(100*test_accuracy_9)
    #     print("Epoch %d Train Accuracy : %d %% (Loss %.2f): ; Test Accuracy : %d %%(%d %%): " %(i,100*train_accuracy, average_loss,100*test_accuracy,100*test_accuracy_9 ))

    #     if test_accuracy_9 >= best and consec ==2 and test_accuracy>0.70:
    #         model_scripted = torch.jit.script(model) # Export to TorchScript
    #         model_title = dt.datetime.now().strftime("%Y-%m-%d")
    #         model_scripted.save(f'{PATH_DIR}/models/model_scripted-{model_title}.pt')
    #         with open(f'{PATH_DIR}/models/scaler-{model_title}.pkl','wb') as f:
    #             pickle.dump(sc, f)
    #         break

    #     elif test_accuracy_9 >= best and consec <2:
    #         consec += 1
    #     else: 
    #         consec = 0
    #     if test_accuracy_9 > 0.9:   
    #         best = test_accuracy_9

    best = 0.8
    consec = 0
    threshold = 0.60
    test_correct_9 = 0
    test_correct = 0

    count_correct_9 = 0
    count_correct = 0

    for i in range(5000):
        correct = 0
        df_batch_count = 0
        loss_amount = 0
        
        for j in range(int(round(len(df_train)/batch_size,0))):
            train_input = torch.tensor(X_train[j*batch_size:(j+1)*batch_size], dtype=torch.float32)
            train_label =torch.tensor(y_train[j*batch_size:(j+1)*batch_size],  dtype=torch.int64) 
        # train_input = torch.tensor(df_batch.iloc[:,6:-2] .values, dtype=torch.float32)
    #         train_input = torch.tensor(df_batch.drop([label_column], axis=1).values, dtype=torch.float32)
    #         train_label =torch.tensor(df_batch[label_column] .values,  dtype=torch.int64)    
            output = model(train_input)
            loss = criterion(output, train_label.squeeze(0))
            loss_amount += loss.item() 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, 1)
            correct = correct + (predicted == train_label).sum().item()
            df_batch_count= df_batch_count + len(train_label)
            train_accuracy = correct / df_batch_count
        average_loss = loss_amount/j
        
        with torch.no_grad():
            model.eval()
            rand = torch.randint(0, len(X_test), (150,))
            test_input = torch.tensor(X_test[rand], dtype=torch.float32)
    #        test_input = torch.tensor(df_test.drop([label_column], axis=1).values, dtype=torch.float32)
        # normalize_test_input = normalize(test_input)
            test_label = torch.tensor(y_test[rand],  dtype=torch.int64) 
            test_output = model(test_input)
            loss = criterion(test_output, test_label.squeeze(0))
            _t, test_predicted = torch.max(test_output.data, 1)
            
            x_list = [[exp(x[0])/(exp(x[0])+exp(x[1])),exp(x[1])/(exp(x[0])+exp(x[1]))] for x in test_output]
            # output
            x_list = np.max(np.array(x_list), axis=1)       
            check = float(np.sort(x_list)[::-1][int(len(x_list)*0.08)-1:int(len(x_list)*0.08)])
            print(check)
            test_correct = test_correct + (test_predicted == test_label).sum().item()
            test_correct_9 = test_correct_9 + ((test_predicted == test_label) & (torch.tensor(x_list)>check)).sum().item()
            count_correct = count_correct + len(test_label)
            count_correct_9 = count_correct_9 + (torch.tensor(x_list)>check).sum().item()      
            
            test_accuracy =  test_correct /  count_correct 
            try:
                test_accuracy_9 =  test_correct_9 / count_correct_9
            except:
                test_accuracy_9 = 0
        train_accuracy_list.append(100*train_accuracy)
        test_accuracy_list.append(100*test_accuracy)
        high_test_accuracy_list.append(100*test_accuracy_9)
       # print("Epoch %d Train Accuracy : %d %% (Loss %.2f): ; Test Accuracy : %d %%(%d %%): " %(i,100*train_accuracy, average_loss,100*test_accuracy,100*test_accuracy_9 ))

        if test_accuracy_9 >= best and consec ==2 and test_accuracy>threshold:
            model_scripted = torch.jit.script(model) # Export to TorchScript
            model_title = dt.datetime.now().strftime("%Y-%m-%d")
            model_scripted.save(f'{PATH_DIR}/models/model_scripted-{model_title}.pt')
            with open(f'{PATH_DIR}/models/scaler-{model_title}.pkl','wb') as f:
                pickle.dump(sc, f)
            #break 
            print("Model Saved")
            print("Epoch %d Train Accuracy : %d %% (Loss %.2f): ; Test Accuracy : %d %%(%d %%): " %(i,100*train_accuracy, average_loss,100*test_accuracy,100*test_accuracy_9 ))
            #break 
            threshold = test_accuracy +0.01
        elif test_accuracy_9 >= best and consec <2:
            consec += 1
        else: 
            consec = 0
        if test_accuracy_9 > 0.8:   
            best = test_accuracy_9


    try:
        model_saved = torch.jit.load(f'{PATH_DIR}/models/model_scripted-{model_title}.pt')
    except:
        model_saved = model
    model_saved.eval()
    required_confidences = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
    data = dict()
    subdata = dict()
    for required_confidence in required_confidences:
        subdata[required_confidence] = dict()
        total_trades = 0
        total_confident_guessed = 0
        total_confident_correct = 0
        for x in range(100):
            with torch.no_grad():
                model_saved.eval()
                rand = torch.randint(0, len(X_test), (50,))
                test_input = torch.tensor(X_test[rand], dtype=torch.float32)
        #        test_input = torch.tensor(df_test.drop([label_column], axis=1).values, dtype=torch.float32)
            # normalize_test_input = normalize(test_input)
                test_label = torch.tensor(y_test[rand],  dtype=torch.int64) 
                test_output = model_saved(test_input)
                loss = criterion(test_output, test_label.squeeze(0))
                _t, test_predicted = torch.max(test_output.data, 1)
                total_trades = total_trades + len(test_label)
                x_list = [[exp(x[0])/(exp(x[0])+exp(x[1])),exp(x[1])/(exp(x[0])+exp(x[1]))] for x in test_output]
                # output
                x_list = np.max(np.array(x_list), axis=1)       

                test_correct = (test_predicted == test_label).sum().item()

                test_correct_9 = ((test_predicted == test_label) & (torch.tensor(x_list)>required_confidence)).sum().item()
                test_accuracy =  test_correct / len(test_label)

                try:
                    test_accuracy_9 = test_correct_9 / (torch.tensor(x_list)>required_confidence).sum().item()
                except:
                    test_accuracy_9 = 0
                total_confident_correct = total_confident_correct + test_correct_9 
                total_confident_guessed = total_confident_guessed + (torch.tensor(x_list)>required_confidence).sum().item()
                try:
                    total_confident_accuracy = 100*total_confident_correct / total_confident_guessed
                except:
                    total_confident_accuracy = 0 
                subdata[required_confidence]["total_confident_accuracy"] = total_confident_accuracy
                subdata[required_confidence]["total_confident_guessed"] = total_confident_guessed           
                subdata[required_confidence]["total_trades"] = total_trades          
                
            print("Test Accuracy : %d %%(%d %% - overall %d %% from %d confident guesses out of %d):" \
                %(100*test_accuracy,100*test_accuracy_9, total_confident_accuracy,total_confident_guessed, total_trades))

            data["train_accuracy_list"] = train_accuracy_list
            data["test_accuracy_list"] = test_accuracy_list
            data["high_test_accuracy_list"] = high_test_accuracy_list
            data["data"] =subdata
            with open(f'{PATH_DIR}/assets/training_result.json', 'w', encoding ='utf8') as json_file:
                json.dump(data, json_file, ensure_ascii = False)
    return data

if __name__ == '__main__':
    transform_and_model()