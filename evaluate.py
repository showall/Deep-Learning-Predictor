import datetime as dt
import glob
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler

from math import exp
import pickle 
from transform_model import PATH_DIR
test_data = dict()

def test():

    pick_the_date = dt.datetime.now().strftime("%Y%m%d-")

    all_files = glob.glob(os.path.join("mydir",pick_the_date, "*.csv"))


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

    label_column = "Ah_home_win"

    # USE FOR CALIBRATION
    last_seven_day = dt.datetime.now() - dt.timedelta(days=7)
    last_seven_day = dt.datetime(last_seven_day.year, last_seven_day.month, last_seven_day.day)
    last_seven_day
    df_calibrate = df[df["Start Time"] <= last_seven_day]
    df_display = df[df["Start Time"] > last_seven_day]

    df_display = df_display.dropna()
    #df = df[["total_goals"]+what_we_need]
    df_display = df_display[["Ah_home_win"]+list(what_we_need)]

    list_of_files = glob.glob(f'{PATH_DIR}/models/*.pt') 
    latest_model = max(list_of_files, key=os.path.getctime).split("\\")[-1]
    previous_model = torch.jit.load(f'{PATH_DIR}/models/{ latest_model }')
    previous_model.eval()

    # previous_model_title = dt.datetime(2023,1,31).strftime("%Y-%m-%d") 
    # previous_model = torch.jit.load(f'{PATH}/models/model_scripted-{ previous_model_title }.pt')
    # previous_model.eval()

    list_of_files = glob.glob(f'{PATH_DIR}/models/*.pkl') 
    latest_model_sc = max(list_of_files, key=os.path.getctime).split("\\")[-1]

    with open(f'{PATH_DIR}/models/{ latest_model_sc  }','rb') as f: 
        previous_sc = pickle.load(f)

    X_eval = df_display.drop([label_column], axis=1)
    y_eval = df_display[label_column].values

    X_eval = previous_sc.transform(X_eval)



    df_redisplay = df[df["Start Time"] > last_seven_day].dropna()

    _input = torch.tensor(X_eval, dtype=torch.float32)
    _label =torch.tensor(y_eval,  dtype=torch.int64)

    output = previous_model(_input)

    x_list = [[exp(x[0])/(exp(x[0])+exp(x[1])),exp(x[1])/(exp(x[0])+exp(x[1]))] for x in output]
    x_list = np.max(np.array(x_list), axis=1)
    required_confidence =float (np.sort(x_list)[::-1][int(len(x_list)*0.08)-1:int(len(x_list)*0.08)]) 
    t, prediction = torch.max(output.data, 1)
    df_redisplay["win_confidence"] = np.round(x_list * 100 , 2)
    df_redisplay["can_bet"] = df_redisplay["win_confidence"] >= required_confidence * 100
    df_redisplay["bet_home_team"] = prediction
    df_redisplay["actual_outcome"] = _label

    df_2_bet = df_redisplay[df_redisplay["can_bet"]==True] 
    number_of_bets = len(df_2_bet)
    accurate_prediction = len(df_2_bet[df_2_bet["actual_outcome"] == df_2_bet["bet_home_team"]])

    df_2_bet["Won"] = (df_2_bet["actual_outcome"] == df_2_bet["bet_home_team"])* \
                    ((1-df_2_bet["bet_home_team"])*(df_2_bet["Ah_02"]-1) + (df_2_bet["bet_home_team"])*(df_2_bet["Ah_01"]-1)) \
                + (df_2_bet["actual_outcome"] != df_2_bet["bet_home_team"])*-1

    test_data["period"] = f"{last_seven_day.strftime('%Y-%m-%d')}to{ dt.datetime.now().strftime('%Y-%m-%d') }"
    test_data["return_for_the_week_per"] = df_2_bet["Won"].sum()/number_of_bets * 100
    test_data["number_of_bets"] = len(df_2_bet)
    test_data["accuracy"] = accurate_prediction/number_of_bets *100
    df_2_bet = df_2_bet[["Match","Start Time","League","Ah_01_Hcap","Ah_01","Ah_02_Hcap","Ah_02","win_confidence","bet_home_team","actual_outcome","Result","Won"]]
    df_2_bet.to_csv(f"results/result{ test_data['period'] }.csv", index=False)
    return test_data

if __name__ == '__main__':
    test()