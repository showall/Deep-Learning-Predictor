from selenium.webdriver.support.ui import Select
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import glob
import pickle 
from bs4 import BeautifulSoup
from math import exp
import torch
import numpy as np
import pandas as pd
import datetime as dt
import os


def recommend():
    URL_LIVE = "https://online.singaporepools.com/en/sports/category/1/football"
    PATH_DIR = "."

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver=webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=chrome_options)
    wait = WebDriverWait(driver, 10)
    driver.get(URL_LIVE)
    get_url = driver.current_url
    wait.until(EC.presence_of_element_located((By.XPATH,'//select[@class="form-control event-list__filter__date"]/option[@value="ANYTM"]')))
    if get_url == URL_LIVE:
        page_source = driver.page_source
    #header=driver.find_element(By.ID, "toc0")

    #print(header.text)
    #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    select = Select(driver.find_element(by=By.XPATH, value='//select[@class="form-control event-list__filter__date"]'))
    select.select_by_visible_text('Anytime')

    driver.find_element(by=By.XPATH, value='//button[@class="btn-block button button--orange btn btn-default"]').click()

    button = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH,'//div[@class="event-list__load-all-events"]')))
    button.click()

    links = []
    for i in driver.find_elements(By.XPATH, "//span[@class='event-list__event-name']/a"):
        links.append(i.get_attribute("href"))
    links = list(set(links))

    df_new = pd.DataFrame({})

    for link in links:
        wait =  WebDriverWait(driver, 5) 
        driver.get(link)
        try:
            wait.until(EC.presence_of_element_located((By.XPATH,'//div[@class="event-markets"]')))
        except:
            next
        soup = BeautifulSoup(driver.page_source, "html.parser")
        x = soup.find_all("button", class_ ="button button-outcome text-nowrap button-outcome--full")
        y = []
        y1 = []

        
        need_to_be_available = ["Asian Handicap",
                                "1X2",
                                "1/2 Goal",
                                "Pick The Score",
                                "Total Goals",
                                "Total Goals Odd/Even",
                                "Team to Score 1st Goal",
                                "Halftime PTS",
                                "Halftime 1X2",
                                "Halftime Total Goals Odd/Even",
                                "Half Time Team To Score 1st Goal",
                                "Half Time Will Both Teams Score",
                                "Halftime Total Goals Odd/Even",
                                "Halftime Total Goals",
                                "Will Both Teams Score"
                            ]
        check = True
        for heading in need_to_be_available:
            if soup.find_all("span", text = heading) == []:
                check = False
        if check==True:
            try:
                for i in x:
                    y.append(i.find_all("span", class_ ="button-outcome__price"))
                    y1.append(i.find_all("span", class_ ="button-outcome__text"))    

                y = [item for sublist in y for item in sublist]
                y1 = [item for sublist in y1 for item in sublist]
                z = [i.text for i in y]
                z1 = [i.text.strip().encode("ascii", "ignore").decode("ascii").split() for i in y1]    
            #     try:
                #new_row = df.loc[0:0].copy()
                new_row = dict()
                new_row["ID"] = soup.find("span", id ="sports-event-retail-id" ).text
                new_row["Match"] = soup.find("span", id ="sports-event-name" ).text
                new_row["League"] =  soup.find_all("span", class_ ="sppl-breadcrumb__trail")[-2].text.strip().encode("ascii", "ignore").decode("ascii")
                new_row["Result"] = "-"
                new_row["Ah_01_Hcap"] = z1[0][-1]
                new_row["Ah_01"] =  z[0]
                new_row["Ah_02_Hcap"] =z1[1][-1]
                new_row["Ah_02"] = z[1]
                new_row["Start Time"] = soup.find("span", id ="sports-event-start-time" ).text

                new_row["Ft1X2_01"] = z[2]
                new_row["Ft1X2_02"] = z[3]
                new_row["Ft1X2_03"] = z[4]

                new_row["Htft_01"] = z[9+2]
                new_row["Htft_02"] = z[10+2]
                new_row["Htft_03"] = z[11+2]
                new_row["Htft_04"] = z[12+2]
                new_row["Htft_05"] = z[13+2]
                new_row["Htft_06"] = z[14+2]
                new_row["Htft_07"] = z[15+2]
                new_row["Htft_08"] = z[16+2]
                new_row["Htft_09"] = z[17+2]

                new_row["Pts_10"] = z[18+2]
                new_row["Pts_20"] = z[19+2]
                new_row["Pts_21"] = z[20+2]
                new_row["Pts_30"] = z[21+2]
                new_row["Pts_31"] = z[22+2]
                new_row["Pts_32"] = z[23+2]
                new_row["Pts_40"] = z[24+2]
                new_row["Pts_41"] = z[25+2]
                new_row["Pts_42"] = z[26+2]
                new_row["Pts_43"] = z[27+2]
                new_row["Pts_50"] = z[28+2]
                new_row["Pts_51"] = z[29+2]
                new_row["Pts_52"] = z[30+2]
                new_row["Pts_53"] = z[31+2]
                new_row["Pts_54"] = z[32+2]
                new_row["Pts_00"] = z[33+2]
                new_row["Pts_11"] = z[34+2]
                new_row["Pts_22"] = z[35+2]
                new_row["Pts_33"] = z[36+2]
                new_row["Pts_44"] = z[37+2]
                new_row["Pts_99"] = z[38+2]
                new_row["Pts_01"] = z[39+2]
                new_row["Pts_02"] = z[40+2]
                new_row["Pts_12"] = z[41+2]
                new_row["Pts_03"] = z[42+2]
                new_row["Pts_13"] = z[43+2]
                new_row["Pts_23"] = z[44+2]
                new_row["Pts_04"] = z[45+2]
                new_row["Pts_14"] = z[46+2]
                new_row["Pts_24"] = z[47+2]
                new_row["Pts_34"] = z[48+2]
                new_row["Pts_05"] = z[49+2]
                new_row["Pts_15"] = z[50+2]
                new_row["Pts_25"] = z[51+2]
                new_row["Pts_35"] = z[52+2]
                new_row["Pts_45"] = z[53+2]


                new_row["Tg_00"] = z[56]
                new_row["Tg_01"] = z[57]
                new_row["Tg_02"] = z[58]
                new_row["Tg_03"] = z[59]
                new_row["Tg_04"] = z[60]
                new_row["Tg_05"] = z[61]
                new_row["Tg_06"] = z[62]
                new_row["Tg_07"] = z[63]
                new_row["Tg_08"] = z[64]
                new_row["Tg_09"] = z[65]

                new_row["T1g_01"] = z[66]
                new_row["T1g_02"] = z[67]
                new_row["T1g_03"] = z[68]

                new_row["Oe_01"] = z[69]
                new_row["Oe_02"] = z[70]

                new_row["Ht1X2_01"] = z[71]
                new_row["Ht1X2_02"] = z[72]
                new_row["Ht1X2_03"] = z[73]

                new_row["Htpts_10"] = z[74]
                new_row["Htpts_20"] = z[75]
                new_row["Htpts_21"] = z[76]
                new_row["Htpts_00"] = z[77]

                new_row["Htpts_11"] = z[78]
                new_row["Htpts_22"] = z[79]
                new_row["Htpts_99"] = z[80]
                new_row["Htpts_01"] = z[81]
                new_row["Htpts_02"] = z[82]
                new_row["Htpts_12"] = z[83]

                new_row["Ou_hcap"] = z1[7][-1]
                new_row["Ou_01"] = z[7]
                new_row["Ou_02"] = z[8]

                new_row["Htou_Hcap"] = z1[84][-1]
                new_row["Htou_01"] = z[84]
                new_row["Htou_02"] =z[85]


                new_row["Hg_01_Hcap"] = z1[5][-1]
                new_row["Hg_01"] = z[5]
                new_row["Hg_02_Hcap"] = z1[6][-1]
                new_row["Hg_02"] = z[6]

                new_row["Bg_01"] = z[9]
                new_row["Bg_02"] = z[10]

                new_row["Htoe_01"] =z[86]
                new_row["Htoe_02"] = z[87]

                new_row["Httg_00"] = z[88]
                new_row["Httg_01"] = z[89]
                new_row["Httg_02"] = z[90]
                new_row["Httg_03"] = z[91]


                new_row["Htt1g_01"] = z[92]
                new_row["Htt1g_02"] = z[93]
                new_row["Htt1g_03"] = z[94]

                new_row["Htbg_01"] = z[95]
                new_row["Htbg_02"] = z[96]
                #     except:
            #         continue
                WebDriverWait(driver, 5) 
                new_row=pd.DataFrame(new_row,index=[0])
                df_new = pd.concat([df_new, new_row.copy()], axis=0)
            except:
                WebDriverWait(driver, 5) 
                next
    driver.close()
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
    df_new = df_new.drop_duplicates()
    df_new = df_new.dropna()
    if len(df_new) != 0:
        X_final  = df_new[list(what_we_need)]
        for col in X_final.columns:
            X_final[col]= X_final[col].astype(float)

     #   current_model_title = dt.datetime.now().strftime("%Y-%m-%d")
        list_of_files = glob.glob(f'{PATH_DIR}/models/*.pt') 
        latest_model = max(list_of_files, key=os.path.getctime).split("\\")[-1]
        current_model = torch.jit.load(f'{PATH_DIR}/models/{ latest_model }')
        current_model.eval()

        list_of_files = glob.glob(f'{PATH_DIR}/models/*.pkl') 
        latest_sc = max(list_of_files, key=os.path.getctime).split("\\")[-1]


        with open(f'{PATH_DIR}/models/{latest_sc}','rb') as f: 
            current_model_sc = pickle.load(f)


        X_final = current_model_sc.transform(X_final)

        required_confidence = 0.7

        final_input = torch.tensor(X_final, dtype=torch.float32)
        final_input.shape

        output = current_model(final_input)

        x_list = [[exp(x[0])/(exp(x[0])+exp(x[1])),exp(x[1])/(exp(x[0])+exp(x[1]))] for x in output]
        x_list = np.max(np.array(x_list), axis=1) 
        
        t, prediction = torch.max(output.data, 1)
        required_confidence =float (np.sort(x_list)[::-1][int(len(x_list)*0.08)-1:int(len(x_list)*0.08)]) 
        df_new["win_confidence"] = np.round(x_list * 100 , 2)
        
        if required_confidence > np.max(x_list):
            required_confidence = np.max(x_list) - 0.05
        df_new["can_bet"] = df_new["win_confidence"] >= required_confidence * 100
        df_new["bet_home_team"] = prediction
        df_new = df_new[df_new["can_bet"]== True]


        data_recommend = dict(df_new[["Match","Start Time","League","Ah_01_Hcap","Ah_01","Ah_02_Hcap","Ah_02","win_confidence","bet_home_team"]])
    else :
        data_recommend = {"error":"No recommendation found at this moment. Please come back later"}
    pd.DataFrame(data_recommend).reset_index().to_csv(f'{PATH_DIR}/assets/recommendation.csv')

    return data_recommend


if __name__ == '__main__':
    recommend()