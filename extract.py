# download https://sgodds.com/football/data
# download https://sgodds.com/football/data
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import os
import requests
import datetime as dt
import pickle 



def extract():
    URL = "https://sgodds.com/football/data"
    PATH = "Project-Football-Score-Prediction"
    DIR = "assets"

    pick_the_date = dt.datetime.now().strftime("%Y%m%d-")


    response = requests.get(URL)

    chrome_options = Options()
    chrome_options
    chrome_options.add_argument("--headless")
    driver=webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=chrome_options)
    wait = WebDriverWait(driver, 2)
    driver.get(URL)
    get_url = driver.current_url
    wait.until(EC.url_to_be(URL))
    if get_url == URL:
        page_source = driver.page_source
    #header=driver.find_element(By.ID, "toc0")
    topic = driver.find_elements(by=By.XPATH, value="//td[contains(.,'English Premier') \
                                            or contains(.,'Dutch League') \
                                            or contains(.,'English League Champ') \
                                            or contains(.,'French League') \
                                            or contains(.,'German League') \
                                            or contains(.,'Italian League') \
                                            or contains(.,'Spanish League') \
                                            ] \
                                            /parent::*//a")
    #print(header.text)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    href = []
    for t in topic:
        href.append(t.get_attribute("href"))

    driver.close()
    def download(url: str, dest_folder: str):
        date = dt.datetime.now().strftime("%Y%m%d-")
        if not os.path.exists(os.path.join(dest_folder,date)):
            os.makedirs(os.path.join(dest_folder,date))  # create folder if it does not exist


        filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
        filename = f"{date}/{filename}"  
        file_path = os.path.join(dest_folder,filename)

        r = requests.get(url, stream=True)
        if r.ok:
            print("saving to", os.path.abspath(file_path))
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 8):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
        else:  # HTTP status code 4XX/5XX
            print("Download failed: status code {}\n{}".format(r.status_code, r.text))

        return os.path.abspath(file_path)

    filename = []
    for i in href:
        filename.append(download(i, dest_folder="mydir"))
    return

if __name__ == '__main__':
    extract()