from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
options=Options()
options.add_experimental_option("detach",True)

driver =webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                         options=options)

url='https://www.bettingpros.com/nfl/picks/prop-bets/minnesota-vikings-vs-philadelphia-eagles/'

driver.get(url)
driver.maximize_window()
links = driver.find_elements("xpath", "//a[]")
for link in links:
    print(link)