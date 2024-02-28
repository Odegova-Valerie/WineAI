import pandas as pd
from bs4 import BeautifulSoup
from configparser import ConfigParser
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep
from datetime import datetime
from itertools import groupby


console_output = True
excel_output = True
ExcelFileName = 'TheWineShop_urls.xlsx'


options = webdriver.ChromeOptions()
options.add_argument('--disable-notifications')
# options.add_argument('headless')
driver = webdriver.Chrome(options=options)


config = ConfigParser()
config.read('config.ini', encoding='utf-8')
base_url = config.get('Site', 'base_url')


class Catalog:
    def __init__(self, name, href, volume):
        self.name = name
        self.href = href
        self.url = base_url + href
        self.volume = volume


Catalogs = list()
for catalog in [section for section in config.sections() if section.endswith('Catalog')]:
    name = config.get(catalog, 'name')
    href = config.get(catalog, 'href')
    volume = config.get(catalog, 'volume')
    Catalogs.append(Catalog(name, href, volume))


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def scroll_to_bottom(height=0, found=0, out_of=0):
    print(f'{now()}  height: {height}, found: {found}/{out_of}  Scrolling down...')
    scroll_pause_time = 0.1
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        sleep(scroll_pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    sleep(1)
    return last_height


print('Redirection to site:', base_url, end='\n\n')

WineLinks = []

for catalog in Catalogs:

    print(f'Catalog {catalog.name}')

    page_url = base_url + catalog.href
    driver.get(page_url)
    sleep(1)

    last_height = 0
    stagnation_counter = 0
    goods_hrefs = []
    while True:

        soup_page = BeautifulSoup(driver.page_source, 'html.parser')

        goods_box = soup_page.find('div', class_='row product-grid')
        goods_list = goods_box.find_all('link', href=True)
        goods_hrefs = [product.get('href') for product in goods_list]
        goods_hrefs = [href for href, _ in groupby(goods_hrefs)]

        page_height = scroll_to_bottom(height=last_height, found=len(goods_hrefs), out_of=catalog.volume)
        if last_height == page_height:
            stagnation_counter += 1
            print(now(), f'Stagnation on the same altitude {stagnation_counter}')
            if stagnation_counter == 3:
                print(now(), f'End of scrolling')
                print('========', end='\n\n')
                break
        last_height = page_height

        more_button = driver.find_elements(By.CLASS_NAME, 'btn.btn-show-more.col-12.col-sm-4')
        if more_button:
            more_button = more_button[0]
            try:
                ac = ActionChains(driver)
                x_offset = -0.4 * more_button.size['width']
                ac.move_to_element(more_button).move_by_offset(x_offset, 0).click().perform()
                sleep(5)
            except Exception as ex:
                print(now(), ex)
                body_element = driver.find_element(By.XPATH, '/html/body')
                body_element.send_keys(Keys.ESCAPE)
                sleep(60)
        else:
            print('Button not found')
            print('========', end='\n\n')
            break

    WineLinks += [f'{base_url}{href}' for href in goods_hrefs]

    if excel_output:

            with open(f'{catalog.name}.html', 'w', encoding='utf-8') as file:
                file.write(driver.page_source)

            print()
            print(f'Saving to Excel file has started', end='\n\n')

            df = pd.DataFrame(columns=['Urls'])

            for link in WineLinks:
                df.loc[df.shape[0]] = [link]

            writer = pd.ExcelWriter(ExcelFileName)
            df.to_excel(writer, index=False)
            writer._save()

            print(f'Saving to file {ExcelFileName} completed successfully')
            print('Database volume is', df.shape[0], end='\n\n')

    if console_output:
        print(f'{len(goods_hrefs)} products found', end='\n\n')
        for i, href in enumerate(goods_hrefs):
            print(f'{i+1}. {base_url}{href}')
        print('========', end='\n\n')

