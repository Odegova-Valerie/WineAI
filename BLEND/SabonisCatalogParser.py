import pandas as pd
from bs4 import BeautifulSoup
from configparser import ConfigParser
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep
from datetime import datetime


console_output = True
excel_output = True
ExcelFileName = 'sabonis_links.xlsx'


options = webdriver.ChromeOptions()
options.add_argument('--disable-notifications')
options.add_argument('headless')
driver = webdriver.Chrome(options=options)


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


config = ConfigParser()
config.read('config.ini', encoding='utf-8')

base_url = config.get('Site', 'base_url')
Catalogs = config.get('Site', 'catalog_hrefs').split(', ')

print('Redirection to site:', base_url, end='\n\n')

countries = ['Италия', 'Франция', 'Испания', 'Португалия', 'Германия', 'Австрия', 'США', 'Австралия', 'Новая Зеландия',
             'Аргентина', 'Чили', 'ЮАР', 'Грузия', 'Армения', 'Греция', 'Венгрия', 'Люксембург', 'Словения',
             'Бразилия', 'Сербия', 'Израиль', 'Латвия', 'Китай', 'Уругвай', 'Япония', 'Исландия']

codes = [232, 233, 234, 235, 236, 237, 238, 239, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319,
         2320, 2320, 2321, 2321, 2324, 2324, 2325, 2327]

WineLinks = []

for i, country in enumerate(countries):

    print(country)
    catalog_href = Catalogs[0]
    country_href = f'?filter%5Bstrana_proishozhdeniya%5D%5B%5D=%{codes[i]}%23{country}'

    page_url = base_url + catalog_href + country_href

    driver.get(page_url)
    sleep(1)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    number_of_wines = int(soup.find('strong', class_='uk-text-primary').get_text(strip=True))

    goods_links = []
    last_height = 0
    stagnation_counter = 0
    while True:

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        goods_box = soup.find(id='catalog')
        goods_list = goods_box.find_all('a', href=True)
        goods_links = [product.get('href') for product in goods_list]

        WineLinks += [f'{base_url}{link}' for link in goods_links if f'{base_url}{link}' not in WineLinks]

        page_height = scroll_to_bottom(height=last_height, found=len(goods_links), out_of=number_of_wines)
        if last_height == page_height:
            stagnation_counter += 1
            print(now(), f'Stagnation on the same altitude {stagnation_counter}')
            if stagnation_counter == 3:
                print(now(), f'End of scrolling')
                print('========', end='\n\n')
                break
        last_height = page_height

        more_button = driver.find_elements(By.ID, 'more')
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
            print('========', end='\n\n')
            break

        with open('html.html', 'w', encoding='utf-8') as file:
            file.write(driver.page_source)

    if excel_output:

        print(f'Saving to Excel file has started', end='\n\n')

        df = pd.DataFrame(columns=['Links'])

        for link in WineLinks:
            df.loc[df.shape[0]] = [link]

        writer = pd.ExcelWriter(ExcelFileName)
        df.to_excel(writer, index=False)
        writer._save()

        print(f'Saving to file {ExcelFileName} completed successfully')
        print('Database volume is', df.shape[0])
        print('========', end='\n\n')

    if console_output:
        print(f'{len(WineLinks)} products found', end='\n\n')
        for i, link in enumerate(WineLinks):
            print(f'{i+1}. {link}')
        print('========', end='\n\n')


