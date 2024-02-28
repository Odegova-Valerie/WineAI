import pandas as pd
import requests
from bs4 import BeautifulSoup
from configparser import ConfigParser
from itertools import groupby


console_output = True
excel_output = True
ExcelFileName = 'TheWineShop_database.xlsx'


class Configuration:
    def __init__(self):
        config = ConfigParser()
        config.read('config.ini', encoding='utf-8')
        self.BaseUrl = config.get('Site', 'base_url')

        self.Parameters = list()
        for section in [section for section in config.sections() if section.endswith('Parameters')]:
            self.Parameters += [config.get(section, param).split(', ') for param in config.options(section)
                                if config.get(section, param).split(', ')[-2] != '...']

        self.MainParameters = [config.get('MainParameters', parameter).split(', ') for parameter
                               in config.options('MainParameters')]
        self.Descriptions = [config.get('DescriptionParameters', parameter).split(', ') for parameter
                              in config.options('DescriptionParameters')]

        self.DatabaseColumns = config.get('DatabaseColumns', 'columns').replace('\n', ' ').split(', ')


class WineProduct:
    def __init__(self, url, info):
        self.url = url
        self.name = info['name']

        for parameter in Config.Parameters:
            setattr(self, parameter[-2], info[parameter[-2]])

    def get_values(self):

        values_list = list()
        for column in Config.DatabaseColumns:
            for parameter in Config.Parameters:
                if parameter[-1] == column:
                    values_list.append(getattr(self, parameter[-2]))
                    break
        return values_list


class Characteristic:
    def __init__(self, title, html_label, value=''):
        self.title = title
        self.html_label = html_label
        self.value = value

    def assign_value(self, value):
        self.value = value


Config = Configuration()

WineLinks = pd.read_excel('results/TheWineShop_urls.xlsx', header=0)['Links'].tolist()
WineLinks = [href for href, _ in groupby(WineLinks)]

WineProducts = list()

base_url = Config.BaseUrl
print('Redirection to site:', base_url, end='\n\n')

for wine_url in WineLinks:

    response_wine = requests.get(wine_url)

    if response_wine.status_code != 200:
        print("Ошибка запроса при переходе к странице товара")
        exit()

    soup_wine = BeautifulSoup(response_wine.text, 'html.parser')

    # Название вина
    name = ''
    try:
        name = soup_wine.find('h2', class_='product-name hide-on-mobile').get_text(strip=True)
    except:
        print(soup_wine.find('h2', class_='product-name hide-on-mobile'))

    if name[:4].isnumeric():
        spare_vintage = int(name[:4])
        name = name[5:]

    else:
        spare_vintage = None

    print(wine_url)
    print(name)
    if not console_output:
        print()

    # Бренд
    Brand = Characteristic('brand', 'Brand')
    try:
        brand = soup_wine.find('h2', class_='product-brand hide-on-mobile').get_text(strip=True)
        Brand.assign_value(brand)
    except:
        pass

    # Цвет
    wine_index = WineLinks.index(wine_url)+1
    if wine_index <= 311:
        color = 'red'
    elif 311 < wine_index <= 311+47:
        color = 'white'
    else:
        color = 'rose'
    Color = Characteristic('color', 'Color', value=color)

    # Основаная информация
    MainInfo = [Characteristic(parameter[1], parameter[0]) for parameter in Config.MainParameters]

    try:
        main_info_box = soup_wine.find('table', class_='table')
        main_info_list = main_info_box.find_all('tr')
    except:
        print('No main info', end='\n\n')
        continue

    for line in main_info_list:
        label = line.find('th', class_='tInfo-label').get_text(strip=True)

        if label in [param.html_label for param in MainInfo]:
            current_parameter = MainInfo[[param.html_label for param in MainInfo].index(label)]
        else:
            current_parameter = None

        if current_parameter is not None:
            try:
                value = line.find('td', class_='tInfo-value').get_text(strip=True)
                current_parameter.assign_value(value)
            except:
                print(line)
        else:
            if label not in ['Varietal', 'Enclosure Type', 'Case Production']:
                print(line)
            continue

    # Запасной год
    Year = MainInfo[[item.title for item in MainInfo].index('year')]
    if not Year.value and spare_vintage is not None:
        print('<SPARE YEAR>')
        Year.assign_value(spare_vintage)

    # Добвление бренда и цвета
    MainInfo.append(Brand)
    MainInfo.append(Color)

    # Описания
    Descriptions = [Characteristic(parameter[1], parameter[0]) for parameter in Config.Descriptions]
    add_info_list = soup_wine.find_all('div', class_='col-12 col-lg-6 tasteDescription-tastingNote')
    add_info_list += soup_wine.find_all('div', class_='col-12 col-lg-6 tasteDescription-wineMakingNote')

    for item in add_info_list:
        label = item.find('h2').get_text(strip=True)

        if label in [param.html_label for param in Descriptions]:
            current_parameter = Descriptions[[param.html_label for param in Descriptions].index(label)]
        else:
            current_parameter = None

        if current_parameter is not None:
            try:
                value = item.get_text(strip=True)[len(label):]
                if value.startswith(label):
                    value = value[len(label):]
                elif label == 'Tasting Notes' and value.startswith('Tasting Note'):
                    value = value[len('Testing Note'):]
                current_parameter.assign_value(value)
            except:
                print(item)
        else:
            if label not in []:
                print(item)
            continue

    # Виноградник и отзывы
    Vineyard = Characteristic('vineyard', 'Vineyard')
    Descriptions.append(Vineyard)
    Reviews = Characteristic('reviews', 'Reviews')
    Descriptions.append(Reviews)

    # Виноградник
    vineyard = soup_wine.find('div', class_='row region')
    if vineyard is not None:
        try:
            vineyard = vineyard.find('p').get_text(strip=True)
        except:
            vineyard = vineyard.get_text(strip=True)
        Vineyard.assign_value(vineyard)

    # Отзывы
    reviews_list = soup_wine.find_all('div', class_='col-award-quote')
    reviews = '\n'.join([review.get_text(strip=True) for review in reviews_list])
    Reviews.assign_value(reviews)

    # Купаж
    Blend = MainInfo[[item.title for item in MainInfo].index('blend')]
    if Blend.value and 'Realer than Real' not in Blend.value:
        blend_is_present = True
    else:
        blend_is_present = False

    if console_output:
        for parameter in MainInfo:
            print(f'{parameter.title}: {parameter.value}')
        for parameter in Descriptions:
            if parameter.value:
                value = '...'
                # value = parameter.value
            else:
                value = ''

            print(f'{parameter.title}: {value}')
        print('========')
        if blend_is_present:
            print('YES BLEND')
        else:
            print('NO BLEND')
        print('========', end='\n\n')

    info = {'url': wine_url, 'name': name}
    for parameter in MainInfo+Descriptions:
        info[parameter.title] = parameter.value

    if blend_is_present:
        WineProducts.append(WineProduct(wine_url, info))


if excel_output:

    print(f'Saving to Excel file has started', end='\n\n')

    df = pd.DataFrame(columns=Config.DatabaseColumns)

    for product in WineProducts:
        df.loc[df.shape[0]] = product.get_values()


    writer = pd.ExcelWriter(ExcelFileName)
    df.to_excel(writer, index=False)
    writer._save()

    print(f'Saving to file {ExcelFileName} completed successfully')
    print('Database volume is', df.shape[0])
    print('========', end='\n\n')




