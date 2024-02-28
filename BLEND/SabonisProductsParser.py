import pandas as pd
import requests
from bs4 import BeautifulSoup
from configparser import ConfigParser


console_output = True
excel_output = False
ExcelFileName = 'sabonis_database.xlsx'


class Configuration:
    def __init__(self):
        config = ConfigParser()
        config.read('config.ini', encoding='utf-8')
        self.BaseUrl = config.get('Site', 'base_url')
        sections = [section for section in config.sections() if section.endswith('Parameters')]
        self.Parameters = list()
        for section in sections:
            self.Parameters += [config.get(section, param).split(', ') for param in config.options(section)
                                if config.get(section, param).split(', ')[-2] != '...']

        self.MainParameters = [config.get('MainParameters', parameter).split(', ') for parameter
                               in config.options('MainParameters')]
        self.AddParameters = [config.get('AddParameters', parameter).split(', ') for parameter
                              in config.options('AddParameters')]

        self.DatabaseColumns = config.get('DatabaseColumns', 'columns').replace('\n', ' ').split(', ')

        # Color exception
        color_line = config.get('Exception', 'color').split(', ')
        self.ColorException = Characteristic(color_line[1], color_line[0])


class WineProduct:
    def __init__(self, url, info):
        self.url = url
        self.name = info['name']
        self.description = info['description']

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
    def __init__(self, title, html_label):
        self.title = title
        self.html_label = html_label
        self.value = ''

    def assign_value(self, value):
        self.value = value


Config = Configuration()

WineLinks = pd.read_excel('results/sabonis_vino_links.xlsx', header=0)['Links'].tolist()
WineLinks += pd.read_excel('results/sabonis_vino_rossii_links.xlsx', header=0)['Links'].tolist()
BrokenUrls = list()

WineProducts = list()

base_url = Config.BaseUrl
print('Redirection to site:', base_url, end='\n\n')

for wine_url in WineLinks:

    response_wine = requests.get(wine_url)

    if response_wine.status_code != 200:
        print(wine_url)
        print("Ошибка запроса при переходе к странице товара")
        BrokenUrls.append(wine_url)
        continue

    soup_wine = BeautifulSoup(response_wine.text, 'html.parser')

    # Название вина
    name = soup_wine.find('div', class_='product-page-subheader uk-h2').get_text(strip=True)
    print(wine_url)
    print(name)
    if not console_output:
        print()

    # Запасной год
    spare_year = None
    spare_volume = None

    # Основаная информация
    MainInfo = [Characteristic(parameter[1], parameter[0]) for parameter in Config.MainParameters]
    if '/vino-rossii/' in wine_url:
        MainInfo.append(Config.ColorException)

    main_info_box = soup_wine.find('ul', class_='product-page-intro uk-list')
    main_info_list = main_info_box.find_all('li')

    for line in main_info_list:
        label = line.find('strong').get_text().rstrip()

        if label in [param.html_label for param in MainInfo]:
            current_parameter = MainInfo[[param.html_label for param in MainInfo].index(label)]

        else:
            current_parameter = None

        if current_parameter is not None:
            try:
                value = line.get_text()[len(label)+1:].strip()
                if label == 'Объем:':
                    cut_volumes = [value[:index] for index in range(len(value) - 1, 0, -1)]
                    volume_values = [item for item in cut_volumes if item.replace('.', '').isnumeric()]
                    if volume_values:
                        spare_volume = volume_values[0] + ' л'
                    else:
                        spare_volume = value
                else:
                    current_parameter.assign_value(value)
            except:
                print('Label:', label)
                print(line)
        else:
            if label not in ['Объем:']:
                print(line)
            continue

    MainInfo = [item for item in MainInfo if item.title != '...']

    # Дополнительная информация
    AddInfo = [Characteristic(parameter[1], parameter[0]) for parameter in Config.AddParameters]
    if '/vino-rossii/' in wine_url:
        AddInfo.pop([item.title for item in AddInfo].index('color'))

    add_info_box = soup_wine.find('ul', class_='product-page-chars uk-list uk-list-divider')
    add_info_list = add_info_box.find_all('li')

    for line in add_info_list:
        label = line.find('div', class_='uk-width-2-5').get_text(strip=True)

        current_parameter = None
        for param in AddInfo:
            if label in param.html_label:
                current_parameter = param

        if current_parameter is not None:
            try:
                value = line.find('div', class_='uk-width-3-5').get_text(strip=True)
                if label == 'Год:':
                    spare_year = value
                else:
                    current_parameter.assign_value(value)
            except:
                print('Label:', label)
                print(line)
        else:
            if label not in ['Подарочная упаковка:', 'Черная пятница:', 'Год:', 'Линейка:', 'Потенциал Хранения:',
                             'Выдержка:', 'П/К:']:
                print(f'Label: "{label}"')
                print(line)
            continue

    AddInfo = [item for item in AddInfo if item.title != '...']

    # Винтаж и объём

    volume = Characteristic('volume', 'Объем:')
    year = Characteristic('year', 'Год:')
    try:
        vintage_box = soup_wine.find('ul', class_='product-page-variants')
        vintage_list = vintage_box.find_all('li')
        vintage_list = [vintage.find('span', class_='title').get_text(strip=True) for vintage in vintage_list]
        vintage_list = [vintage.replace('\xa0/\xa0', '') for vintage in vintage_list]

        possible_separators = ['л л.', ' л.']
        separator = None
        for sep in possible_separators:
            if sep in vintage_list[0]:
                separator = sep
                break
        if separator is not None:
            vintage_list = [(vintage.split(separator)) for vintage in vintage_list]
            vintage_list = [(item[0] + ' л', item[1]) for item in vintage_list]
            if console_output:
                print(vintage_list)

        years = [int(item[1]) for item in vintage_list if item[1].isnumeric()]
        volumes = [float(item[0][:-2]) for item in vintage_list]
        if not [v for v in volumes if v >= 0.75]:
            volume.assign_value(f'{max(volumes)} л')
        else:
            volume.assign_value(f'{min([v for v in volumes if v >= 0.75])} л')
        if years:
            year.assign_value(max(years))
    except:
        if console_output:
            print([])
        if spare_year is not None:
            year.assign_value(spare_year)
        if spare_volume is not None:
            volume.assign_value(spare_volume)

    MainInfo += [volume, year]

    # Описание
    Description = [Characteristic('description', 'Описание:')]
    try:
        description = soup_wine.find('span', itemprop='description').get_text(strip=True)
        Description[0].assign_value(description)
    except:
        pass

    if console_output:
        for parameter in MainInfo+AddInfo:
            if '&' in parameter.html_label:
                print(parameter.html_label.split(' & ')[-1], parameter.value)
            else:
                print(parameter.html_label, parameter.value)
        for parameter in Description:
            if parameter.value:
                value = '...'
            else:
                value = ''
            print(parameter.html_label, value)
        print('========', end='\n\n')
    else:
        print()

    info = {'url': wine_url, 'name': name}
    for parameter in MainInfo+AddInfo+Description:
        info[parameter.title] = parameter.value

    WineProducts.append(WineProduct(wine_url, info))


if BrokenUrls:
    print(f'{len(BrokenUrls)} products were not parsed')
    for url in BrokenUrls:
        print(url)
    print('========', end='\n\n')
else:
    print('All products were parsed. No errors')
    print('========', end='\n\n')


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

