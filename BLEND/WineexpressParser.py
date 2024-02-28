import pandas as pd
import requests
from bs4 import BeautifulSoup

console_output = False
excel_output = True
ExcelFileName = 'wineexpress_database.xlsx'


class VineProduct:
    def __init__(self, url, info):
        self.url = url
        self.name = info['name']
        self.type = info['wine_type']
        self.color = info['color']
        self.sugar = info['sugar']
        self.grape = info['grape']
        self.strength = info['strength']
        self.country = info['country']
        self.region = info['region']
        self.subregion = info['subregion']
        self.appellation = info['appellation']
        self.manufacturer = info['manufacturer']
        self.brand = info['brand']
        self.volume = info['volume']
        self.harvest_year = info['harvest_year']
        self.color_gamut = info['color_gamut']
        self.aroma = info['aroma']
        self.taste = info['taste']
        self.recommendations = info['recommendations']
        self.description = info['description']

    def get_values(self):
        return [self.name,
                self.type,
                self.color,
                self.sugar,
                self.grape,
                self.strength,
                self.country,
                self.region,
                self.subregion,
                self.appellation,
                self.manufacturer,
                self.brand,
                self.volume,
                self.harvest_year,
                self.color_gamut,
                self.aroma,
                self.taste,
                self.recommendations,
                self.description,
                self.url
                ]


class Characteristic:
    def __init__(self, title, html_label):
        self.title = title
        self.html_label = html_label
        self.value = ''

    def assign_value(self, value):
        self.value = value


class Catalog:
    def __init__(self, name, href, pages):
        self.name = name
        self.href = href
        self.pages = pages
        self.url = base_url + href


VineProducts = []

base_url = "https://wineexpress.ru"
Catalogs = [Catalog('Обычные вина', '/catalog/wine/?PAGEN_1=', 35),
            Catalog('Игристые вина', '/catalog/igristye-vina/?PAGEN_1=', 9)
            ]

print('Redirection to site:', base_url, end='\n\n')

for catalog in Catalogs:
    for page in range(1, catalog.pages+1):
        print(f'#{page} page', end='\n\n')
        page_url = catalog.url + f"{page}"

        response_page = requests.get(page_url)

        if response_page.status_code != 200:
            print("Ошибка при запросе к странице каталога")
            exit()

        print('Response 200. all ok')

        soup_page = BeautifulSoup(response_page.text, 'html.parser')

        goods_box = soup_page.find('div', class_='row row-m')
        goods_list = goods_box.find_all('a', href=True)
        raw_goods_links = [product.get('href') for product in goods_list]

        goods_links = []
        for link in raw_goods_links:
            if not link.startswith('/product'):
                continue
            if link not in goods_links:
                goods_links.append(link)

        print(f'{len(goods_links)} products found', end='\n\n')

        if console_output:
            for i, link in enumerate(goods_links):
                print(f'{i+1}. {base_url}{link}')
            print('========', end='\n\n')

        for href in goods_links:

            wine_url = base_url + href
            response_wine = requests.get(wine_url)

            if response_wine.status_code != 200:
                print("Ошибка запроса при переходе к странице товара")
                exit()

            soup_wine = BeautifulSoup(response_wine.text, 'html.parser')

            # Название вина
            name = soup_wine.find('h1', class_='card-title').get_text(strip=True)
            print(name)

            # Характеристики
            Characteristics = [Characteristic('wine_type', 'Тип:'),  # wine_type
                               Characteristic('color', 'Цвет:'),  # color
                               Characteristic('sugar', 'Сахар:'),  # sugar
                               Characteristic('grape', 'Сорт:'),  # grape
                               Characteristic('strength', 'Крепость:'),  # strength
                               Characteristic('country', 'Страна:'),  # country
                               Characteristic('region', 'Регион:'),  # region
                               Characteristic('subregion', 'Субрегион:'),  # subregion
                               Characteristic('appellation', 'Аппелласьон:'),  # appellation
                               Characteristic('manufacturer', 'Производитель:'),  # manufacturer
                               Characteristic('brand', 'Бренд:'),  # brand
                               Characteristic('volume', 'Объём:'),  # volume
                               Characteristic('harvest_year', 'Год урожая:')  # harvest_year
                               ]

            characteristics_box = soup_wine.find('ul', class_='card-spec list-unstyled')
            characteristics_list = characteristics_box.find_all('li', class_='card__spec-item')

            for line in characteristics_list:
                label = line.find('span', class_='card__spec-label').get_text()

                current_parameter = None
                for parameter in Characteristics:
                    if parameter.html_label == label:
                        current_parameter = parameter
                        break

                if current_parameter is not None:
                    try:
                        value = ', '.join([item.get_text(strip=True) for item in line.find_all('a', href=True)])
                        current_parameter.assign_value(value)
                    except:
                        print(line)
                else:
                    print(line)
                    continue

            # Описание
            Description = [Characteristic('description', 'Описание')]
            try:
                description = soup_wine.find('div', class_='col-lg-8').get_text()
                description = description.split('\n')[-1].lstrip()
                Description[0].assign_value(description)
            except:
                description = ''

            # Гастрономические преколы
            Gastronomic = [Characteristic('color_gamut', 'Цвет'),  # color_gamut
                           Characteristic('aroma', 'Аромат'),  # aroma
                           Characteristic('taste', 'Вкус'),  # taste
                           Characteristic('recommendations', 'Гастрономические сочетания'),  # recommendations
                           ]

            try:
                gastronomic_box = soup_wine.find('ul', class_='card-gastronomic list-unstyled')
                gastronomic_list = gastronomic_box.find_all('li', class_='card-gastronomic__item')
            except:
                gastronomic_list = list()

            for line in gastronomic_list:
                label = line.find('div', class_='card-gastronomic__title').get_text()

                current_parameter = None
                for parameter in Gastronomic:
                    if parameter.html_label == label:

                        current_parameter = parameter
                        break

                if current_parameter is not None:
                    value = line.get_text().split('\n')[-1].strip()
                    if (value.startswith('Аромат с нотами персика, цитрусовых фруктов') and
                        value.endswith('легкими травяными нюансами.')):
                        value = 'Аромат с нотами персика, цитрусовых фруктов и легкими травяными нюансами.'
                    current_parameter.assign_value(value)
                else:
                    print(line)
                    continue

            # Убираем лишнее из названия
            wine_type = Characteristics[0].value
            name = name[len(wine_type)+1:]

            if console_output:
                for parameter in Characteristics:
                    print(parameter.html_label, parameter.value)
                for parameter in Gastronomic+Description:
                    if parameter.value:
                        value = '...'
                        # value = parameter.value
                    else:
                        value = ''
                    print(f'{parameter.html_label}: {value}')
                print('========', end='\n\n')

            info = {'url': wine_url, 'name': name}
            for parameter in Characteristics+Gastronomic+Description:
                info[parameter.title] = parameter.value

            VineProducts.append(VineProduct(wine_url, info))

        if excel_output:

            print(f'Saving to Excel file has started', end='\n\n')

            columns = ['Название', 'Тип', 'Цвет', 'Сахар', 'Сорт винограда', 'Крепость',
                       'Страна', 'Регион', 'Субрегион', 'Аппелласьон', 'Производитель', 'Бренд', 'Объем', 'Год урожая',
                       'Цветовая гамма', 'Аромат', 'Вкус', 'Гастрономические сочетания', 'Описание', 'Link']

            df = pd.DataFrame(columns=columns)

            for product in VineProducts:
                df.loc[df.shape[0]] = product.get_values()

            writer = pd.ExcelWriter(ExcelFileName)
            df.to_excel(writer, index=False)
            writer._save()

            print(f'Saving to file {ExcelFileName} completed successfully')
            print('Database volume is', df.shape[0])
            print('========', end='\n\n')


# if excel_output:
#
#     print(f'Saving to Excel file has started', end='\n\n')
#
#     columns = ['Название', 'Тип', 'Цвет', 'Сахар', 'Сорт винограда', 'Крепость',
#                'Страна', 'Регион', 'Субрегион', 'Аппелласьон', 'Производитель', 'Бренд', 'Объем', 'Год урожая',
#                'Цветовая гамма', 'Аромат', 'Вкус', 'Гастрономические сочетания', 'Описание', 'Link']
#
#     df = pd.DataFrame(columns=columns)
#
#     for product in VineProducts:
#         df.loc[df.shape[0]] = product.get_values()
#
#     writer = pd.ExcelWriter(ExcelFileName)
#     df.to_excel(writer, index=False)
#     writer._save()
#
#     print(f'Saving to file {ExcelFileName} completed successfully')
#     print('Database volume is', df.shape[0])
#     print('========', end='\n\n')

