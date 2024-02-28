import pandas as pd
import requests
from bs4 import BeautifulSoup


console_output = False
excel_output = True
ExcelFileName = 'altavina_database.xlsx'


class VineProduct:
    def __init__(self, url, info):
        self.url = url
        self.name = info['name']
        self.color = info['color']
        self.country = info['country']
        self.region = info['region']
        self.sugar = info['sugar']
        self.grape = info['grape']
        self.manufacturer = info['manufacturer']
        self.strength = info['strength']
        self.volume = info['volume']
        self.year = info['year']
        self.color_gamut = info['color_gamut']
        self.aroma = info['aroma']
        self.taste = info['taste']
        self.aftertaste = info['aftertaste']
        self.recommendations = info['recommendations']
        self.temperature = info['temperature']

    def get_values(self):
        return [self.name,
                self.color,
                self.country,
                self.region,
                self.sugar,
                self.grape,
                self.manufacturer,
                self.strength,
                self.volume,
                self.year,
                self.color_gamut,
                self.aroma,
                self.taste,
                self.aftertaste,
                self.recommendations,
                self.temperature,
                self.url
                ]


VineProducts = []

base_url = "https://altavina.ru"
catalog_url = base_url + "/catalog/wine/?PAGEN_1="

print('Redirection to site:', base_url, end='\n\n')

for page in range(1, 443):
    print(f'#{page} page', end='\n\n')
    page_url = catalog_url + f"{page}"

    response_page = requests.get(page_url)

    if response_page.status_code != 200:
        print("Ошибка при запросе к странице каталога")
        exit()

    print('Response 200. all ok')

    soup_page = BeautifulSoup(response_page.text, 'html.parser')

    goods_box = soup_page.find('div', class_="catalog-list catalog-list--s_1")
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
        name = soup_wine.find('div', class_='card-top-slider__header').get_text(strip=True)

        # Год
        title_line = soup_wine.find('h1', class_='heading').get_text()
        year = ''
        for part in title_line.split(', '):
            if part.isnumeric():
                year = int(part)
                break

        # Главная информация
        main_info_box = soup_wine.find('div', class_='card-top-slider__info')
        main_info_list = main_info_box.find_all('div', class_='card-top-slider__descr-row')

        # Убираем плашку "напрямую от производителя" если она есть
        if 'Напрямую от производителя!' in str(main_info_list[-1]):
            main_info_list.pop(-1)

        # Бежим по строчкам из списка
        color, country, region, sugar, grape, manufacturer, strength = '', '', '', '', '', '', ''
        for line in main_info_list:

            icon = line.find('div', class_='card-top-slider__descr-icon').find('div')
            if icon is None:
                icon = line.find('div', class_='card-top-slider__descr-icon').find('img')

            if icon is None:
                icon_class = '__flag'
            else:
                try:
                    icon_class = icon.get('class')[0]
                except Exception as ex:
                    print(ex)
                    print(line, end='\n\n')
                    continue

            # Цвет вина
            if icon_class.endswith('__color'):
                color = line.find('a', class_='card-top-slider__descr-link').get_text()

            # Страна и регион
            elif icon_class.endswith('__flag'):
                origin_box = line.find_all('a', class_='card-top-slider__descr-link')
                if len(origin_box) == 2:
                    country, region = [item.get_text() for item in origin_box]

                elif len(origin_box) > 2:
                    country, region = [item.get_text() for item in origin_box][:2]
                    print(f'Block Country/Region contains more than 2 fields: {wine_url}', end='\n\n')

                elif len(origin_box) == 1:
                    region = origin_box[0].get_text()
                    print(f'Block Country/Region contains only 1 field: {wine_url}', end='\n\n')

            # Сахар
            elif icon_class.endswith('__sweet'):
                sugar = line.find('a', class_='card-top-slider__descr-link').get_text()
                sugar = sugar.replace(' ', '').replace('"', '').replace('\n', '')

            # Виноград
            elif icon_class.endswith('__grapes'):

                grape_box = line.find('a', class_='card-top-slider__descr-link')
                grape = grape_box.get_text()

                more_grape_class = 'prod-card__info popover-open card-top-slider__descr-link-more'
                more_grape = main_info_box.find('a', class_=more_grape_class)

                if more_grape is not None:
                    more_grape = more_grape.get('data-content')
                    # Перевод этого: '<a href="/catalog/wine/canaiolo/">Канайоло</a></br>
                    #                <a href="/catalog/wine/cabernet_sauvignon/">Каберне Совиньон</a></br>
                    #                <a href="/catalog/wine/merlot/">Мерло</a></br>'
                    # В это: [Канайоло, Каберне Совиньон, Мерло]
                    grapes_list = [grape[::-1][:grape[::-1].find('>')][::-1] for grape in more_grape.split('</a>')]
                    grapes_list = [grape] + grapes_list
                    if '' in grapes_list:
                        grapes_list.remove('')
                    grape = ', '.join(grapes_list)

            # Производитель
            elif icon_class.endswith('__name'):
                manufacturer = line.find('a', class_='card-top-slider__descr-link').get_text()

            # Крепость
            elif icon_class.endswith('__water'):
                strength = line.find('span', class_='card-top-slider__descr-link not-link').get_text()

        # Объём
        volume = soup_wine.find('div', class_='card-top-slider__vol-link-lg').get_text()

        # Доплнительные характеристики ("Дегустационные заметки")
        notes_box = soup_wine.find('table')
        notes_list = notes_box.find_all('td')

        color_gamut, aroma, taste, aftertaste, recommendations, temperature = '', '', '', '', '', ''

        last_header = None
        for i, line in enumerate(notes_list):

            if i % 2 == 0:
                line_type = 'header'
            else:
                line_type = 'content'

            if line_type == 'header':
                last_header = line.find('b').get_text().replace(':', '')
                continue

            elif line_type == 'content':

                content = notes_list[i].get_text()

                if last_header == 'Глаз':
                    color_gamut = content
                elif last_header == 'Нос':
                    aroma = content
                elif last_header == 'Рот':
                    taste = content
                elif last_header == 'Послевкусие':
                    aftertaste = content
                elif last_header == 'Гастрономические рекомендации':
                    recommendations = content
                elif last_header == 'Температура подачи':
                    temperature = content
                else:
                    continue

        if console_output:
            print(wine_url)
            print('Name:', name)
            print('Color:', color)
            print(f'Country: {country}, Region: {region}')
            print('Sugar:', sugar)
            print('Grape:', grape)
            print('Manufacturer:', manufacturer)
            print('Strength:', strength)
            print('Volume:', volume)
            print('Year:', year)
            print('Color gamut:', color_gamut)
            print('Aroma:', aroma)
            print('Taste:', taste)
            print('Aftertaste:', aftertaste)
            print('Gastronomic recommendations:', recommendations)
            print('Serving temperature:', temperature)
            print('========', end='\n\n')

        wine_info = dict()
        wine_info['name'] = name
        wine_info['color'] = color
        wine_info['country'] = country
        wine_info['region'] = region
        wine_info['sugar'] = sugar
        wine_info['grape'] = grape
        wine_info['manufacturer'] = manufacturer
        wine_info['strength'] = strength
        wine_info['volume'] = volume
        wine_info['year'] = year
        wine_info['color_gamut'] = color_gamut
        wine_info['aroma'] = aroma
        wine_info['taste'] = taste
        wine_info['aftertaste'] = aftertaste
        wine_info['recommendations'] = recommendations
        wine_info['temperature'] = temperature

        VineProducts.append(VineProduct(wine_url, wine_info))



if excel_output:

    print(f'Saving to Excel file has started', end='\n\n')

    columns = ['Название', 'Цвет', 'Страна', 'Регион', 'Сахар', 'Сорта винограда',
               'Производитель', 'Крепость', 'Объем', 'Год', 'Цветовая гамма', 'Аромат',
               'Вкус', 'Послевкусие', 'Гастрономические рекомендации', 'Температура подачи', 'Link']

    df = pd.DataFrame(columns=columns)

    for product in VineProducts:
        df.loc[df.shape[0]] = product.get_values()

    writer = pd.ExcelWriter(ExcelFileName)
    df.to_excel(writer, index=False)
    writer._save()

    print(f'Saving to file {ExcelFileName} completed successfully')
    print('Database volume is', df.shape[0])
    print('========', end='\n\n')






