import requests
from bs4 import BeautifulSoup
import openpyxl
from openpyxl.styles import Font

wb = openpyxl.Workbook()
ws = wb.active

headers = ["Название", "Цена", "Цвет", "Сахар", "Крепость", "Объем", "Производитель", "Страна", "Регион", "Сорта винограда", "Тип", "Температура подачи",
           "Бренд", "Аромат", "Вкус", "Цветовая гамма", "Способ выдержки", "Способ производства", "Дегустационные характеристики",
           "Гастрономическое сопровождение"]
ws.append(headers)

bold_font = Font(bold=True)

for cell in ws[1]:
    cell.font = bold_font

base_url = "https://joia.ru"

catalog_url = base_url + "/catalog/vino/"

response = requests.get(catalog_url)

if response.status_code != 200:
    print("Ошибка при запросе к странице каталога")
    exit()

soup = BeautifulSoup(response.text, 'html.parser')

pagination = soup.find('div', class_='pagination')
if pagination:
    page_links = pagination.find_all('a', class_='pagination__link')
    if page_links:
        last_page = int(page_links[-1].text)
    else:
        last_page = 1
else:
    last_page = 1

for page in range(1, last_page + 1):  # last_page + 1
    page_url = catalog_url + f"?PAGEN_1={page}"

    page_response = requests.get(page_url)

    if page_response.status_code != 200:
        print(f"Ошибка при запросе к странице {page_url}")
        continue

    page_soup = BeautifulSoup(page_response.text, 'html.parser')

    wine_cards = page_soup.find_all('div', class_='card js__product-card')

    for card in wine_cards:
        wine_data = []

        wine_link = card.find('a', class_='card__link')['href']
        wine_url = base_url + wine_link

        wine_response = requests.get(wine_url)

        if wine_response.status_code != 200:
            print(f"Ошибка при запросе к странице вина: {wine_url}")
            continue

        wine_soup = BeautifulSoup(wine_response.text, 'html.parser')

        wine_data.append(
            wine_soup.find('h1', class_='product__h1').text.split(",")[0].strip()
        )
        wine_data.append(
            wine_soup.find('input', class_='custom-control-input')['data-price']
        )

        characteristics = wine_soup.find_all('div', class_='feature')
        char_dict = {}  # Создаем словарь для хранения характеристик

        for char in characteristics:
            char_type = char.find('div', class_='feature__type').text.strip()
            char_text = char.find('div', class_='feature__text').text.strip()
            char_dict[char_type[:-1]] = char_text

        details = wine_soup.find('div', id='specifications')
        if details:
            tables = details.find_all('table', class_='table-features')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    columns = row.find_all('td')
                    if len(columns) == 2:
                        char_type = columns[0].text.strip()
                        char_text = columns[1].text.strip()
                        char_dict[char_type[:-1]] = char_text

        else:
            # Если нет подробных характеристик, добавляем пустые строки
            for _ in range(6):
                wine_data.append('')

        ef_details = wine_soup.find_all('div', class_='ef')

        for dt in ef_details:
            dt_type = dt.find('div', class_='product__title').text.strip()
            dt_text = dt.find('div', class_='product__text').text.strip()
            char_dict[dt_type] = dt_text

        mb_details = wine_soup.find_all('div', class_='mb-md-30')

        for dt in mb_details:
            dt_type = dt.find('div', class_='product__title').text.strip()
            dt_text = dt.find('div', class_='product__text').text.strip()
            char_dict[dt_type] = dt_text

        for header in headers[2:]:
            wine_data.append(char_dict.get(header, ''))

        ws.append(wine_data)

    print(page)

for column_cells in ws.columns:
    max_length = 0
    column = column_cells[0].column_letter
    for cell in column_cells:
        try:
            if len(str(cell.value)) > max_length:
                max_length = len(str(cell.value))
        except:
            pass
    adjusted_width = (max_length + 2)
    ws.column_dimensions[column].width = adjusted_width

wb.save("wine_dataset.xlsx")

print("Парсинг завершен. Данные сохранены в wine_dataset.xlsx")
