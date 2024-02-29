import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

excel_file = r"C:\Users\Госпожа Виктория\Desktop\WINE\Датасет_Вика (1)ВЕРСИЯ1.xlsx"
sheet_name_1 = 'Хим состав'
df = pd.read_excel(excel_file, sheet_name = sheet_name_1)

#VISUALISATION OF YEARS
df[['Year']] = df[['Year']]
df_by_year = df['Year'].value_counts()

dpi = 80
fig, ax = plt.subplots(dpi=dpi, figsize = (1000/dpi, 600/dpi))
plt.rcParams.update({'font.size': 11})
ax.set_title('Распределение по годам сбора урожая, (%)')

wedges, texts, autotexts = ax.pie(
    df_by_year, autopct='', pctdistance = 1.1, radius = 1.1,
    startangle = 90, counterclock = False, wedgeprops=dict(width=0.4))
legend_labels = [f'{year} ({pct:.1f}%)' for year,
                 pct in zip(df_by_year.index, df_by_year/df_by_year.sum()*100)]

plt.legend(legend_labels, bbox_to_anchor=(1.1, 1), loc='upper left')
plt.show()

#VISUALISATION OF GRAPE VARIETIES
default_value = 0
df[['Grape sort']] = df[['Grape sort']].astype(str)
df['Grape sort'] = df['Grape sort'].str.lower()
df_by_sort = df['Grape sort'].fillna(default_value).value_counts()


fig, ax = plt.subplots() #создание графика (гистограмма)

for i, (sort, count) in enumerate(df_by_sort.items()):
    ax.bar(sort, count, label=sort)  
    ax.text(sort, count + 0.1, f"{count}%", rotation=45, ha='center', va='bottom', fontsize=7)
    ax.text(sort, count - 0.2, sort, rotation=45, ha='right', va='top', fontsize=6) 
ax.get_xaxis().set_visible(False)

ax.set_title('Распределение по сортам винограда (%)')
ax.set_ylabel('%')

plt.show()
