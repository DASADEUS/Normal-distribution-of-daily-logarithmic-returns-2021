# -*- coding: utf-8 -*-
"""2. Предварительный анализ данных
"""

import numpy as np
import pandas as pd
from google.colab import files
import datetime
import matplotlib.pyplot as plt
import scipy.stats as sp

uploaded = files.upload()

WBA=pd.read_csv('WBA.csv', delimiter=',')
WBA=WBA.stack().str.replace(',','.').unstack()
UNH=pd.read_csv('UNH.csv', delimiter=',')
UNH=UNH.stack().str.replace(',','.').unstack()
PFE=pd.read_csv('PFE.csv', delimiter=',')
PFE=PFE.stack().str.replace(',','.').unstack()
MRK=pd.read_csv('MRK.csv', delimiter=',')
MRK=MRK.stack().str.replace(',','.').unstack()
MMM=pd.read_csv('MMM.csv', delimiter=',')
MMM=MMM.stack().str.replace(',','.').unstack()
LLY=pd.read_csv('LLY.csv', delimiter=',')
LLY=LLY.stack().str.replace(',','.').unstack()
JNJ=pd.read_csv('JNJ.csv', delimiter=',')
JNJ=JNJ.stack().str.replace(',','.').unstack()
DHR=pd.read_csv('DHR.csv', delimiter=',')
DHR=DHR.stack().str.replace(',','.').unstack()
BMY=pd.read_csv('BMY.csv', delimiter=',')
BMY=BMY.stack().str.replace(',','.').unstack()
ABT=pd.read_csv('ABT.csv', delimiter=',')
ABT=ABT.stack().str.replace(',','.').unstack()

for i in range(len(JNJ['Дата'])):
  if JNJ['Объём'][i][len(JNJ['Объём'][i])- 1]=='K':
    JNJ['Объём'][i]=str(float(JNJ['Объём'][i][0:len(JNJ['Объём'][i])- 1])/1000)[0:4]+'М'
  if JNJ['Объём'][i]=='-':
    JNJ['Объём'][i]='0.00М'
for i in range(len(ABT['Дата'])):
  if ABT['Объём'][i][len(ABT['Объём'][i])- 1]=='K':
    ABT['Объём'][i]=str(float(ABT['Объём'][i][0:len(ABT['Объём'][i])- 1])/1000)[0:4]+'М'
  if ABT['Объём'][i]=='-':
    ABT['Объём'][i]='0.00М'
for i in range(len(LLY['Дата'])):
  if LLY['Объём'][i][len(LLY['Объём'][i])- 1]=='K':
    LLY['Объём'][i]=str(float(LLY['Объём'][i][0:len(LLY['Объём'][i])- 1])/1000)[0:4]+'М'
  if LLY['Объём'][i]=='-':
    LLY['Объём'][i]='0.00М'
for i in range(len(DHR['Дата'])):
  if DHR['Объём'][i][len(DHR['Объём'][i])- 1]=='K':
    DHR['Объём'][i]=str(float(DHR['Объём'][i][0:len(DHR['Объём'][i])- 1])/1000)[0:4]+'М'
  if DHR['Объём'][i]=='-':
    DHR['Объём'][i]='0.00М'

def numdays(Tiker):
  Tikern7,Tikern8,Tikern9,Tikern20=0,0,0,0
  for i in Tiker['Дата']:
    if int(i[6:10])==2017: Tikern7+=1
    if int(i[6:10])==2018: Tikern8+=1
    if int(i[6:10])==2019: Tikern9+=1
    if int(i[6:10])==2020: Tikern20+=1
  return(Tikern7,Tikern8,Tikern9,Tikern20)

PFEd=numdays(PFE)
BMYd=numdays(BMY)
MRKd=numdays(MRK)
JNJd=numdays(JNJ)
WBAd=numdays(WBA)
ABTd=numdays(ABT)
LLYd=numdays(LLY)
DHRd=numdays(DHR)
MMMd=numdays(MMM)
UNHd=numdays(UNH)

"""Таблица 2.Число торговых дней"""

d = {"2017": pd.Series([PFEd[0],BMYd[0],MRKd[0],JNJd[0],WBAd[0],ABTd[0],LLYd[0],DHRd[0],MMMd[0],UNHd[0]], 
                       index=['PFE', 'BMY', 'MRK','JNJ','WBA','ABT','LLY','DHR',' MMM','UNH']),
     "2018": pd.Series([PFEd[1],BMYd[1],MRKd[1],JNJd[1],WBAd[1],ABTd[1],LLYd[1],DHRd[1],MMMd[1],UNHd[1]], 
                       index=['PFE', 'BMY', 'MRK','JNJ','WBA','ABT','LLY','DHR',' MMM','UNH']),
     "2019": pd.Series([PFEd[2],BMYd[2],MRKd[2],JNJd[2],WBAd[2],ABTd[2],LLYd[2],DHRd[2],MMMd[2],UNHd[2]], 
                       index=['PFE', 'BMY', 'MRK','JNJ','WBA','ABT','LLY','DHR',' MMM','UNH']),
     "2020": pd.Series([PFEd[3],BMYd[3],MRKd[3],JNJd[3],WBAd[3],ABTd[3],LLYd[3],DHRd[3],MMMd[3],UNHd[3]], 
                       index=['PFE', 'BMY', 'MRK','JNJ','WBA','ABT','LLY','DHR',' MMM','UNH'])}
pd.DataFrame(d)

def deviations(Tiker):
  dev2017,dev2018,dev2019,dev2020=[],[],[],[]
  for i in range(len(Tiker['Откр.'])):
    if int(Tiker['Дата'][i][6:10])==2017: dev2017.append(float(Tiker['Цена'][i])/float(Tiker['Откр.'][i])-1)
    if int(Tiker['Дата'][i][6:10])==2018: dev2018.append(float(Tiker['Цена'][i])/float(Tiker['Откр.'][i])-1)
    if int(Tiker['Дата'][i][6:10])==2019: dev2019.append(float(Tiker['Цена'][i])/float(Tiker['Откр.'][i])-1)
    if int(Tiker['Дата'][i][6:10])==2020: dev2020.append(float(Tiker['Цена'][i])/float(Tiker['Откр.'][i])-1)
  return([max(dev2017),min(dev2017)],[max(dev2018),min(dev2018)],[max(dev2019),min(dev2019)],[max(dev2020),min(dev2020)])

"""Таблица 3. Максимальные скачки вверх"""

d = {"2017": pd.Series([deviations(PFE)[0][0],deviations(BMY)[0][0],deviations(MRK)[0][0],deviations(JNJ)[0][0],deviations(WBA)[0][0],deviations(ABT)[0][0],deviations(LLY)[0][0],deviations(DHR)[0][0],deviations(MMM)[0][0],deviations(UNH)[0][0]],
                       index=['PFE', 'BMY', 'MRK','JNJ','WBA','ABT','LLY','DHR',' MMM','UNH']),
     "2018": pd.Series([deviations(PFE)[1][0],deviations(BMY)[1][0],deviations(MRK)[1][0],deviations(JNJ)[1][0],deviations(WBA)[1][0],deviations(ABT)[1][0],deviations(LLY)[1][0],deviations(DHR)[1][0],deviations(MMM)[1][0],deviations(UNH)[1][0]],
                       index=['PFE', 'BMY', 'MRK','JNJ','WBA','ABT','LLY','DHR',' MMM','UNH']),
     "2019": pd.Series([deviations(PFE)[2][0],deviations(BMY)[2][0],deviations(MRK)[2][0],deviations(JNJ)[2][0],deviations(WBA)[2][0],deviations(ABT)[2][0],deviations(LLY)[2][0],deviations(DHR)[2][0],deviations(MMM)[2][0],deviations(UNH)[2][0]],
                       index=['PFE', 'BMY', 'MRK','JNJ','WBA','ABT','LLY','DHR',' MMM','UNH']),
     "2020": pd.Series([deviations(PFE)[3][0],deviations(BMY)[3][0],deviations(MRK)[3][0],deviations(JNJ)[3][0],deviations(WBA)[3][0],deviations(ABT)[3][0],deviations(LLY)[3][0],deviations(DHR)[3][0],deviations(MMM)[3][0],deviations(UNH)[3][0]],
                       index=['PFE', 'BMY', 'MRK','JNJ','WBA','ABT','LLY','DHR',' MMM','UNH']),
     "Max": pd.Series([max(deviations(PFE))[0],max(deviations(BMY))[0],max(deviations(MRK))[0],max(deviations(JNJ))[0],max(deviations(WBA))[0],max(deviations(ABT))[0],max(deviations(LLY))[0],max(deviations(DHR))[0],max(deviations(MMM))[0],max(deviations(UNH))[0]],
                      index=['PFE', 'BMY', 'MRK','JNJ','WBA','ABT','LLY','DHR',' MMM','UNH'])}
pd.DataFrame(d)

"""Таблица 4. Максимальные скачки вниз"""

d = {"2017": pd.Series([deviations(PFE)[0][1],deviations(BMY)[0][1],deviations(MRK)[0][1],deviations(JNJ)[0][1],deviations(WBA)[0][1],deviations(ABT)[0][1],deviations(LLY)[0][1],deviations(DHR)[0][1],deviations(MMM)[0][1],deviations(UNH)[0][1]],
                       index=['PFE', 'BMY', 'MRK','JNJ','WBA','ABT','LLY','DHR',' MMM','UNH']),
     "2018": pd.Series([deviations(PFE)[1][1],deviations(BMY)[1][1],deviations(MRK)[1][1],deviations(JNJ)[1][1],deviations(WBA)[1][1],deviations(ABT)[1][1],deviations(LLY)[1][1],deviations(DHR)[1][1],deviations(MMM)[1][1],deviations(UNH)[1][1]],
                       index=['PFE', 'BMY', 'MRK','JNJ','WBA','ABT','LLY','DHR',' MMM','UNH']),
     "2019": pd.Series([deviations(PFE)[2][1],deviations(BMY)[2][1],deviations(MRK)[2][1],deviations(JNJ)[2][1],deviations(WBA)[2][1],deviations(ABT)[2][1],deviations(LLY)[2][1],deviations(DHR)[2][1],deviations(MMM)[2][1],deviations(UNH)[2][1]],
                       index=['PFE', 'BMY', 'MRK','JNJ','WBA','ABT','LLY','DHR',' MMM','UNH']),
     "2020": pd.Series([deviations(PFE)[3][1],deviations(BMY)[3][1],deviations(MRK)[3][1],deviations(JNJ)[3][1],deviations(WBA)[3][1],deviations(ABT)[3][1],deviations(LLY)[3][1],deviations(DHR)[3][1],deviations(MMM)[3][1],deviations(UNH)[3][1]],
                       index=['PFE', 'BMY', 'MRK','JNJ','WBA','ABT','LLY','DHR',' MMM','UNH']),
     "Min": pd.Series([max(deviations(PFE))[1],max(deviations(BMY))[1],max(deviations(MRK))[1],max(deviations(JNJ))[1],max(deviations(WBA))[1],max(deviations(ABT))[1],max(deviations(LLY))[1],max(deviations(DHR))[1],max(deviations(MMM))[1],max(deviations(UNH))[1]],
                       index=['PFE', 'BMY', 'MRK','JNJ','WBA','ABT','LLY','DHR',' MMM','UNH'])}
pd.DataFrame(d)

"""Рисунок 5. График цен тикера WBA"""

plt.figure(figsize=(8,6))
li=[]
for i in range(len(WBA['Цена'])): 
  li.append(float(WBA['Откр.'][i]))
plt.plot(range(len(WBA['Цена'])),li[::-1],linewidth = 1.2)
plt.grid()

"""Рисунок 6. График цен тикера DHR"""

plt.figure(figsize=(8,6))
li=[]
for i in range(len(DHR['Цена'])): 
  li.append(float(DHR['Откр.'][i]))
plt.plot(range(len(DHR['Цена'])),li[::-1],linewidth = 1.2)
plt.grid()

"""4. Проверка гипотезы для модельных данных

Рисунок 8. P-значения при нормальном распределении
"""

pvalues = [] 
statistics = [] 
ks=[]
for i in range(1000): 
  distr = sp.shapiro(sp.norm.rvs(size = 250)) 
  statistics.append(distr[0]) 
  pvalues.append(distr[1]) 
kl=(sp.kstest(pvalues, cdf='uniform')[1])
print("P-значение критерия Колмогорова-Смирнова:", round(kl,4))
plt.title('P-значения критерия Шапиро-Уилка') 
plt.hist(pvalues,rwidth=0.85)
plt.show()

"""Таблица 7. Квантили модельных данных"""

q9 = np.quantile(statistics, np.arange(0.1, 1, 0.1)) 
d = {'':pd.Series([round(q9[0],4),round(q9[1],4),round(q9[2],4),round(q9[3],4),round(q9[4],4),round(q9[5],4),round(q9[6],4),round(q9[7],4),round(q9[8],4)],
    index=['Квантиль 0.1', 'Квантиль 0.2', 'Квантиль 0.3','Квантиль 0.4','Квантиль 0.5','Квантиль 0.6','Квантиль 0.7','Квантиль 0.8',' Квантиль 0.9'])}
pd.DataFrame(d)

"""5.	Проверка гипотезы для реальных данных"""

def profvolume(Tiker):
  ldox=[]
  for i in range(1007):
    ldox.append(np.log(float(Tiker["Цена"][i])/float(Tiker["Откр."][i])))
  d = {"Дата":Tiker["Дата"],"Откр":Tiker["Откр."],"Закр":Tiker["Цена"],"Л.Доход":ldox, "Объём": Tiker["Объём"]}
  df = pd.DataFrame(d, index=range(1007))
  for i in range(len(df['Объём'])):
    df['Объём'][i]=float(df['Объём'][i][0:4])
    df['Дата'][i]=int(df['Дата'][i][6:10])

  q9 = np.quantile(df["Объём"], np.linspace(0, 1,4)) 
  df1 = df[df['Объём'] <= q9[1]]
  df11 = df[(df['Объём'] > q9[1])]
  df2 = df11[df11['Объём'] < q9[2]]
  df3 = df[df['Объём'] >= q9[2]]
  return df1,df2,df3

"""Шапиро Уилки"""

def pvvolumes(df1,df2,df3):
  pm,pc,pb=[],[],[]
  dfd1 = df1[df1['Дата'] == 2020]
  pm.append(sp.shapiro(dfd1['Л.Доход'])[1])
  dfd2 = df1[df1['Дата'] == 2019]
  pm.append(sp.shapiro(dfd2['Л.Доход'])[1])
  dfd3 = df1[df1['Дата'] == 2018]
  pm.append(sp.shapiro(dfd3['Л.Доход'])[1])
  dfd4 = df1[df1['Дата'] == 2017]
  pm.append(sp.shapiro(dfd4['Л.Доход'])[1])

  dfd1 = df2[df2['Дата'] == 2020]
  pc.append(sp.shapiro(dfd1['Л.Доход'])[1])
  dfd2 = df2[df2['Дата'] == 2019]
  pc.append(sp.shapiro(dfd2['Л.Доход'])[1])
  dfd3 = df2[df2['Дата'] == 2018]
  pc.append(sp.shapiro(dfd3['Л.Доход'])[1])
  dfd4 = df2[df2['Дата'] == 2017]
  pc.append(sp.shapiro(dfd4['Л.Доход'])[1])

  dfd1 = df3[df3['Дата'] == 2020]
  pb.append(sp.shapiro(dfd1['Л.Доход'])[1])
  dfd2 = df3[df3['Дата'] == 2019]
  pb.append(sp.shapiro(dfd2['Л.Доход'])[1])
  dfd3 = df3[df3['Дата'] == 2018]
  pb.append(sp.shapiro(dfd3['Л.Доход'])[1])
  dfd4 = df3[df3['Дата'] == 2017]
  pb.append(sp.shapiro(dfd4['Л.Доход'])[1])
  return pm,pc,pb

def datafPvalues(Tiker):
  df1,df2,df3=profvolume(Tiker)
  Tikervp=pvvolumes(df1,df2,df3)
  d = {"2017": pd.Series([Tikervp[0][3],Tikervp[1][3],Tikervp[2][3]], index=['М.Объем', 'С.Объем', 'Б.Объем']),
      "2018": pd.Series([Tikervp[0][2],Tikervp[1][2],Tikervp[2][2]], index=['М.Объем', 'С.Объем', 'Б.Объем']),
      "2019": pd.Series([Tikervp[0][1],Tikervp[1][1],Tikervp[2][1]], index=['М.Объем', 'С.Объем', 'Б.Объем']),
      "2020": pd.Series([Tikervp[0][0],Tikervp[1][0],Tikervp[2][0]], index=['М.Объем', 'С.Объем', 'Б.Объем'])}
  f=pd.DataFrame(d)
  return f

"""Рисунок 9. P-значения реальных данных по годам и объему."""

datafPvalues(WBA)

""" Гистограммы Р-значений Рисунок 10-21"""

xm2020=[datafPvalues(PFE)['2020'][2],
        datafPvalues(BMY)['2020'][2],
        datafPvalues(MRK)['2020'][2],
        datafPvalues(JNJ)['2020'][2],
        datafPvalues(WBA)['2020'][2],
        datafPvalues(ABT)['2020'][2],
        datafPvalues(LLY)['2020'][2],
        datafPvalues(DHR)['2020'][2],
        datafPvalues(MMM)['2020'][2],
        datafPvalues(UNH)['2020'][2]]
print("P-значение критерия Колмогорова:", sp.kstest(xm2020, cdf='uniform')[1])
plt.hist(xm2020,rwidth=0.85)
plt.show()