# -*- coding: utf-8 -*-
"""
Created on Sun May  5 15:35:55 2019

@author: swooty
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression, Ridge
import lineFit

df_culture = pd.read_csv('Societal_Culture_Data.csv')
df_suicides = pd.read_csv('master.csv')



#rename things be the same as in the suicide dataset
df_culture.loc[df_culture['Country Name']=='England', 'Country Name'] = 'United Kingdom'
df_culture.loc[df_culture['Country Name']=='Russia', 'Country Name']='Russian Federation'
df_culture.loc[df_culture['Country Name']=='South Korea', 'Country Name']='Republic of Korea'
df_culture.loc[df_culture['Country Name']=='French Switzerland', 'Country Name']='Switzerland'
df_culture.loc[df_culture['Country Name']=='Canada (English-speaking)', 'Country Name']='Canada'
df_culture.loc[df_culture['Country Name']=='USA', 'Country Name']='United States'

tot = pd.merge(df_suicides,df_culture,left_on = 'country',right_on = 'Country Name')

#find increase or decrease rate for each country
slope = np.zeros(tot.country.unique().size)
for ind, country in enumerate(tot.country.unique()):
    country_data = tot.loc[tot['country']==country]
    by_year = country_data.groupby('year').mean()['suicides/100k pop']
    A,B,dA,dB = lineFit.lineFit(by_year.index.tolist(),by_year.tolist(),'year', 'collisions',plot=False)
#    print(country, A, dA)
#    print(by_year.index.tolist())
    slope[ind] = A
slope_df = pd.DataFrame(slope, columns= ['slope'], index = tot.country.unique())
tot2 = pd.merge(tot,slope_df,left_on = 'country',right_index = True)

#plot suicide rates and slopes by region
suicide_bycountry = tot2.groupby('Country Cluster').mean()['suicides/100k pop']
slope_bycountry = tot2.groupby('Country Cluster').mean()['slope']
fig = plt.figure()
fig.clear()
fig.set_size_inches(7.0,5.0)
gs = gridspec.GridSpec(2,1)
gs.update(left=0.15, right=0.95, top=0.9, bottom = 0.35)
gs.update(hspace=0.2,wspace=0.5)
pan = fig.add_subplot(gs[0])
pan.plot(suicide_bycountry, 'bo')
pan.set_ylabel('Suicides per \n 100k population')
pan.set_xticklabels([])
pan.set_title('Suicide rates and trends 1985-2015 by region')
#pan.set_title('Suicides per year averaged 1987-2016')

pan2 = fig.add_subplot(gs[1])
pan2.plot(slope_bycountry, 'go')
pan2.axhline(linestyle='--', color = 'k')
pan2.set_ylabel('Trend in \n suicide rate')
#pan2.set_title('Suicides per year averaged 1987-2016')
plt.xticks(suicide_bycountry.index, rotation = 60)
plt.savefig('Rates_and_trends_by_region.pdf',transparent=True)

egal_byCountry = tot.groupby('Country Name').mean()['Gender Egalitarianism Societal Practices']
males = tot.loc[tot['sex']=='male']
females = tot.loc[tot['sex']=='female']
suicide_byCountry_males = males.groupby('Country Name').mean()['suicides/100k pop']
suicide_byCountry_females = females.groupby('Country Name').mean()['suicides/100k pop']

#plot suicide rates as a function of egalitarianism for men and women
fig = plt.figure()
fig.clear()
fig.set_size_inches(7.0,4.0)
gs = gridspec.GridSpec(1,1)
gs.update(left=0.15, right=0.95, top=0.9, bottom = 0.15)
gs.update(hspace=0.5,wspace=0.5)
pan = fig.add_subplot(gs[0])
pan.plot(egal_byCountry,suicide_byCountry_males, 'bo')
pan.plot(egal_byCountry,suicide_byCountry_females, 'ro')
pan.set_xlabel('Gender Egalitarianism Societal Practices Score')
pan.set_ylabel('Suicides per 100k population')
pan.set_title('Suicides per year averaged 1987-2016')

#run linear regression to see the strongest predictor of suicide rate
X = tot.groupby('Country Name').mean()[tot.columns[14:-1]]
y = tot.groupby('Country Name').mean()['suicides/100k pop']
reg = LinearRegression(normalize=True).fit(X,y)
print(reg.score(X,y))
print(reg.coef_)

simp_names = ['Uncertainty Avoidance Practices',
       'Future Orientation Practices',
       'Power Distance Practices',
       'Institutional Collectivism Practices',
       'Humane Orientation Practices',
       'Performance Orientation Practices',
       'In-group Collectivism Practices',
       'Gender Egalitarianism Practices',
       'Assertiveness Practices',
       'Uncertainty Avoidance Values',
       'Future Orientation Values', 'Power Distance Values',
       'Institutional Collectivism Values',
       'Human Orientation Values',
       'Performance Orientation Values',
       'In-group Collectivism Values',
       'Gender Egalitarianism Values',
       'Assertiveness Values']
norm = reg.coef_/np.sum(np.abs(reg.coef_))

XM = males.groupby('Country Name').mean()[tot.columns[14:-1]]
yM= males.groupby('Country Name').mean()['suicides/100k pop']
regM = LinearRegression(normalize=True).fit(XM,yM)
print(regM.score(XM,yM))
normM = regM.coef_/np.sum(np.abs(regM.coef_))
print(regM.coef_)

XF = females.groupby('Country Name').mean()[tot.columns[14:-1]]
yF = females.groupby('Country Name').mean()['suicides/100k pop']
regF = LinearRegression(normalize=True).fit(XF,yF)
print(regF.score(XF,yF))
normF = regF.coef_/np.sum(np.abs(regF.coef_))
print(regF.coef_)


fig = plt.figure()
fig.clear()
fig.set_size_inches(7.0,5.0)
gs = gridspec.GridSpec(1,1)
gs.update(left=0.4, right=0.95, top=0.9, bottom = 0.1)
gs.update(hspace=0.5,wspace=0.5)
pan = fig.add_subplot(gs[0])
all_coef = np.array([reg.coef_,regM.coef_,regF.coef_]).transpose()
pan.barh(np.arange(tot.columns[14:-1].size)-0.3, norm, height = 0.3,color='b',tick_label=simp_names, label='all')
pan.barh(np.arange(tot.columns[14:-1].size), normM, height = 0.3,color='g',tick_label=simp_names, label = 'male')
pan.barh(np.arange(tot.columns[14:-1].size)+0.3, normF, height = 0.3,color='r',tick_label=simp_names, label = 'female')
pan.set_xlabel('Correlation strength with suicide rates')
pan.set_title('All countries and genders. %.2f R^2 coefficient' %reg.score(X,y))
plt.legend()

plt.savefig('Cultural_predictors.pdf',transparent=True)

X = tot.groupby('Country Name').mean()[tot.columns[14:-1]]
y = tot.groupby('Country Name').mean()['suicides/100k pop']

X=X.to_numpy()
X_evr = np.abs(X[:,:9]-X[:,9:])
X_tra = np.concatenate((X,X_evr),axis=1)
regXtra = Ridge(alpha=0.01,normalize=True).fit(X_tra,y)
print(regXtra.score(X_tra,y))
print(regXtra.coef_)

future_orient_prac = tot.groupby('Country Name').mean()[tot.columns[15]]



fig = plt.figure()
fig.clear()
fig.set_size_inches(7.0,5.0)
gs = gridspec.GridSpec(1,1)
gs.update(left=0.4, right=0.95, top=0.9, bottom = 0.1)
gs.update(hspace=0.5,wspace=0.5)
pan = fig.add_subplot(gs[0])
#pan.plot(future_orient_prac, slope, 'bo')

X_up = tot.groupby('Country Name').mean()[tot.columns[14:-1]]
y_up = slope
reg_up = LinearRegression(normalize=True).fit(X_up,y_up)
print(reg_up.score(X_up,y_up))
print(reg_up.coef_)
norm = reg_up.coef_/np.sum(np.abs(reg_up.coef_))
pan.barh(np.arange(tot.columns[14:-1].size), norm, height = 0.5,color='b',tick_label=simp_names, label='all')

pan.set_xlabel('Correlation strength with suicide slope 1985-2015')
pan.set_title('All countries and genders. %.2f R^2 coefficient' %reg_up.score(X_up,y_up))
