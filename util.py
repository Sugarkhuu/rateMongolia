import csv
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt


def get_regions(region_path):
    region  = pd.read_csv(region_path, delimiter=',')
    regions = region['Country Name'].unique()
    regions = regions[:-3]
    return regions

def get_data(data_path, regions):
    df = pd.read_csv('wdi_data.csv', delimiter=',')
    df = df[:-5]
    df = df[~df['Country Name'].isin(list(regions))]
    df.rename(columns={'Series Code':'series',
                          'Country Code':'cc_code',
                          'Country Name':'cc_name'}, 
                          inplace=True)
    df.drop(columns = "Series Name", inplace=True)
    col_names = df.columns
    col_names = [re.sub(r'\s\[YR\d*\]$', '', col) for col in col_names]
    df.columns = col_names
    df["series"].replace({"FP.CPI.TOTL.ZG": "Inflation (%, annual)","FR.INR.LEND": "Lending rate (%, pa)","FR.INR.DPST": "Deposit rate (%, pa)",
          "FR.INR.LNDP": "i_sprd","PX.REX.REER": "reer","PA.NUS.FCRF": "lcy_usd","PA.NUS.PPPC.RF": "p_rat"}, inplace=True)
    df = df.replace(r'^\.+$', np.nan, regex=True)

    num_col_names = df.drop(columns = ['cc_name','cc_code','series']).columns
    df[num_col_names] = df[num_col_names].apply(pd.to_numeric, errors='coerce')
    return df

def calc_means(df,first_final, last_final):
    df = df.copy()
    for first in range(1960,2011,10):
        for last in range(first+9,2021,10):
            col = df.loc[: , str(first):str(last)]
            df['mean_' + str(first) + ':' + str(last)] = col.mean(axis=1)
    first = first_final
    last  = last_final
    df['mean_' + str(first) + ':' + str(last)] = col.mean(axis=1)
    return df

def calc_fx_depr(df,first_final, last_final):
    df = df.copy()
    for first in range(1960,2011,10):
        for last in range(first+9,2021,10):
            df['depr_' + str(first) + ':' + str(last)] = 100*((df[str(first)]/df[str(last)])**(1/(last-first))-1)
    first = first_final
    last  = last_final
    df['depr_' + str(first) + ':' + str(last)] = -100*((df[str(first)]/df[str(last)])**(1/(last-first))-1)
    return df

def myplot(plot_df, x_series, y_series,first_final, last_final):
    plot_df = plot_df.copy()
    x_coords = plot_df[x_series]
    y_coords = plot_df[y_series]
    names    = plot_df['cc_name']
    
    for i,name in enumerate(names):
        x = x_coords[i]
        y = y_coords[i]
        if name != 'Mongolia':
            plt.scatter(x, y, marker='o', color='red')
            plt.text(x+0.1, y+0.1, name, fontsize=10)
        else:
            plt.scatter(x, y, marker='*',s=300,color='blue',alpha = 0.5)
            plt.annotate(name, xy=(x, y),  xycoords='data',
                xytext=(x-5, y+5),size=25,
                arrowprops=dict(arrowstyle="->",linewidth=3,
                                connectionstyle="arc3,rad=0",color='black'))
    z = np.polyfit(x_coords, y_coords, 1)
    p = np.poly1d(z)
    plt.plot(x_coords,p(x_coords),"r--")
    plt.ylabel(y_series)
    plt.xlabel(x_series)
    plt.title(x_series + ' versus ' + y_series + ' (average between ' +  str(first_final) + ' and ' + str(last_final) + ')',
              fontweight="bold",fontsize=20)
    plt.grid()
    plt.show()
    

region_path = 'wdi_data_region.csv'
data_path   = 'wdi_data.csv'
first_final = 2010
last_final  = 2018

regions = get_regions(region_path)
data    = get_data(data_path, regions)
data_with_mean = calc_means(data,first_final, last_final)

fx_data = data[data['series']=='lcy_usd'].reset_index()
fx_depr_with_mean = calc_fx_depr(fx_data,first_final, last_final)
depr_var = 'depr_' + str(first_final) + ':' + str(last_final)
fx_depr_short = fx_depr_with_mean[['cc_name',depr_var]]
fx_depr_short = fx_depr_short.rename(columns={depr_var: "LCY depreciation against USD (%, annual)"})

mean_var = 'mean_' + str(first_final) + ':' + str(last_final)
short_df = data_with_mean[['cc_name','cc_code','series',mean_var]]
plot_df = short_df.pivot_table(mean_var, ['cc_name','cc_code'],'series')
plot_df = pd.DataFrame(plot_df.to_records())


plot_df = plot_df.merge(fx_depr_short,how='inner')
plot_df = plot_df.fillna(plot_df.mean())
plot_df = plot_df[(plot_df['cc_name']!='Venezuela, RB') & (plot_df['cc_name']!='South Sudan')
            & (plot_df['cc_name']!='Myanmar') ].reset_index()


"Inflation (%, annual)"
"Lending rate (%, pa)"
"Deposit rate (%, pa)"
"LCY depreciation vs USD (%, annual)"

myplot(plot_df,"Deposit rate (%, pa)","Lending rate (%, pa)",first_final, last_final)
myplot(plot_df,"Inflation (%, annual)","Deposit rate (%, pa)",first_final, last_final)
myplot(plot_df,"Inflation (%, annual)","LCY depreciation against USD (%, annual)",first_final, last_final)
myplot(plot_df,"LCY depreciation against USD (%, annual)","Deposit rate (%, pa)",first_final, last_final)


mon_data = data[data['cc_name'] == 'Mongolia']
var_names = list(mon_data['series'].unique())
mon_data = mon_data.T
mon_data = mon_data[3:]
mon_data.columns = var_names
mon_data = mon_data.loc[mon_data.index >= "1998",:]


plt.plot(mon_data.index, mon_data['Lending rate (%, pa)'],'red', label = 'Lending rate (%, pa)')
plt.plot(mon_data.index, mon_data['Deposit rate (%, pa)'],'blue', label = 'Deposit rate (%, pa)')
plt.plot(mon_data.index, mon_data['Inflation (%, annual)'],'k',marker='o', label = 'Inflation (%, annual)')
plt.ylabel('percent')
plt.title('Interest rates and Inflation',fontweight="bold",fontsize=20)
plt.legend()
plt.grid()
plt.show()



#import seaborn
#seaborn.set(style='ticks')
#
#np.random.seed(0)
#N = 37
#_genders= ['Female', 'Male', 'Non-binary', 'No Response']
#df = pd.DataFrame({
#    'Height (cm)': np.random.uniform(low=130, high=200, size=N),
#    'Weight (kg)': np.random.uniform(low=30, high=100, size=N),
#    'Gender': np.random.choice(_genders, size=N)
#})
#
#fg = seaborn.FacetGrid(data=df, hue='Gender', hue_order=_genders, aspect=1.61)
#fg.map(plt.scatter, 'Weight (kg)', 'Height (cm)').add_legend()


#csv_reader = csv.reader("wdi_data.csv", delimiter=',')


#import csv
#
#with open('employee_birthday.txt') as csv_file:
#    csv_reader = csv.reader(csv_file, delimiter=',')
#    line_count = 0
#    for row in csv_reader:
#        if line_count == 0:
#            print(f'Column names are {", ".join(row)}')
#            line_count += 1
#        else:
#            print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
#            line_count += 1
#    print(f'Processed {line_count} lines.')