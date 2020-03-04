import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')



class HBV():

    def __init__(self, id, cdir):
        self.id = id
        self.cdir = cdir

        try:
            self.result = pd.read_csv(self.cdir , delimiter=r'\s+', encoding='cp1252', skiprows=[1], parse_dates=[['date', 'time']])

            for col in self.result.columns[1:]:
                self.result[col] = self.result[col].astype(float)

            self.residual = self.result['1OBSRUNOFF'] - self.result['1SIMRUNOFF']
            self.r2 = 1 - np.sum(self.residual ** 2)/np.sum((self.result['1OBSRUNOFF']-np.average(self.result['1SIMRUNOFF'])) ** 2)

        except IOError:
            print('File not accessible\t ID = {}'.format(self.id))

    def plot_discharge(self):
        fig = plt.figure()
        ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
        plt.title('Catchment id {}'.format(self.id), fontsize=15)
        ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)
        plt.title('NSE = {}'.format(self.r2), fontsize=15)

        ax1.plot(self.result['1OBSRUNOFF'], label='OBSRUNOFF', linewidth=1)
        ax1.plot(self.result['1SIMRUNOFF'], label='SIMRUNOFF', linewidth=1)
        ax1.legend()

        ax2.plot(abs(self.residual), label='Res', linewidth=1)
        ax2.legend()

        plt.show()

    def plot_acc(self):
        fig = plt.figure()

        ax1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
        plt.title('Catchment id {}'.format(self.id), fontsize=15)
        ax2 = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)
        plt.title('NSE = {}'.format(self.r2), fontsize=15)

        ax1.plot(self.result['1ACCOBSRUNOFF'], label='OBSRUNOFF', linewidth=1)
        ax1.plot(self.result['1ACCSIMRUNOFF'], label='SIMRUNOFF', linewidth=1)
        ax1.legend()

        ax2.plot(self.residual, label='Res', linewidth=1)
        ax2.legend()

        plt.show()

    def write_output(self):
        output = self.result[['date_time', '1OBSRUNOFF', '1SIMRUNOFF']]
        file_name = 'output_{}.xlsx'.format(self.id)
        output.to_excel(file_name)

def get(ids):
    catchments = {}
    for id in ids:
        cdir = 'C:\PINE\Data\pineout{}.txt'.format(id)
        catchments[id] = HBV(id, cdir)
    return catchments

def all_catchments():
    missing = [21, 57, 68]
    all = [index for index in range(1, 105)]
    for n in missing: all.remove(n)
    catchments = get(all)

    return catchments

def write_r2(catchments):
    data = {'NSE': []}
    for key in catchments:
        data['NSE'].append(catchments[key].r2)

    df = pd.DataFrame(data, index=[key for key in catchments])
    df.to_csv('r2.txt')
    print(df)


#r2 = pd.read_csv('r2.txt', index_col=0)
#best = r2.sort_values(by='NSE', ascending=False).head(n=10)

catch_14 = HBV(14, 'C:\PINE\Data\pineout14.txt')

#catch_14.plot_discharge()
catch_14.plot_discharge()

cirtical_columns = ['date_time', '1OBSRUNOFF', '1SIMRUNOFF']
