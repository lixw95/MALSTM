from jqdatasdk import *
import jqdatasdk
import pandas as pd
import numpy as np
jqdatasdk.auth('17764217140', '17764217140xW')
indexs = get_index_stocks('000300.XSHG')
a = ['2017q4']

for i in range(0, len(a)):
    #locals()['df' + a[i]] = get_fundamentals(query(income, balance, cash_flow).filter(income.code.in_(indexs)), statDate=a[i])
    dataList = []
    df = get_fundamentals(query(valuation.code, valuation.capitalization, valuation.market_cap, valuation.circulating_market_cap,
            valuation.pe_ratio, valuation.pe_ratio_lyr, valuation.pb_ratio, \

            balance.total_assets, balance.total_liability, balance.total_owner_equities, \

            income.operating_cost, income.total_operating_cost, income.net_profit, \
            cash_flow.subtotal_operate_cash_inflow, cash_flow.net_operate_cash_flow, cash_flow.net_invest_cash_flow,
            cash_flow.cash_equivalent_increase
                                                   ).filter(valuation.code.in_(indexs)),
                                             statDate=a[i])



    df["date"] = a[i]
    df.to_csv('D:/download_code/DA-RNN-attchange1/DA-RNN-master/dataset/JQresult.csv')

'''
dataList.append(df)
finalDf = pd.concat(dataList)
print(finalDf)
'''