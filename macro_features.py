import pandas as pd
from curve_engine import building_features
import numpy as np

def macro_engine():

    def load_data():
        df= building_features()
        return df

    df = load_data()

#Inflation (YOY)


    def add_inflation(df):
        df = df.sort_index()
        df.index = pd.to_datetime(df.index)
        monthly_cpi = df["CPI"].resample("M").last()
        infl = np.log(monthly_cpi).diff(12)
        df["inflation_yoy"] = df.join(infl.rename("inflation_yoy"), how="left")["inflation_yoy"]
        df["inflation_yoy"]  = df["inflation_yoy"].ffill()
        return df 
    df = add_inflation(df)

# real rates
    def add_real_rates(df):
        df["real_2Y"]   = df["2Y"] - df["inflation_yoy"]
        df["real_10Y"] = df["10Y"]  - df["inflation_yoy"]
        return df 
    df = add_real_rates(df)


#momentum
    def add_momentum(df):
        df["inflation_momentum"] = df["inflation_yoy"].diff()
        df["real_rate_momentum"] = df["real_2Y"].diff()
        return df
    df = add_momentum(df)


# z score

    def add_zscores(df):

        window = 60

        # curve slope z score
        df["slope_z"] = (
            (df["curve_slope"] - df["curve_slope"].rolling(window).mean()) 
        / df["curve_slope"].rolling(window).std()
        )

        # real rate z score
        df["real_2Y_z"]  = (
            (df["real_2Y"] - df["real_2Y"].rolling(window).mean()) 
        / df["real_2Y"].rolling(window).std()
        )

        #inflation momentum z score
        df["infl_mom_z"] = (
            (df["inflation_momentum"] - df["inflation_momentum"].rolling(window).mean()) 
        / df["inflation_momentum"].rolling(window).std()
        )

        return df

    df = add_zscores(df)

# macro fx signal

    def macro_fx_signal(row):
        slope_z = row["slope_z"]
        real_z = row["real_2Y_z"]
        infl_mom_z = row["infl_mom_z"]

        # score = 0
        score = (
        1.5 * real_z +
        1.0 * slope_z +
        0.5 * infl_mom_z
    )
        
        
        if score >= 1:
            return "USD strong"
        elif score <=-1:
            return "USD weak"
        else:
            return "neutral"
        
    
    df["macro_fx_signal"]  = df.apply(macro_fx_signal, axis = 1)
    return df 

df = macro_engine()
print(df.tail())