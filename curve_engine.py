import pandas as pd
from data_pipeline import master_data

def building_features():

    df = master_data()

# simple avg 
    df["curve_level"]  = (df["2Y"] + df["10Y"])/2

#slope

    df["curve_slope"] = df["10Y"] - df["2Y"]

    df["slope_change"] = df["curve_slope"].diff()

    def classify_regime(slope):
        if slope > 1.0:
            return "steep"
        elif slope < 0:
            return "inverted"
        else:
            return "flat" 
    
    df["curve_regime"] = df["curve_slope"].apply(classify_regime)


    def fx_bias(row):
        slope = row["curve_slope"]
        level = row["curve_level"]

        if slope> 0.5 and level>3:
            return "USD bullish"
        elif slope < 0:
            return "USD bearish"
        else:
            return "neutral"
        
    

    df["usd_bias"]  =  df.apply(fx_bias, axis = 1)

    return df

df = building_features()

print(df.tail())