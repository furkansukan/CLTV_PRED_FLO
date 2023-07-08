##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################


##############################################################
# 1. Verinin Hazırlanması (Data Preperation)
##############################################################

# FLO satış ve pazarlama faaliyetleri için roadmap
# belirlemek istemektedir. Şirketin orta uzun vadeli plan
# yapabilmesi için var olan müşterilerin gelecekte şirkete
# sağlayacakları potansiyel değerin tahmin edilmesi
# gerekmektedir.

# Veri Seti Hikayesi

# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.
# 13 Değişken 19.945 Gözlem

# Değişkenler
# master_id Eşsiz müşteri numarası
# order_channel Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel En son alışverişin yapıldığı kanal
# first_order_date Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


##########################
# Gerekli Kütüphane ve Fonksiyonlar
##########################

# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


#########################
# Görev 1: Veriyi Hazırlama
#########################

# ADIM 1 : Verinin Okunması


df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy()

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(5))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Adım2: Aykırı değerleri baskılamak için gerekli olan outlier_thresholds
# ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.
# Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

# Adım3: "order_num_total_ever_online", "order_num_total_ever_offline",
# "customer_value_total_ever_offline",
# "customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

df.describe().T

# Adım4: Omnichannel müşterilerin hem online'dan hem de offline
# platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df.head()

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df["total_order"] = df["total_order"].astype(int)
df["total_value"] = df["total_value"].astype(int)

# Adım5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.dtypes

df["first_order_date"] = df["first_order_date"].astype("datetime64")
df["last_order_date"] = df["last_order_date"].astype("datetime64")
df["last_order_date_online"] = df["last_order_date_online"].astype("datetime64")
df["last_order_date_offline"] = df["last_order_date_offline"].astype("datetime64")



#########################
# Görev 2: CLTV Veri Yapısının Oluşturulması
#########################

# Adım1: Veri setindeki en son alışverişin yapıldığı tarihten
# 2 gün sonrasını analiz tarihi olarak alınız.

df["last_order_date"].max()
# Timestamp('2021-05-30 00:00:00')
today_date = dt.datetime(2021,6,1)

# Adım2: customer_id, recency_cltv_weekly, T_weekly,
# frequency ve monetary_cltv_avg değerlerinin yer
# aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak,
# recency ve tenure değerleri ise haftalık cinsten ifade edilecek.


cltv_df = pd.DataFrame()

cltv_df["customer_id"] = df["master_id"]

cltv_df["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]")/ 7

cltv_df["T_weekly"] = (today_date - df["first_order_date"]).astype("timedelta64[D]") / 7

cltv_df["frequency"] = df["total_order"]

cltv_df["monetary_cltv_avg"] = df["total_value"] / df["total_order"]
cltv_df.dtypes
#########################
# Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
#########################

# Adım1: BG/NBD modelini fit ediniz.

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv
# dataframe'ine ekleyiniz.

cltv_df["exp_sales_3_month"] = bgf.predict(12,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv
# dataframe'ine ekleyiniz.

cltv_df["exp_sales_6_month"] = bgf.predict(24,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])


plot_period_transactions(bgf)
plt.show(block=True)


# Adım2: Gamma-Gamma modelini fit ediniz.
# Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv
# dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"],
        cltv_df["monetary_cltv_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
        cltv_df["monetary_cltv_avg"])

# Adım3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency_cltv_weekly"],
                                   cltv_df["T_weekly"],
                                   cltv_df["monetary_cltv_avg"],
                                   time=6, # 6 aylık
                                   freq="W", # T'nin frekans bilgisi
                                   discount_rate=0.01
                                   )

# Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv.sort_values(ascending=False).head(20)

#########################
# Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması
#########################

# Adım1: 6 aylık CLTV'ye göre tüm müşterilerinizi
# 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv_df["cltv"] = cltv

cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels= ["D", "C", "B", "A"])

cltv_df.groupby("segment").agg({"mean", "sum", "count"})








