
#Importing libraries
import pandas as pd
import numpy as np
import os 
from matplotlib import pyplot as plt 
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

#Importing data 
path = os.getcwd()#getting working directory
print(path)
tvdata = pd.read_csv("tv_shows.csv")

#Data infos and basic operations
tvdata.head(6)
tvdata.info()
tvdata.describe()
tvdata = tvdata.drop("Unnamed: 0", axis=1)
#Data type edits
for i in ["Title","Year","Netflix","Hulu","Prime Video","Disney+"]:
    tvdata[i] = tvdata[i].astype("category")    

tvdata = tvdata.drop("type",axis=1)
tvdata["Rotten Tomatoes"] = tvdata['Rotten Tomatoes'].str.replace(r"\D", '')
tvdata["Rotten Tomatoes"] = tvdata["Rotten Tomatoes"].astype("float64")
Total_Year = len((tvdata["Year"].cat.categories))
years = tvdata["Year"].cat.categories

tvdata["Age"] = tvdata["Age"].str.replace(r"\D", '')
tvdata["Age"] = tvdata["Age"].astype("category")
Total_Target_Age = len((tvdata["Age"].cat.categories))
Ages = tvdata["Age"].cat.categories

np.where(pd.isnull(tvdata))
print(tvdata[tvdata.isnull().any(axis=1)]["Age"].head())
#nan values are not non for Age column, eiditing them as NaN
tvdata["Age"] = tvdata["Age"].replace("", np.nan)

#Renaming Prime Video column
tvdata.rename(columns={"Prime Video": "Prime_Video"}, inplace=True)

#Dividing all categorical variables into indexes to see corrupted or blank values
Total_Target_Netflix = len((tvdata["Netflix"].cat.categories))
Netflix = tvdata["Netflix"].cat.categories

Total_Target_Hulu = len((tvdata["Hulu"].cat.categories))
Hulu = tvdata["Hulu"].cat.categories

Total_Target_Prime_Video = len((tvdata["Prime_Video"].cat.categories))
Prime_Video = tvdata["Prime_Video"].cat.categories

Total_Target_Disney = len((tvdata["Disney+"].cat.categories))
Prime_Video_Disney = tvdata["Disney+"].cat.categories

#Continious variables
tvdata.rename(columns={"Rotten Tomatoes":"Rotten_Tomatoes"}, inplace=True)
#tvdata.info()
tvdata["Rotten_Tomatoes"] = tvdata["Rotten_Tomatoes"].div(10)
#tvdata.describe()
  
#visulizations
vistvdata = tvdata.copy()
vistvdata.nlargest(10,"IMDb").Title
vistvdata.nlargest(10,"IMDb").IMDb

#TOP 10 IMDb series
f = plt.figure(figsize=(10,5))
f.add_subplot(111)
plt.axes(axisbelow=True)
plt.barh(vistvdata.sort_values("IMDb", ascending = False)["Title"].values[:10],
         vistvdata.sort_values("IMDb", ascending = False)["IMDb"].values[:10],color="Red",
         hatch="u",edgecolor='Blue')
for index, value in enumerate(vistvdata.sort_values("IMDb", ascending = False)["IMDb"].values[:10]):
    plt.text(value, index, str(value))
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("IMDb Ratings",fontsize=18)
plt.title("Top 10 Series",fontsize=20)
plt.grid(alpha=0.3,which='both')

#TOP 10 Rotten_Tomatoes series
f = plt.figure(figsize=(10,5))
f.add_subplot(111)
plt.axes(axisbelow=True)
plt.barh(vistvdata.sort_values("Rotten_Tomatoes", ascending = False)["Title"].values[:10],
         vistvdata.sort_values("Rotten_Tomatoes", ascending = False)["Rotten_Tomatoes"].values[:10],color="Green",
         hatch="u",edgecolor='Red')
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Rotten_Tomatoes Ratings",fontsize=18)
plt.title("Top 10 Series",fontsize=20)
plt.grid(alpha=0.3,which='both')


#TOP 10 series respect too imdb points and rotten tomatoes
f = plt.figure(figsize=(10,5))
f.add_subplot(111)
plt.axes(axisbelow=True)
plt.barh(vistvdata.sort_values("IMDb", ascending = False)["Title"].values[:10],
         vistvdata.sort_values("IMDb", ascending = False)["Rotten_Tomatoes"].values[:10],color="Blue",
         hatch="o",edgecolor='Red')
for index, value in enumerate(vistvdata.sort_values("IMDb", ascending = False)["Rotten_Tomatoes"].values[:10]):
    plt.text(value, index, str(value))
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Rotten Tomatoes Ratings",fontsize=18)
plt.title("Top 10 Series",fontsize=20)
plt.grid(alpha=0.3,which='both')




#Plotting changes in years of imdb
vistvdata_imdb = vistvdata.drop(["Rotten_Tomatoes", "Age"], axis=1)
vistvdata_imdb = vistvdata_imdb.dropna().reset_index(drop=True)
Year_imdb = vistvdata_imdb.groupby("Year")["IMDb"].agg(["mean", "count"]).sort_values("mean", ascending=False)
Year_imdb.columns = ["IMDb Mean", "Number of Shows"]
Year_imdb.head(10)
Year_imdb.info()
Year_imdb.rename(columns={"IMDb Mean":"IMDb_Mean"}, inplace=True)
Year_imdb.rename(columns={"Number of Shows":"Number_series"}, inplace=True)

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(Year_imdb.sort_index()["IMDb_Mean"],color="red")
plt.xlabel("Years",color="black",fontsize=20)
plt.ylabel("IMDb Ratings",color="black",fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.show()


#Plot05
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.relplot(x="IMDb_Mean", y="Number_series", data=Year_imdb,)
plt.title("Distribution of IMDb Ratings Respect to Number of Series")
plt.xlabel("IMDb Ratings", fontsize=15)
plt.ylabel("Number of Series", fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()

#plot06
vistvdata["Year"] = vistvdata["Year"].astype("int32") 
vistvdata['Year_Cut'] = pd.qcut(vistvdata['Year'], q=4)
vistvdata["Year_Cut"].cat.categories

sns.set_style("whitegrid")
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
plt.title("IMDb Ratings by Years", fontsize=25, 
          color="DarkBlue", fontname="Console")
z = sns.violinplot(data=vistvdata,
                   x = "Year_Cut", y= "IMDb",
                   palette= "YlOrRd")
plt.ylabel("IMDb Ratings", fontsize=25,color="DarkBlue")
plt.xlabel("Year Interval", fontsize=25,color="DarkBlue")
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(frameon=True, fancybox=True, shadow=True, framealpha=1, prop={"size":20})
plt.show()

#plot07
vistvdata_platform = vistvdata.drop(["Age", "Year"], axis=1)
vistvdata_platform = vistvdata_platform.drop("Year_Cut", axis=1)
vistvdata_platform.dropna()
NetflixPl = vistvdata_platform[vistvdata_platform["Netflix"]==1]
HuluPl = vistvdata_platform[vistvdata_platform["Hulu"]==1]
Prime_VideoPl = vistvdata_platform[vistvdata_platform["Prime_Video"]==1]
DisneyPl = vistvdata_platform[vistvdata_platform["Disney+"]==1]
nsr = NetflixPl.dropna(subset=["IMDb"]).mean()
hsr = HuluPl.dropna(subset=["IMDb"]).mean()
psr = Prime_VideoPl.dropna(subset=["IMDb"]).mean()
dsr = DisneyPl.dropna(subset=["IMDb"]).mean()
nsr = nsr.to_frame()
hsr = hsr.to_frame()
psr = psr.to_frame()
dsr = dsr.to_frame()
mergedDf = nsr.merge(hsr, left_index=True, right_index=True)
mergedDf = mergedDf.merge(psr, left_index=True, right_index=True)
mergedDf = mergedDf.merge(dsr, left_index=True, right_index=True)
mergedDf
mergedDf.rename(columns={"0_x": "Netflix", "0_y":"Hulu", "0_x":"Prime_Video", "0_y":"Disney"}, inplace=True)
mergedDf2 = mergedDf.copy()
mergedDf2.columns = ["Netflix", "Hulu", "Prime_Video", "Disney"]
mergedDf2 = mergedDf2.reset_index()

#IMDb and Tomatoes Ratings
mergedDf2[0:2].plot(kind="bar")
# Which platform is prefered
mergedDf2[2:6].plot(kind="bar")

sns.set_style("whitegrid")
f , axes = plt.subplots(1, 2, figsize = (10,7))
k1 = mergedDf2[0:2].plot(kind="bar",ax=axes[0])
k2 = mergedDf2[2:6].plot(kind="bar",ax=axes[1])
axes[0].set_title("IMDb and Tomatoes Ratings")
axes[0].set_xticks([0,1])
axes[0].set_xticklabels(["IMDb", "Rotten Tomatoes"],rotation=20)
axes[1].set_title("Platform Preferences")
axes[1].set_xticklabels(["Netflix", "Hulu", "Prime_Video", "Disney"],rotation=20)
#k1.xticks([0,1],labels1,rotation=20, fontsize = 10)
axes[0].set_ylabel("Number of Movies", fontsize=10,color="black")
axes[0].set_xlabel("IMDB and Tomatoes Ratings", fontsize=10,color="black")
axes[1].set_ylabel("Population %", fontsize=10,color="black")
axes[1].set_xlabel("Platforms", fontsize=10,color="black")
axes[0].legend(loc='upper center',bbox_to_anchor=(0.8, 1), frameon=True, fancybox=True, shadow=False, framealpha=0.6, prop={"size":9})
plt.show()

    



