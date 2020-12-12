import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""First, we need to read the files and create pandas dataframes from the excel files."""

"""
Let's list out the columns we have in the metadata
"""

"""
Point 1: "You can judge a nation, and how successful it will be, based on how it treats its women and its girls."
 — Barack Obama

Based on the data we have, we can investigate the economic performance of a country using the following factors:
* GDP per capita (PPP)
* Annual growth in GDP (%)
* GNI per capita (PPP)
* Annual growth in GNI (%)

And using the variables that show statistics about women's development, we can establish correlations between women
empowerment and a country's economic performance.

- To perform this investigation, we first need to identify the variables that include women/female in their definition.

- Then we can identify the correlation between women parameters and economic parameters. 

- To elaborate further, we can analyse the correlation between women parameters and education parameters. Also maybe 
birth parameters. 

- We can also build a predictive model that predicts the economic growth based on the development of women.


Point 2: 
“Our progress as a nation can be no swifter than our progress in education. The human mind is our fundamental resource.” 
- John F. Kennedy

- Using the education metrics that we have, we can train a model to predict the growth in economy of a country based on
the improvement in their education. Elaborate on this further. 

- We can correlate education with manufacturing output and plot some graphs about that. 

- Then we can correlate education with the national income and other income based parameters.

- Using clustering modes, we can cluster countries that have similar education parameters and make some statistics from 
them. These would give us a much better perspective as to how education is crucial for a country's economic growth.


Point 3: "An imbalance between the poor and the rich us the oldest and most fatal ailment of all republics." - Plutarch

- Economic inequality can be measured using the GINI index. We can use multiple parameters to draw correlations between
economic inequality and the rest of the parameters listed in the excel sheet.

- We can build predictive models that predict the economic growth/decline based on the inequality.

- As crime correlates with inequality, we can plot some graphs to see how this happens.


"""
data_df = pd.read_excel("/Users/csatyajith/Datasets/WDIdata3decades.xlsx")
data_df_90s = data_df[:79]
data_df_00s = data_df[79:158]
data_df_10s = data_df[158:]
metadata_excel = pd.ExcelFile("/Users/csatyajith/Datasets/WDImetadata.xlsx")

metadata_df = pd.read_excel(metadata_excel, 1)
country_codes = pd.read_excel(metadata_excel, 0)
country_code_mapping = {}

print(metadata_df.columns.tolist())


women_rows = metadata_df.loc[metadata_df["Long definition"].str.contains("women|female|Women|Female")]
women_rows = women_rows.loc[~women_rows["Long definition"].str.contains("Population")]
print(women_rows)

"""Identifying columns with plenty of null values to discard them. We only consider the columns with more than 100 
samples as relevant as anything less than that is insufficient to draw conclusions. Also, we can eliminate all 
population statistics as that is not our primary concern for this analysis."""
null_stats = {}

reduced_df = data_df[women_rows["Code"].tolist()]
reduced_df = reduced_df.dropna(axis=1, thresh=100)
print(reduced_df.columns)

print("Total valid columns are: {}".format(reduced_df.columns))

print("\n The valid columns are: \n")
valid_columns = women_rows.loc[women_rows["Code"].isin(reduced_df.columns.tolist())]
print(valid_columns[["Code", "Indicator Name"]])

"""Now, we can identify the parameters which we wish to consider for analysing the impact of women's development

For effective analysis, we can consider the following parameters which we can intuitively see that might influence the 
economic condition:
1: Literacy rate, youth (ages 15-24), gender parity index (GPI): SE.ADT.1524.LT.FM.ZS
2: Nonpregnant and nonnursing women can do the same jobs as men (1=yes; 0=no): SG.JOB.NOPN.EQ
3: Ratio of female to male labor force participation rate (%) (modeled ILO estimate): SL.TLF.CACT.FM.ZS
4: School enrollment, tertiary (gross), gender parity index (GPI): SE.ENR.TERT.FM.ZS
5: Women who believe a husband is justified in beating his wife (any of five reasons) (%): SG.VAW.REAS.ZS
6: Women who were first married by age 15 (% of women ages 20-24): SP.M15.2024.FE.ZS
7: Women who were first married by age 18 (% of women ages 20-24): SP.M18.2024.FE.ZS
8: Adolescent fertility rate (births per 1,000 women ages 15-19): SP.ADO.TFRT
9: Proportion of seats held by women in national parliaments (%): SG.GEN.PARL.ZS
10: Sex ratio at birth (male births per female births): SP.POP.BRTH.MF
"""

women_parameters = ["SE.ADT.1524.LT.FM.ZS", "SG.JOB.NOPN.EQ", "SL.TLF.CACT.FM.ZS", "SE.ENR.TERT.FM.ZS",
                    "SG.VAW.REAS.ZS", "SP.M15.2024.FE.ZS", "SP.M18.2024.FE.ZS", "SP.ADO.TFRT", "SG.GEN.PARL.ZS",
                    "SP.POP.BRTH.MF"]

womens_df_90s = data_df_90s[women_parameters]
womens_df_00s = data_df_00s[women_parameters]
womens_df_10s = data_df_10s[women_parameters]
womens_df = data_df[women_parameters]

"""
Then, let us put together the economic parameters:
GDP per capita growth (annual %): NY.GDP.PCAP.KD.ZG 
GDP per capita, PPP (constant 2011 international $): NY.GDP.PCAP.PP.KD
GNI growth (annual %): NY.GNP.MKTP.KD.ZG
GNI per capita (constant 2010 US$): NY.GNP.PCAP.KD
"""

economic_parameters = ["NY.GDP.PCAP.KD.ZG", "NY.GDP.PCAP.PP.KD", "NY.GNP.MKTP.KD.ZG", "NY.GNP.PCAP.KD"]
economic_df_90s = data_df_90s[economic_parameters]
economic_df_00s = data_df_00s[economic_parameters]
economic_df_10s = data_df_10s[economic_parameters]
economic_df = data_df[economic_parameters]

"""
Now we can start analysing the data using plots. First, let's plot the joint scatter plot with youth literacy rate 
GPI on the x axis and the GDP per capita PPP on the Y axis.
"""
women_economic_data_10s = pd.concat([womens_df_10s, economic_df_10s], axis="columns")
women_economic_data_00s = pd.concat([womens_df_00s, economic_df_00s], axis="columns")
women_economic_data_90s = pd.concat([womens_df_90s, economic_df_90s], axis="columns")
women_economic_data = pd.concat([womens_df, economic_df], axis="columns")

sns.jointplot(x="SE.ADT.1524.LT.FM.ZS",
              y="NY.GDP.PCAP.PP.KD",
              data=women_economic_data)
plt.show()

"""From the above graph we can observe the following points:
* The countries which have a low "Youth Literacy Rate GPI" have a low GDP per capita PPP.
* The countries which have a high "Youth Literacy Rate GPI" don't necessarily have a high GDP per capita PPP.
* We can also observe that there is no point existing on the top left part of the graph.
Thus, with reasonable accuracy, we can conclude that the countries that don't prioritize women's education and have a
low women's literacy rate are bound to have a low GDP per capita. Prioritizing equal education doesn't really guarantee
success. But, not prioritizing equal education pretty much guarantees failure.
"""

"""
Moving on to the next point,
Vogue magazine once quoted Australian activist GD Anderson as saying: “Feminism isn’t about making women stronger. 
Women are already strong. It’s about changing the way the world perceives that strength.” 

Using the data we have in the existing dataset we can try to identify correlation between a country's perception on
women and their GNI per capita. We can start off with using the following statistic:
Women who believe a husband is justified in beating his wife (any of five reasons) (%): SG.VAW.REAS.ZS
"""
g = sns.regplot(x="SG.VAW.REAS.ZS",
                y="NY.GNP.PCAP.KD",
                data=women_economic_data)
g.set(xlabel="Women who believe a husband is justified in beating his wife (any of five reasons) (%)",
      ylabel="GNI per capita (constant 2010 US$)")
plt.show()

"""
Looking at the downward slope of the above regplot, we can understand that the societies that believe that beating women
is justified have a lower Gross National Income per capita. 

We can further strengthen this point by using the following statistic:
Nonpregnant and nonnursing women can do the same jobs as men (1=yes; 0=no): SG.JOB.NOPN.EQ
"""
g = sns.catplot(x="SG.JOB.NOPN.EQ",
            y="NY.GNP.PCAP.KD", data=women_economic_data, kind="box", order=[0.0, 1.0])
g.set(xlabel="Nonpregnant and nonnursing women can do the same jobs as men (1=yes; 0=no): SG.JOB.NOPN.EQ",
      ylabel="GNI per capita (constant 2010 US$)")
plt.ylim(0, 10000)
plt.show()

"""In the above graph, the graph is rendered unreadable because of the high GNI in the US. Hence, we can acknowledge
that US has a high GNI per capita and that Nonpregnant and nonnursing women can do the same jobs as men in the US. 
And now, if we remove US from the graph, it gets more readable.

The results are rather interesting. In the countries where women can't do the same jobs as men, the median GNI is
higher. However, in the countries where women can do the same jobs as men, then 75th percentile is significantly higher.
Also, the inter-quartile range of the countries where women can do the same jobs as men is higher.

This proves that in countries where there is equality in jobs, there is once again no guarantee of economic performance.
But in countries where there is no equality in jobs, economic under-performance can be expected.
"""

"""
Early marriage and adolescent fertility are often great obstacles to the education of women. Hence we can observe the 
correlation between both those parameters and literacy. The parameters we'll use are the following:
* Literacy rate, youth (ages 15-24), gender parity index (GPI): SE.ADT.1524.LT.FM.ZS
* Women who were first married by age 18 (% of women ages 20-24): SP.M18.2024.FE.ZS
* Adolescent fertility rate (births per 1,000 women ages 15-19): SP.ADO.TFRT
"""
g = sns.relplot(x="SP.M18.2024.FE.ZS",
            y="SP.ADO.TFRT", size="SE.ADT.1524.LT.FM.ZS", data=women_economic_data)
g.set(xlabel="Women who were first married by age 18 (% of women ages 20-24)",
      ylabel="Adolescent fertility rate (births per 1,000 women ages 15-19)")
plt.show()


"""
The above graph is particularly interesting. From this graph, we can observe the following:
* The higher the percentage of women who were first married by the age of 18, the higher is their adolescent fertility
rate
* And if both those are higher, the point is in the top right corner.
* If the point is in the top right corner, it can observed that the size of the point is lower.
* As we previously established a correlation between low women literacy and GDP, we can make a conclusion with 
reasonable accuracy that high adolescent fertility rate, and early marriage correlate with a poor economy.
"""

"""
Next, we can try to identify the correlation between the percentage of women parliamentarians and the Gender parity 
index of school enrolment. For that, we use the following parameters:
* Proportion of seats held by women in national parliaments (%)
* Ratio of female to male labor force participation rate (%) (modeled ILO estimate): 
"""
g = sns.regplot(x="SG.GEN.PARL.ZS",
                y="SL.TLF.CACT.FM.ZS",
                data=women_economic_data)
g.set(xlabel="Proportion of seats held by women in national parliaments (%)",
      ylabel="Ratio of female to male labor force participation rate (%) (modeled ILO estimate)")
plt.show()

"""
As the graph is trending upwards, we can see that the higher the ratio of female to male labor force participation rate,
the higher is the percentage of female parliamentarians
"""

"""Finally, we can use all the parameters we have to try and successfully predict the GDP per capita of a country. 
For this, we'll use the first 210 rows as the training set and the next 28 rows as the test set. 

We can use a multiple linear regression mode to input
"""

import sklearn.linear_model as sklm

train_data = womens_df[:210]
train_data = train_data.fillna(train_data.mean())
train_labels = economic_df[:210]
train_labels = train_labels.fillna(train_labels.mean())
test_data = data_df[210:]
test_data = test_data.fillna(train_data.mean())
test_labels = economic_df[210:]
test_labels = test_labels.fillna(train_labels.mean())

wdi_women = sklm.LinearRegression()
X = train_data[women_parameters].values
Y = train_labels["NY.GDP.PCAP.PP.KD"].values
wdi_women.fit(X, Y)

predictions = wdi_women.fit(X, Y).predict(test_data[women_parameters].values)
vis_df = pd.DataFrame()
vis_df["predictions"] = predictions
vis_df["actual_values"] = test_labels["NY.GDP.PCAP.PP.KD"].values
vis_df["difference"] = predictions - test_labels["NY.GDP.PCAP.PP.KD"].values
print(vis_df)

"""
Looking at the predictions, we can make the following points:
* A country's economic performance can somewhat be predicted by its women parameters. For example, we can see that
the difference between the predictions and actual values is more often than not less than 1000. 
* For countries with high GDP per capita, our model does not accurately predict the GDP. But it accurately identifies
them as high GDP countries. For example, consider rows 16 and 20. These are clearly high GDP countries and our model 
gave these countries the highest GDP predictions among the available samples.
"""

"""
"An imbalance between the poor and the rich us the oldest and most fatal ailment of all republics." - Plutarch

To analyze this imbalance we use the Gini coefficient. In economics, the Gini coefficient, sometimes called the
 Gini index or Gini ratio, is a measure of statistical dispersion intended to represent the income or wealth 
 distribution of a nation's residents, and is the most commonly used measurement of inequality.
 
In our dataset, the GINI coefficient is given by the following code:
GINI index (World Bank estimate): SI.POV.GINI

Some effects of economic inequality, as researchers found are:
* Higher rates of health problems
* A lower population wide level of happiness
* Increase in crime levels

We can investigate the above effects using the following parameters from the dataset:
1: Life expectancy at birth, total (years): SP.DYN.LE00.IN
2: Proportion of population spending more than 10% of household consumption or income on 
out-of-pocket health care expenditure (%): SH.UHC.OOPC.10.ZS
3: Intentional homicides (per 100,000 people): VC.IHR.PSRC.P5
4: Bribery incidence (% of firms experiencing at least one bribe payment request): IC.FRM.BRIB.ZS
5: Firms experiencing losses due to theft and vandalism (% of firms): IC.FRM.THEV.ZS
6: Coverage of social safety net programs (% of population): per_sa_allsa.cov_pop_tot
"""

"""
First, let us look for correlations between intentional homicides and the GINI co-efficient. To get a better 
representation, we remove an outlier on the top right of the graph. However, the outlier supports our point too.
"""
g = sns.regplot(x="SI.POV.GINI",
                y="VC.IHR.PSRC.P5",
                data=data_df)
g.set(xlabel="GINI index (World Bank estimate)",
      ylabel="Intentional homicides (per 100,000 people)")
plt.ylim(0, 80)
plt.show()


"""
As can be observed in the above graph, the rate of intentional homicides increases as the GINI coefficient increases.
Thus, there is a clear correlation between high inequality and high homicide rate.
"""

"""Then we can also investigate the correlation between the life expectancy and income inequality. """

g = sns.regplot(x="SI.POV.GINI",
                y="SP.DYN.LE00.IN", data=data_df)
g.set(xlabel="GINI index (World Bank estimate)",
      ylabel="Life expectancy at birth, total (years)")
plt.show()

"""
Through the graph above, we can see that high inequality correlates with low life expectancy. 
"""
"""
Finally, we can use the above metrics and try to predict the inequality in a particular country. Once again, we separate
our data into training and test. The first 210 rows in our data will be used for training while the rest will be used to
test our model's results.
"""
pred_parameters = ["SP.DYN.LE00.IN", "SH.UHC.OOPC.10.ZS", "VC.IHR.PSRC.P5", "IC.FRM.BRIB.ZS", "IC.FRM.THEV.ZS",
                   "per_sa_allsa.cov_pop_tot"]

train_data = data_df[:210]
train_data = train_data.fillna(train_data.mean())
test_data = data_df[210:]
test_data = test_data.fillna(train_data.mean())

wdi_women = sklm.LinearRegression()
X = train_data[pred_parameters].values
Y = train_data["SI.POV.GINI"].values
wdi_women.fit(X, Y)

predictions = wdi_women.fit(X, Y).predict(test_data[pred_parameters].values)
vis_df = pd.DataFrame()
vis_df["predictions"] = predictions
vis_df["actual_values"] = test_data["SI.POV.GINI"].values
vis_df["difference"] = predictions - test_data["SI.POV.GINI"].values
print(vis_df)

"""
As the error is quite low as seen in the table, we can confidently assert that predicting the GINI coefficient from the
parameters that we listed out.
"""