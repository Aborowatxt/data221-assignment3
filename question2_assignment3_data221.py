import pandas as pd
import matplotlib.pyplot as plt

crime_dataframe = pd.read_csv("crime.csv")  #Loading crime.csv into a pandas Dataframe

# Plotting Histogram
plt.hist(crime_dataframe["ViolentCrimesPerPop"], edgecolor="black")  #edgecolor separates each bar
plt.title('Distribution of Violent Crime Rates per Population') #Title
plt.xlabel("ViolentCrimesPerPop")  #X-Axis Labeling
plt.ylabel("Frequency") #Y-Axis Labeling
plt.show()  #Display the Histogram


#Plotting boxplot
plt.boxplot(crime_dataframe["ViolentCrimesPerPop"])
plt.title("Box Plot of ViolentCrimesPerPop") #Title
plt.xlabel("ViolentCrimesPerPop") #X-Axis labeling
plt.ylabel("ViolentCrimesPerPop") #Y-Axis labeling
plt.show() #Display the Boxplot.


"""
Looking at the histogram, most of the values are concentrated at lower levels, with fewer values as the crime rate 
increases. This shows that the ViolentCrimesPerPop data is right-skewed. The box plot shows that the median is closer 
to the lower end of the data range. This supports the right skew observed in the histogram. The box plot also indicates
the presence of outliers, particularly at the higher end of the data.
"""





