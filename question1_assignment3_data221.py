import pandas as pd

crime_dataframe = pd.read_csv("crime.csv")  #Loading crime.csv into a pandas Dataframe

#Computing multiple statistics all at once(using .agg())
violent_crimes_column = crime_dataframe["ViolentCrimesPerPop"].agg(
    ["mean","median","std","min","max"]
)

#Printing the statistical measures.  #Rounding to 4 decimal places
print("Following statistical measures for ViolentCrimesPerPop:")
print(f"Mean: {violent_crimes_column['mean']:.4f}")
print(f"Median: {violent_crimes_column['median']:.4f}")
print(f"Standard deviation: {violent_crimes_column['std']:.4f}")
print(f"Minimum value: {violent_crimes_column['min']:.4f}")
print(f"Maximum value: {violent_crimes_column['max']:.4f}")



"""
Compare the mean and median. Does the distribution look symmetric or skewed? Explain briefly.
- The mean of the ViolentCrimesPerPop column is 0.4412 and the median is 0.3900
  We can see the mean > median(mean is greater than the median) so the distribution of this column appears to be
  right skewed. It may be a slight skew to the right because the difference between the mean and median is about 0.05.


If there are extreme values (very large or very small), which statistic is more affected: mean or median? Explain why.
- The mean will be more affected by extreme values(outliers), because it is calculated using every value in the dataset,
  so outliers are taken into account, while the median depends on the order position of the values, as a result is less
  influenced by outliers. 
"""
