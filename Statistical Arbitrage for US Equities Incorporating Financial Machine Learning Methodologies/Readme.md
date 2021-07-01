This is an ongoing project where I am investigating whether machine learning based methods will improve the performance of a statistical arbitrage system. 
I conduct my analysis on the S&P500 constituents downloaded from CRSP, starting from 1960 till the end of 2020. The current high level process of the system is as follows,
1) Start with the entire universe of S&P500
2) Next conduct a multi step pair selection procedure:
 - Cluster based identification
 - Next, graphical lasso on each cluster to detect pairs with conditional dependence
 - Super cointegration and fractional cointegration test
 - Next, select a subset based on the spread's 1) Hurst exponent and 2) Halflife
3) Next, fit a primary model based on a given threshold rule. (I would like to investigate whether deep learning can add value here)
4) Next, fit an ensemble of meta models to assess the probability of the pair's profitability
5) Use the meta model as a risk management overlay on top of the primary model. 
