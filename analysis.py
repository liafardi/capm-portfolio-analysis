import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.optimize as sco


#Read data returns from Excell
file = '/Users/afardi/Documents/MSc/3ο Trimester/Finance/Assignement/Returns.xlsx'

returns=pd.read_excel(file,sheet_name='Returns')
mcap=pd.read_excel(file,sheet_name='Market Cap')
rf=pd.read_excel(file,sheet_name='Risk Free')



#Clean dates
returns['Date'] = pd.to_datetime(returns['Date'])
mcap['Date'] = pd.to_datetime(mcap['Date'])
rf['Date'] = pd.to_datetime(rf['Date'])


#returns
returns_clear = returns.drop(columns=['Date'])
returns_clear = returns_clear.apply(pd.to_numeric, errors='coerce')
#mcap
mcap_clear = mcap.drop(columns=['Date'])
mcap_clear = mcap_clear.apply(pd.to_numeric, errors='coerce')
#align 
mcap_clear = mcap_clear.iloc[1:].reset_index(drop=True)
returns_clear = returns_clear.reset_index(drop=True)

returns_clear = returns_clear.loc[:, ~returns_clear.columns.str.contains("EURIBOR")]

#risk free
rf_clear = rf.iloc[:, 1].reset_index(drop=True)
rf_clear = pd.to_numeric(rf_clear, errors='coerce')
#align
rf_clear = rf_clear.iloc[1:].reset_index(drop=True)



#Market Portfolio (VALUE-WEIGHTED)
market_cap = returns_clear.columns[:int(len(returns_clear.columns)/2)]

weights = mcap_clear[market_cap].div(
    mcap_clear[market_cap].sum(axis=1), axis=0
)

Rm = (returns_clear[market_cap] * weights).sum(axis=1)

#Data
data = returns.loc[1:].copy()
data = data.rename(columns={'Date': 'date'})
data = data.reset_index(drop=True)
data['riskfree'] = rf_clear
data['Market Portfolio'] = Rm



#RF monthly
rf_m = ((1+(rf_clear/100))**(1/12))-1
rf_m = rf_m.reset_index(drop=True)
returns_clear = returns_clear.reset_index(drop=True)
#Excess Returns
excess_returns = returns_clear.sub(rf_m, axis=0)
excess_returns['Market Portfolio'] = Rm.reset_index(drop=True) - rf_m
print(excess_returns.shape)

#statistics
er = excess_returns
mean_er = er.mean()
std_er = er.std()


print(data)
print("\nAverage return\n")
print(mean_er)
print("\nRisk\n")
print(std_er)

#Variance
var=returns_clear.var()
print("\nVariance:\n")
print(var)


#covariance
cov_matrix=returns_clear.cov()
print("\nCovariance Matrix:\n")
print(cov_matrix)


alphas = pd.Series(index=er.columns, dtype=float)
betas = pd.Series(index=er.columns, dtype=float)
p_alpha = pd.Series(index=er.columns, dtype=float)
p_beta = pd.Series(index=er.columns, dtype=float)
r_squared = pd.Series(index=er.columns, dtype=float)

for (columnName, columnData) in er.items(): 
    if columnName=='Market Portfolio':
        continue
    x=er['Market Portfolio']
    y = er[columnName]
    
    df = pd.concat([x, y], axis=1).dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    x_clean = sm.add_constant(df.iloc[:, 0])
    y_clean = df.iloc[:, 1]
    
    result = sm.OLS(y_clean,x_clean).fit()
    
    alphas[columnName] = result.params.iloc[0]
    betas[columnName] = result.params.iloc[1]
    p_alpha[columnName] = result.pvalues.iloc[0]
    p_beta[columnName]  = result.pvalues.iloc[1]
    r_squared[columnName] = result.rsquared
    #print(columnName,result.rsquared)
    #print(result.summary())
print(betas)

beta=betas.T.squeeze()
capm_pred= beta*mean_er['Market Portfolio']
print(capm_pred)
ret = mean_er.T
risk = std_er.T
op_set = pd.concat([ret, risk], axis=1)
op_set.columns = ["return", "risk"]
op_set = op_set.drop(index='Market Portfolio', errors="ignore")
op_set = op_set[op_set["risk"] < 0.2]
op_set = op_set.dropna()


print(ret)
print(risk)
print(op_set)
op_set.plot(
   x="risk", 
   y="return", 
   kind='scatter', 
   c='cornflowerblue',
    title='Opportunity Set',
    xlabel='Risk',
    ylabel='Mean Excess Return'
)
plt.style.use('default')
plt.axhline(0, color='gray', linewidth=1)
plt.show()





sml = pd.concat([mean_er,beta], axis=1)
sml.columns = ["Actual","Beta"]

sml2 = pd.concat([capm_pred, beta], axis=1)
sml2.columns = ["CAPM", "Beta"]

sml = sml.drop(index='Market Portfolio', errors="ignore")
sml2 = sml2.drop(index='Market Portfolio', errors="ignore")

#outlier
sml = sml.drop(index="SEADRILL (FRA)", errors="ignore")
sml2 = sml2.drop(index="SEADRILL (FRA)", errors="ignore")

print(ret)
print(beta)
print(sml)
print(sml2)

coef = np.polyfit(sml["Beta"], sml["Actual"], 1)
poly1d_fn = np.poly1d(coef)
x_vals = np.linspace(sml["Beta"].min(), sml["Beta"].max(), 100)

coef2 = np.polyfit(sml2["Beta"], sml2["CAPM"], 1)
poly2d_fn = np.poly1d(coef2)
x_vals_2= np.linspace(sml2["Beta"].min(), sml2["Beta"].max(), 100)

plt.style.use('default')
fig=plt.figure()
plt.axhline(0, color='gray', linewidth=1)
plt.scatter(
    sml["Beta"], 
    sml["Actual"], 
    color='blue', 
    s=25,
    label="Actual Returns"
)
plt.scatter(
    sml2["Beta"], 
    sml2["CAPM"], 
    color='gray', 
    s=25,
    label="CAPM Predictions"
)
plt.plot(
    x_vals, 
    poly1d_fn(x_vals), 
    '--', 
    color='blue', 
    linewidth=2,
)
plt.plot(
    x_vals_2,
    poly2d_fn(x_vals_2),
    '--',
    color='gray',
    linewidth=2,
)
plt.xlabel("Beta")
plt.ylabel("Mean Excess Return")
plt.title("Security Market Line")
plt.legend()
plt.show()



#REGRESSION SML
df_sml = pd.concat([mean_er, betas], axis=1)
df_sml.columns = ["mean_return", "beta"]

df_sml = df_sml.drop(index='Market Portfolio', errors="ignore")
df_sml = df_sml.replace([np.inf, -np.inf], np.nan).dropna()


X = sm.add_constant(df_sml["beta"])
y = df_sml["mean_return"]

model = sm.OLS(y, X).fit()


print("\nSML Regression \n")
print(model.summary())


#Size Effect
size=mcap_clear.mean()
deciles = pd.qcut(size, 10, labels=False)

portfolio_returns = pd.DataFrame()

for i in range(10):
    stocks = size[deciles == i].index
    portfolio_returns[i] = returns_clear[stocks].mean(axis=1)

portfolio_size = [
    size[deciles == i].mean() for i in range(10)
]

portfolio_returns_mean = portfolio_returns.mean()

df_test = pd.DataFrame({
    "size": portfolio_size,
    "return": portfolio_returns_mean
}).dropna()

X = sm.add_constant(df_test["size"])
y = df_test["return"]

model = sm.OLS(y, X).fit()

print(model.summary())

#Efficient Frontier
threshold = int(0.7 * len(returns_clear))
returns_assets = returns_clear.dropna(axis=1, thresh=threshold)

                                      
assets=returns_clear.columns
assets=assets[assets!="Market Portfolio"]
returns_assets=returns_clear[assets]

mean_returns=returns_assets.mean()
cov_matrix=returns_assets.cov()

risk_free_rate=rf_m.mean()

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_monthly_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def portfolio_monthly_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *12
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(12)
    return std, returns

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_monthly_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_monthly_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients


def display_ef(mean_returns, cov_matrix, risk_free_rate,returns_assets):
    max_sharpe=max_sharpe_ratio(mean_returns,cov_matrix,risk_free_rate)
    sdp,rp=portfolio_monthly_performance(max_sharpe['x'],mean_returns,cov_matrix)

    max_sharpe_allocation=pd.DataFrame(max_sharpe.x,index=mean_returns.index,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    weights = pd.Series(max_sharpe.x, index=mean_returns.index)
    weights = weights[weights > 0.01]
    weights = weights.sort_values(ascending=False)
    weights = (weights * 100).round(2)
    print("\nMax Sharpe Portfolio):\n")
    print(weights.to_frame(name='Weight (%)'))
    
    sdp, rp = portfolio_monthly_performance(max_sharpe['x'], mean_returns, cov_matrix)
    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_monthly_performance(min_vol['x'], mean_returns, cov_matrix)

    min_vol_allocation = pd.DataFrame(min_vol.x, index=mean_returns.index, columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2) for i in min_vol_allocation.allocation]
    
    
    min_vol_allocation = min_vol_allocation.T

    an_vol = returns_assets.std() * np.sqrt(12)
    an_rt = mean_returns * 12
    
    valid_assets = an_vol[an_vol < 1.5].index
    
    returns_assets = returns_assets[valid_assets]
    mean_returns = mean_returns.loc[valid_assets]
    cov_matrix = cov_matrix.loc[valid_assets, valid_assets]

    an_vol = an_vol.loc[valid_assets]
    an_rt = an_rt.loc[valid_assets]
    
    #Market Portdolio (Rm)
    market_vol = Rm.std() * np.sqrt(12)
    market_ret = Rm.mean() * 12

    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio\n")
    print("Return:", round(rp,4))
    print("Volatility:", round(sdp,4))
    print(max_sharpe_allocation)

    print("-"*80)
    print("Minimum Variance Portfolio\n")
    print("Return:", round(rp_min,4))
    print("Volatility:", round(sdp_min,4))
    print(min_vol_allocation)

    print("-"*80)

    # plot
    fig, ax = plt.subplots(figsize=(10,7))

    ax.scatter(an_vol, an_rt, s=40, color='gray', alpha=0.6, label='Assets')


    ax.scatter(sdp, rp, marker='*', color='r', s=400, label='Max Sharpe')
    ax.scatter(sdp_min, rp_min, marker='*', color='b', s=400, label='Min Vol')
    ax.scatter(market_vol, market_ret, marker='*', color='midnightblue', s=400, label='Market')

    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.legend()

    target = np.linspace(rp_min, max(an_rt),100)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)

    vols = [p['fun'] for p in efficient_portfolios]
    rets = target

    ax.plot(vols, rets, '--', color='black', label='Efficient Frontier')

    cml_x = np.linspace(0, sdp, 100)
    cml_y = risk_free_rate + (rp - risk_free_rate)/sdp * cml_x

    ax.plot(cml_x, cml_y, color='slategray', label='CML')

    ax.legend()

    plt.show()
    
cov_matrix =returns_assets.cov()
risk_free_rate = 0; 
display_ef(mean_returns, cov_matrix, risk_free_rate,returns_assets)

#Re-evaluate CAPM (Using Optimal portfolio)
max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
weights = pd.Series(max_sharpe['x'], index=mean_returns.index)
optimal_returns=returns_assets @ weights
optimal_excess=optimal_returns-rf_m
market_excess=excess_returns['Market Portfolio']

#Regression
df_compare = pd.concat([optimal_returns, Rm,returns['Date']], axis=1).dropna()
df_compare.columns = ["Optimal", "Market","Date"]
df_compare = df_compare.set_index("Date")


cum=(1+df_compare).cumprod()

X=sm.add_constant(df_compare["Market"])
y=df_compare["Optimal"]

model=sm.OLS(y,X).fit()
print(model.summary())

#In-Sample Performance Plot


plt.figure(figsize=(10,6))
plt.plot(df_compare.index,cum["Optimal"], label="Optimal")
plt.plot(df_compare.index,cum["Market"], label="Market")
plt.legend()
plt.title("In-Sample Performance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
          
#SML Regression
cols_to_drop = ["SEADRILL (FRA)"]

returns_clear = returns_clear.drop(columns=cols_to_drop, errors="ignore")
excess_returns = excess_returns.drop(columns=cols_to_drop, errors="ignore")
returns_assets = returns_assets.drop(columns=cols_to_drop, errors="ignore")

betas_optimal={}

for col in excess_returns.columns:
    if col=="Market Portfolio":
        continue
    
    df = pd.concat([excess_returns[col], optimal_excess], axis=1).dropna()
    
    cov = np.cov(df.iloc[:,0], df.iloc[:,1])[0,1]
    var_opt = np.var(df.iloc[:,1])
    
    betas_optimal[col]=cov/var_opt
    
betas_optimal=pd.Series(betas_optimal)

mean_optimal = optimal_excess.mean()

capm_optimal_pred = betas_optimal * mean_optimal

comparison = pd.concat([
    mean_er.drop("Market Portfolio"),
    capm_optimal_pred
], axis=1)

comparison.columns = ["Actual Returns", "CAPM (Optimal)"]

print(comparison)

common_index = betas_optimal.index.intersection(comparison.index)

betas_optimal = betas_optimal.loc[common_index]
comparison = comparison.loc[common_index]

#SML PLOT (Re-Evaluate)
plt.figure(figsize=(8,6))

plt.scatter(betas_optimal, comparison["Actual Returns"], label="Actual")
plt.scatter(betas_optimal, comparison["CAPM (Optimal)"], label="CAPM (Optimal)")


coef = np.polyfit(betas_optimal, comparison["Actual Returns"], 1)
x = np.linspace(betas_optimal.min(), betas_optimal.max(), 100)

plt.plot(x, np.poly1d(coef)(x), '--', label="Actual SML")

coef2 = np.polyfit(betas_optimal, comparison["CAPM (Optimal)"], 1)
plt.plot(x, np.poly1d(coef2)(x), '--', label="CAPM SML")

plt.xlabel("Beta (Optimal Portfolio)")
plt.ylabel("Mean Excess Return")
plt.title("Re-evaluated CAPM (Optimal Portfolio)")
plt.legend()
plt.show()



#For the repot

#Top 5 and 5 low assets (for report)

summary = pd.concat([mean_er, var, std_er], axis=1)
summary.columns = ["Mean Return", "Variance", "Std Dev"]

summary = summary.drop(index='Market Portfolio', errors='ignore')

summary = summary.replace([np.inf, -np.inf], np.nan).dropna()

top5 = summary.sort_values(by="Mean Return", ascending=False).head(5).round(3)

bottom5 = summary.sort_values(by="Mean Return", ascending=True).head(5).round(3)


print("\n Top 5 Stocks:\n")
print(top5)

print("\n Bottom 5 Stocks:\n")
print(bottom5)

#capm results for high and low returns
capm_results = pd.concat([alphas, betas, r_squared,p_beta], axis=1).round(4)
capm_results.columns = ["Alpha", "Beta", "R2","p_value (Beta)"]
capm_results = capm_results.drop(index="Market Portfolio", errors="ignore")
capm_results = capm_results.drop(index="SEADRILL (FRA)", errors="ignore")
capm_results = capm_results.dropna()
print(capm_results.head())

#percentage of positive values in covariance matrix
positive_cov = (cov_matrix > 0).sum().sum()
total_cov = cov_matrix.size

print(positive_cov / total_cov)



