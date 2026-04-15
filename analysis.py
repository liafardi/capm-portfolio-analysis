import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.optimize as sco


# -----------------------------
# Load Data
# -----------------------------
def load_data(file):
    returns = pd.read_excel(file, sheet_name='Returns')
    mcap = pd.read_excel(file, sheet_name='Market Cap')
    rf = pd.read_excel(file, sheet_name='Risk Free')

    returns['Date'] = pd.to_datetime(returns['Date'])
    mcap['Date'] = pd.to_datetime(mcap['Date'])
    rf['Date'] = pd.to_datetime(rf['Date'])

    return returns, mcap, rf


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_data(returns, mcap, rf):
    returns_clear = returns.drop(columns=['Date']).apply(pd.to_numeric, errors='coerce')
    mcap_clear = mcap.drop(columns=['Date']).apply(pd.to_numeric, errors='coerce')

    mcap_clear = mcap_clear.iloc[1:].reset_index(drop=True)
    returns_clear = returns_clear.reset_index(drop=True)

    returns_clear = returns_clear.loc[:, ~returns_clear.columns.str.contains("EURIBOR")]

    rf_clear = pd.to_numeric(rf.iloc[:, 1], errors='coerce').iloc[1:].reset_index(drop=True)

    return returns_clear, mcap_clear, rf_clear


# -----------------------------
# Market Portfolio
# -----------------------------
def compute_market_portfolio(returns_clear, mcap_clear):
    market_cap = returns_clear.columns[:int(len(returns_clear.columns)/2)]

    weights = mcap_clear[market_cap].div(
        mcap_clear[market_cap].sum(axis=1), axis=0
    )

    Rm = (returns_clear[market_cap] * weights).sum(axis=1)

    return Rm


# -----------------------------
# Excess Returns
# -----------------------------
def compute_excess_returns(returns_clear, rf_clear, Rm):
    rf_m = ((1+(rf_clear/100))**(1/12))-1

    excess_returns = returns_clear.sub(rf_m, axis=0)
    excess_returns['Market Portfolio'] = Rm - rf_m

    return excess_returns, rf_m


# -----------------------------
# CAPM Regression
# -----------------------------
def run_capm(excess_returns):
    alphas = {}
    betas = {}

    for col in excess_returns.columns:
        if col == "Market Portfolio":
            continue

        df = excess_returns[['Market Portfolio', col]].dropna()

        X = sm.add_constant(df['Market Portfolio'])
        y = df[col]

        model = sm.OLS(y, X).fit()

        alphas[col] = model.params[0]
        betas[col] = model.params[1]

    return pd.Series(alphas), pd.Series(betas)


# -----------------------------
# Plot SML
# -----------------------------
def plot_sml(mean_returns, betas):
    df = pd.concat([mean_returns, betas], axis=1)
    df.columns = ["Return", "Beta"]
    df = df.dropna()

    plt.figure()
    plt.scatter(df["Beta"], df["Return"])

    coef = np.polyfit(df["Beta"], df["Return"], 1)
    x = np.linspace(df["Beta"].min(), df["Beta"].max(), 100)

    plt.plot(x, np.poly1d(coef)(x), '--')

    plt.xlabel("Beta")
    plt.ylabel("Mean Excess Return")
    plt.title("Security Market Line")

    plt.savefig("plots/sml.png")
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    file = "your_file_here.xlsx"   # 👈 άλλαξε το

    returns, mcap, rf = load_data(file)
    returns_clear, mcap_clear, rf_clear = preprocess_data(returns, mcap, rf)

    Rm = compute_market_portfolio(returns_clear, mcap_clear)

    excess_returns, rf_m = compute_excess_returns(returns_clear, rf_clear, Rm)

    mean_returns = excess_returns.mean()

    alphas, betas = run_capm(excess_returns)

    plot_sml(mean_returns, betas)


if __name__ == "__main__":
    main()



