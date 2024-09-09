import yfinance as yf
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
from black_scholes_model import calculate_BS_OPM
import streamlit as st

# 10 year US risk free treasury rate
RF = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[0] / 100

def plot_options_pricing_comp(chart_data, spot_price, call, window) -> None:
    # Plot the data
    fig, ax = plt.subplots()
    ax.scatter(chart_data.index, chart_data["Option Price"], label="Market Option Price", marker='o')
    ax.scatter(chart_data.index, chart_data["Implied Volatility BS Price"], label="Implied Volatility BS Price", marker='o')
    ax.scatter(chart_data.index, chart_data[f"Historical {window} Rolling Volatility BS Price"], label=f"Historical {window} Rolling Volatility BS Price", marker='o')
    ax.axvline(x=spot_price, color='b', linestyle='--', label="Current Stock Price")
    
    if call:
        ax.fill_betweenx(y=np.linspace(0, max(ax.get_ylim()), 100), x1=min(ax.get_xlim()), x2=spot_price, color='lightblue', alpha=0.5)
    else:
        ax.fill_betweenx(y=np.linspace(0, max(ax.get_ylim()), 100), x1=spot_price, x2=max(ax.get_xlim()), color='lightblue', alpha=.5)
    # Add labels and legend
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Option Price ")
    ax.legend()
    

    # Set custom x-axis labels (indices)
    #ax.set_xticks(chart_call_data.index)
    
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    return

def plot_heatmap(ticker_data, call, stock_price_pct_var, time) -> None:
    if call:
        opts = ticker_data[2].calls
    else:
        opts = ticker_data[2].puts
        
    S_range = np.linspace(ticker_data[1] * (1 - stock_price_pct_var), ticker_data[1] * (1 + stock_price_pct_var), 20)  
    sigma_range = np.linspace(opts.iloc[0]['impliedVolatility'] * .8, opts.iloc[0]['impliedVolatility'] * 1.2, 20)  

    # Create a meshgrid for stock prices and volatilities
    S_grid, sigma_grid = np.meshgrid(S_range, sigma_range)

    # Calculate the option prices for each combination of stock price and volatility
    prices = np.zeros_like(S_grid)
    for i in range(len(S_range)):
        for j in range(len(sigma_range)):
            prices[j, i] = calculate_BS_OPM(S_grid[j, i], opts.iloc[0]['strike'], time, sigma_grid[j, i], RF, call)

    # Create a heatmap using seaborn
    fig, ax = plt.subplots()
    heatmap = sns.heatmap(prices, xticklabels=np.round(S_range, 2), yticklabels=np.round(sigma_range, 2), cmap="YlGnBu", ax=ax)

    # Adding label to the color bar
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label('Option Price', rotation=270, labelpad=20)

    # Labeling the axes
    ax.set_xlabel('Stock Price (S)')
    ax.set_ylabel('Volatility (σ)')
    st.pyplot(fig)

def main():
    st.set_page_config(layout="wide")
    st.write("# Black Scholes Option Pricing Model")
    st.write("**A comparison of Black Scholes OPM against actual option price for European companies listed on NYSE. Santander (SAN) is user-adjustable whereas British Petroleum (BP) is the base case.**")
    st.write("## Stock Price Comparisons")
    stock_price_pct_var = st.sidebar.slider("Stock Price Variation (%)", min_value=10, max_value=90, value=10)
    stock_price_pct_var = stock_price_pct_var / 100
    window = st.sidebar.slider("Stock Window", min_value = 10, max_value = 365, value = 30)

    european_tickers = [
            "SAN",  # Santander 
            "BP", # British Petroleum
        ]

    tickers_with_options = {}
    
    def evaluate():
        stock = yf.Ticker(european_tickers[0])
        if stock.options:
            expiration_dates = stock.options
            expiration_date = st.sidebar.selectbox("Option Expiration Date", expiration_dates)
            date_to_exp = (datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days
            st.sidebar.caption(f"{date_to_exp} days to Expiration")
            end_date = datetime.today()
            start_date = end_date - timedelta(days=window) # 30-day rolling window
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            stock_data = yf.download(european_tickers[0], start=start_date_str, end=end_date_str)
            tickers_with_options[european_tickers[0]] = (expiration_date, stock.info["currentPrice"], stock.option_chain(expiration_date), stock_data)

        stock = yf.Ticker(european_tickers[1])
        if stock.options:
            expiration_dates = stock.options
            end_date = datetime.today()
            start_date = end_date - timedelta(days=window) # 30-day rolling window
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            stock_data = yf.download(european_tickers[1], start=start_date_str, end=end_date_str)
            tickers_with_options[european_tickers[1]] = (expiration_dates[0], stock.info["currentPrice"], stock.option_chain(expiration_dates[0]), stock_data)

        if not tickers_with_options or not len(tickers_with_options) == 2:
            st.write("No options found for stock tickers")
            raise Exception("No options for stock tickers")


        san_col, bp_col = st.columns(2)
        with san_col:
            st.write("## SAN Stock Price")
            stock_data = tickers_with_options[european_tickers[0]][3]
            st.line_chart(stock_data['Adj Close'])
        with bp_col:
            st.write("## BP Stock Price")
            stock_data = tickers_with_options[european_tickers[1]][3]
            st.line_chart(stock_data['Adj Close'])

        st.write("## Option Pricing Comparison")
        st.write("**Now we will compare the Predicted Black Scholes Option Price with the Actual Option Price**")
        san_col, bp_col = st.columns(2)
        # Display available stocks with options
        for ticker in tickers_with_options:
            # calculated historical volatility for comparison
            stock_data['Returns'] = tickers_with_options[ticker][3]['Adj Close'].pct_change()
            daily_std = stock_data['Returns'].dropna().std()
            annualized_volatility = daily_std * np.sqrt(252)
            ##print(f"{ticker}: {tickers_with_options[ticker][2].calls}")
            ##print(f"{ticker}: {tickers_with_options[ticker][2].puts}")
            
            chart_call_data = []
            chart_put_data = []
            for idx in tickers_with_options[ticker][2].calls.index:
                call_opt = tickers_with_options[ticker][2].calls.iloc[idx]
                #print(call_opt)
                #print(tickers_with_options[ticker][0])
                # calculated using implied vol
                iv_call_price = calculate_BS_OPM(tickers_with_options[ticker][1], call_opt['strike'], 
                                                (datetime.strptime(tickers_with_options[ticker][0], '%Y-%m-%d') - datetime.now()).days /252, 
                                                call_opt['impliedVolatility'], 
                                                RF,True)
                # calculate using historical vol
                hv_call_price = calculate_BS_OPM(tickers_with_options[ticker][1], call_opt['strike'], 
                                                (datetime.strptime(tickers_with_options[ticker][0], '%Y-%m-%d') - datetime.now()).days /252, 
                                                annualized_volatility, 
                                                RF,True)
                #print("Implied Volatility Call Price: " + str(iv_call_price))
                #print("Historical Volatility Call Price: " + str(hv_call_price))
                chart_call_data.append((call_opt['strike'],call_opt['lastPrice'], iv_call_price, hv_call_price))
                
            for idx in tickers_with_options[ticker][2].puts.index:
                put_opt = tickers_with_options[ticker][2].puts.iloc[idx]
                #print(put_opt)
                #print(tickers_with_options[ticker][0])
                # calculated using implied vol
                iv_put_price = calculate_BS_OPM(tickers_with_options[ticker][1], put_opt['strike'], 
                                                (datetime.strptime(tickers_with_options[ticker][0], '%Y-%m-%d') - datetime.now()).days /252, 
                                                put_opt['impliedVolatility'], 
                                                RF,False)
                # calculate using historical vol
                hv_put_price = calculate_BS_OPM(tickers_with_options[ticker][1], put_opt['strike'], 
                                                (datetime.strptime(tickers_with_options[ticker][0], '%Y-%m-%d') - datetime.now()).days /252, 
                                                annualized_volatility, 
                                                RF,False)
                #print("Implied Volatility Call Price: " + str(iv_put_price))
                #print("Historical Volatility Call Price: " + str(hv_put_price))
                chart_put_data.append((put_opt['strike'], put_opt['lastPrice'], iv_put_price, hv_put_price))
            
            if ticker == european_tickers[0]:
                with san_col:
                    st.write("## SAN Call Option Pricing")
                    chart_call_data = pd.DataFrame(chart_call_data,columns=["Strike", "Option Price", "Implied Volatility BS Price", f"Historical {window} Rolling Volatility BS Price"])
                    chart_call_data.set_index('Strike', inplace=True)
                
                    plot_options_pricing_comp(chart_call_data, tickers_with_options[ticker][1], True, window)
                    
                    st.write("## SAN Put Option Pricing")
                    chart_put_data = pd.DataFrame(chart_put_data,columns=["Strike", "Option Price", "Implied Volatility BS Price", f"Historical {window} Rolling Volatility BS Price"])
                    chart_put_data.set_index('Strike', inplace=True)
                    
                    plot_options_pricing_comp(chart_put_data, tickers_with_options[ticker][1], False, window)
                    
            else: 
                with bp_col:
                    st.write("## BP Call Option Pricing")
                    chart_call_data = pd.DataFrame(chart_call_data, columns=["Strike", "Option Price", "Implied Volatility BS Price", f"Historical {window} Rolling Volatility BS Price"])
                    chart_call_data.set_index('Strike', inplace=True)
                    plot_options_pricing_comp(chart_call_data, tickers_with_options[ticker][1], True, window)
                    
                    st.write("## BP Put Option Pricing")
                    chart_put_data = pd.DataFrame(chart_put_data, columns=["Strike", "Option Price", "Implied Volatility BS Price", f"Historical {window} Rolling Volatility BS Price"])
                    chart_put_data.set_index('Strike', inplace=True)
                    
                    plot_options_pricing_comp(chart_put_data, tickers_with_options[ticker][1], False, window)
                    
        st.write("## Heatmap of Black Scholes OPM")
        san = tickers_with_options[european_tickers[0]]
        call_options = san[2].calls
        st.write(f"### Call Options Heatmap for SAN @ {call_options.iloc[0]['strike']} Strike")
        plot_heatmap(san,True, stock_price_pct_var, date_to_exp)
        
        put_options = san[2].puts
        try:
            st.write(f"### Put Options Heatmap for SAN @ {put_options.iloc[0]['strike']} Strike")
            plot_heatmap(san,True, stock_price_pct_var, date_to_exp)
        except:
            st.write("### No Put Options")

    evaluate()

    
    
main()


