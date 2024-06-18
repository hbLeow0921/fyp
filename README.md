# Final Year Project

##### Introduction:
This project aims to be implementing a multi-agent reinforcement learning (MARL) system for portfolio optimization. The goal is to train multiple agents to manage their investment portfolios by allocating weights to different assets (stocks in this case) in order to maximize returns and minimize risk using the Extended Markowtiz Model Formula. You can try to rerun the system multiple times to get the best allocation.

#####  Library Require:
- Numpy
- Pandas
- yfinance
- pypfopt
- collections
- matplotlib
- scipy

##### Training Data:
The stocks below are used as training data :
- 'MARUTI.NS'
- 'INFY.NS'
- 'TATAMOTORS.NS'
- 'TCS.NS',
- 'GE'
- 'F'
- 'NOK'
-  'BB'
- 'AMZN'
- 'NFLX'

Timestamp:
- Start = '2022-04-01', End = '2023-04-01'
- Start = '2020-01-01' End = '2020-12-31'
- Start = ' 2023-01-01' End = '2023-12-31'



In order to use different stocks you can make changes on the "ticker_list" and start and end date


##### Referencing

Pypfopt Library
- mu = expected_returns.mean_historical_return(prices_df)
- ef = EfficientFrontier(returns, cov_matrix)
- weights = ef.max_sharpe()
- cleaned_weights = ef.clean_weights()
- da = DiscreteAllocation(cleaned_weights, prices, investment_amount)
- allocation, leftover = da.lp_portfolio()

