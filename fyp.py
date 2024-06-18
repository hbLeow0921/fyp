import numpy as np
from scipy.optimize import minimize
import pandas as pd
import datetime
import yfinance as yf
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from collections import defaultdict
import matplotlib.pyplot as plt

# Function to calculate stock limits
def calculate_discrete_allocation(assets, returns, cov_matrix, investment_amount, prices_df, t):
    prices = prices_df.iloc[0]  # Use the prices from the t-th date
    
    # Create an EfficientFrontier instance
    ef = EfficientFrontier(returns, cov_matrix)

    # Compute the optimal portfolio
    weights = ef.max_sharpe()  
    
    # Clean the weights
    cleaned_weights = ef.clean_weights()


    # Create a DiscreteAllocation instance
    da = DiscreteAllocation(cleaned_weights, prices, investment_amount)

    # Compute the discrete allocation
    allocation, leftover = da.lp_portfolio()
    
    return allocation, leftover

def calculate_target_allocation(assets, returns, cov_matrix, investment_amount, prices_df, t):
    target_allocation, _ = calculate_discrete_allocation(assets, returns, cov_matrix, investment_amount, prices_df, t)
    return target_allocation




agent_q_tables = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

class PortfolioEnvironment:
    def __init__(self, assets, returns, cov_matrix):
        self.assets = assets
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.agents = []
        self.time_step = 0
        self.stock_investments = {asset: 0 for asset in assets}  # Initialize with asset tickers as keys

    def add_agent(self, agent):
        self.agents.append(agent)

    def reset(self):
        self.time_step = 0
        for agent in self.agents:
            agent.reset()

    def step(self, prices_df):
        states = []
        actions = []
        rewards = []
        next_states = [None] * len(self.agents)

        try:
            prices = prices_df.iloc[-1]
        except:
            print(f"Error: Unable to download data for tickers {self.assets}")
            return [], [], []

        t = self.time_step  # Get the current time step

        # Calculate the target allocation for the current time step
        for agent in self.agents:
            target_allocation = calculate_target_allocation(self.assets, self.returns, self.cov_matrix, agent.initial_investment, prices_df, t)
            agent.target_allocation = target_allocation

        # Set the allocation for each agent
        for agent in self.agents:
            agent.set_allocation(prices_df, t)
            

        for i, agent in enumerate(self.agents):
            state = agent.get_state()
            action = agent.choose_action(state)
            initial_allocation = agent.allocation.copy()  # Save the initial allocation
            

            # Take the action
            agent.take_action(action, agent.target_allocation)
            

            # Check if the investment in any stock exceeds its limit
            violating_stock = None
            for asset in self.assets:
                self.stock_investments[asset] += agent.allocation.get(asset, 0) - initial_allocation.get(asset, 0)

            if violating_stock is not None:
                # Rollback the action for this agent
                agent.allocation = initial_allocation
                self.stock_investments = {asset: self.stock_investments[asset] - (agent.allocation.get(asset, 0) - initial_allocation.get(asset, 0)) for asset in self.assets}
                reward = agent.calculate_reward(agent.target_allocation)
            else:
                reward = agent.calculate_reward(agent.target_allocation)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # Print the weight allocation for each agent after taking an action
            allocations = {asset: agent.allocation.get(asset, 0) for asset in self.assets}


            # Convert the state to an integer
            state_int = agent.discretize_state(state)

            # Update the Q-tables for each agent
            for asset in self.assets:
                action_int = agent.actions.index(action[asset])
                old_value = agent.q_table[state_int][action_int]
                next_max = np.max(agent.q_table[state_int])
                new_value = (1 - agent.alpha) * old_value + agent.alpha * (reward + agent.gamma * next_max)
                agent.q_table[state_int][action_int] = new_value

            # Store the updated Q-table
            agent_q_tables[agent.id][t] = agent.q_table.copy()

            # Get the next state
            next_states[i] = agent.get_state()

        self.time_step += 1
        return states, actions, rewards

class Agent:
    def __init__(self, id, assets, returns, cov_matrix, initial_investment, epsilon=0.1, alpha=10, gamma=0.6):
        self.id = id
        self.assets = assets
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.initial_investment = initial_investment
        self.allocation = {asset: 0 for asset in assets}  # Initialize allocation with zeros
        self.weight_allocation = {}
        self.target_allocation = {}
        self.leftover = 0
        self.epsilon = epsilon
        self.actions = ['buy', 'sell', 'hold']  # Add 'hold' to the list of actions
        self.alpha = alpha
        self.gamma = gamma
        
        # Define the number of bins for discretization
        num_bins = 10
        
        # Calculate the number of possible states
        num_states = num_bins ** len(assets)
        
        # Initialize q_table with the correct size
        self.q_table = np.zeros((num_states, len(self.actions)))

    def calculate_distance(self, target_allocation):
        distance = {}
        for asset in self.assets:
            distance[asset] = abs(self.allocation.get(asset, 0) - target_allocation.get(asset, 0))
        return distance

    def calculate_risk(self):
        assets = [asset for asset in self.assets if asset in self.allocation and asset in self.cov_matrix]
        weights = np.array([self.allocation[asset] for asset in assets])
        cov_matrix = np.array([[self.cov_matrix[asset1][asset2] for asset2 in assets] for asset1 in assets])
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def calculate_reward(self, target_allocation):
        # Calculate the expected return 
        expected_return = sum(self.returns[asset] * self.allocation[asset] for asset in self.assets if asset in self.allocation and asset in self.returns)
        
        # Calculate the risk
        risk = self.calculate_risk()
        
        # Calculate the distance from the target allocation
        distance = sum(abs(self.allocation[asset] - target_allocation.get(asset, 0)) for asset in self.assets)
        
        # The reward is the expected return minus a penalty for risk and distance from the target allocation
        reward = expected_return - risk - distance 
        
        return reward


    def set_allocation(self, prices_df, t):
        prices = {asset: prices_df[asset].iloc[t] for asset in self.assets}
        target_allocation, _ = calculate_discrete_allocation(self.assets, self.returns, self.cov_matrix, self.initial_investment, prices_df, t)
        max_iterations = 1000  # Maximum number of iterations
        min_distances = {asset: float('inf') for asset in self.assets}  # Initialize minimum distances to infinity
        best_allocation = {asset: 0 for asset in self.assets}  # Initialize best allocation to zeros

        if t == 0:
            # For the initial time step (t=0), use the agent's initial investment
            investment_amount = self.initial_investment
        else:
            # For subsequent time steps, use the sum of the previous weight allocation
            investment_amount = sum(self.weight_allocation.values())

        for _ in range(max_iterations):
            num_assets = len(self.assets)
            random_allocation = np.random.dirichlet(np.ones(num_assets), size=1)[0]
            allocation = {asset: (allocation * investment_amount) / prices[asset] for asset, allocation in zip(self.assets, random_allocation)}
            distance = self.calculate_distance(target_allocation)

            for asset in self.assets:
                if distance[asset] < min_distances[asset]:
                    min_distances[asset] = distance[asset]
                    best_allocation[asset] = allocation[asset]

        self.allocation = best_allocation
        self.weight_allocation = {asset: allocation * prices[asset] for asset, allocation in self.allocation.items()}


    def set_max_allocation(self, prices_df, t):

        # Calculate the expected returns
        mu = expected_returns.mean_historical_return(prices_df)

        # Get the latest prices
        prices = prices_df.iloc[t]

        # Sort the assets by their expected return in descending order
        sorted_assets = mu.sort_values(ascending=False)

        # Allocate 60% of the weight to the asset with the highest expected return
        high_return_asset = sorted_assets.index[0]
        high_return_weight = 0.60 * self.initial_investment

        # Divide the remaining 25% of the weight evenly among the remaining assets
        remaining_assets = sorted_assets.index[1:]
        remaining_weight_per_asset = (0.40 / len(remaining_assets)) * self.initial_investment

        # Calculate the weight for each asset
        weights = {asset: (high_return_weight if asset == high_return_asset else remaining_weight_per_asset)/ prices[asset] for asset in sorted_assets.index}

        # Store the weights in the agent's weight_allocation attribute
        self.allocation = weights
    

    def avg_allocation(self, prices_df, t):
    # Get the latest prices
        prices = {asset: prices_df[asset].iloc[t] for asset in self.assets}
        
        # Calculate the amount to invest in each asset
        investment_per_asset = self.initial_investment / len(self.assets)
        
        # Calculate the weight for each asset based on the latest prices
        
        weight = {asset: investment_per_asset / prices[asset] for asset in self.assets}
        # Store the weights in the agent's weight_allocation attribute
        self.allocation = weight
        

    def choose_action(self, state):
    # Discretize the state
        discretized_state = self.discretize_state(state)

        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            # Choose a random action dictionary
            action = {asset: np.random.choice(['buy', 'sell', 'hold']) for asset in self.assets}
        else:
            # Choose the action with the highest expected reward
            best_action_index = np.argmax(self.q_table[discretized_state])
            action = {asset: self.actions[best_action_index] for asset in self.assets}

        return action

    def discretize_state(self, state):
        
        num_bins = 10  # Increase the number of bins for finer granularity
        
        discretized_state = []
        
        for asset in self.assets:
            weight = state.get(asset, 0.0)  # Handle zero allocations
            bin_index = int(weight * num_bins)  # Discretize weight into bin indices
            discretized_state.append(bin_index)
        
        # Convert the tuple of bin indices to a single integer
        discretized_state_int = sum(bin_idx * (num_bins ** i) for i, bin_idx in enumerate(discretized_state))
        
        # Ensure the discretized state is within the valid range
        discretized_state_int = discretized_state_int % self.q_table.shape[0]
        
        return discretized_state_int
    



    def take_action(self, action, target_allocations):
        # Calculate the total weight of 'hold' assets
        hold_weight = sum(self.allocation[asset] for asset in self.assets if action.get(asset, 'hold') == 'hold')

        # Calculate the total weight of assets to buy or sell
        total_trade_weight = 0
        for asset in self.assets:
            if asset in target_allocations:  # Check if the asset is in target_allocations
                total_trade_weight += target_allocations[asset]
            else:
                target_allocations[asset] = 0
            asset_action = action.get(asset, 'hold')
            if asset_action == 'buy':
                total_trade_weight += target_allocations[asset]
            elif asset_action == 'sell':
                total_trade_weight += self.allocation[asset]

        # Create a dictionary to store the trade weights
        trade_weights = {asset: 0 for asset in self.assets}

        # Calculate the trade weights based on the action
        for asset in self.assets:
            asset_action = action.get(asset, 'hold')
            if asset_action == 'buy':
                trade_weights[asset] = target_allocations[asset] / total_trade_weight
            elif asset_action == 'sell':
                trade_weights[asset] = -self.allocation[asset] / total_trade_weight

        
        # Update the allocation by adding the trade weights to the original allocation
        new_allocation = {asset: self.allocation[asset] + trade_weights[asset] for asset in self.assets}
        
        # Normalize the new allocation to sum up to 1
        # total_allocation = sum(new_allocation.values())
        # new_allocation = {asset: weight / total_allocation for asset, weight in new_allocation.items()}
        


        # Allocate any remaining funds to the 'hold' assets
        remaining_weight = 1 - sum(new_allocation.values())
        if remaining_weight > 0:
            for asset in self.assets:
                if action.get(asset, 'hold') == 'hold':
                    new_allocation[asset] += remaining_weight * (self.allocation[asset] / hold_weight)

        # Ensure allocation remains within bounds
        # for asset in self.assets:
        #     new_allocation[asset] = max(min(new_allocation[asset], 1), 0)

        # Update the agent's allocation with the new allocation
        self.allocation = new_allocation

    def update_q_table(self, states, actions, rewards):
        for i in range(len(states)):
            state_int = self.discretize_state(states[i])
            action_int = [self.actions.index(actions[i][asset]) for asset in self.assets]
            
            old_values = [self.q_table[state_int][action] for action in action_int]
            next_max = np.max(self.q_table[state_int])

            new_values = [(1 - self.alpha) * old_value + self.alpha * (rewards[i] + self.gamma * next_max) for old_value in old_values]
            
            for asset, action, new_value in zip(self.assets, action_int, new_values):
                scaled_value = int(round(new_value * self.q_value_scale))
                self.q_table[state_int][action] = scaled_value

    def get_state(self):
        return self.allocation

    def reset(self):
        self.weights = np.array([1/len(self.assets)] * len(self.assets))
        self.allocation = {asset: 1/len(self.assets) for asset in self.assets}

def calculate_roi(agent, prices, initial_allocation, initial_prices):
    # Ensure all assets are in the allocation and prices
    missing_assets = set(agent.assets) - set(agent.allocation.keys()) - set(prices.keys())

    for asset in missing_assets:
        agent.allocation[asset] = 0
        prices[asset] = initial_prices.get(asset, 0)

    # Calculate the current portfolio value
    current_portfolio_value = sum(agent.allocation[asset] * prices[asset] for asset in agent.assets if asset in agent.allocation and asset in prices)

    # Calculate the initial portfolio value
    initial_portfolio_value = sum(initial_allocation[asset] * initial_prices[asset] for asset in agent.assets if asset in initial_allocation and asset in initial_prices)

    # Calculate the ROI
    if initial_portfolio_value > 0:
        roi = ((current_portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100
        roi = round(roi, 2)
    else:
        roi = 0.0  # Avoid division by zero

    return roi




ticker_list = ['INFY.NS','TCS.NS','TATAMOTORS.NS','MARUTI.NS']
study_prices_df = yf.download(ticker_list, start = '2022-04-01', end = '2023-04-01')['Adj Close']
prices_df = study_prices_df.resample('W').last()
print("Initial Prices",get_latest_prices(prices_df))


check_df = yf.download(ticker_list, start = '2023-04-01', end = '2023-04-08')['Adj Close']
prices_df_2 = check_df.resample('W').last()

prices_check = get_latest_prices(prices_df_2)
print("Current Prices", prices_check)


# Calculate expected returns and covariance matrix that will be used for target allocation
mu = expected_returns.mean_historical_return(prices_df)
# print("Expected Returns:" , mu)
S = risk_models.sample_cov(prices_df)
# print("Covariance Matrix:" , S)
returns = prices_df.pct_change()
volatility = returns.std()


# For prices checking
# print("Initial Prices",prices_df.iloc[0])
# print("Current Price", prices_check)

# Initialize an empty list to store agents
agents = []

# Initialize agents
agent1 = Agent(id=1, assets=ticker_list, returns=mu, cov_matrix=S, initial_investment=100000)
agent2 = Agent(id=2, assets=ticker_list, returns=mu, cov_matrix=S, initial_investment=150000)
agent3 = Agent(id=3, assets=ticker_list, returns=mu, cov_matrix=S, initial_investment=200000)

# Add agents to the list
agents.append(agent1)
agents.append(agent2)
agents.append(agent3)

# Calculate the total portfolio value
total_portfolio_value = sum([agent.initial_investment for agent in agents])

# Create the PortfolioEnvironment and add agents
env = PortfolioEnvironment(ticker_list, mu, S)
for agent in agents:
    env.add_agent(agent)

weekly_prices = prices_df
start_prices = {ticker: weekly_prices[ticker].iloc[0] for ticker in ticker_list}

# Set the allocation for each agent
for agent in agents:
    agent.set_allocation(prices_df, 0)
    


for agent in agents:
    agent.target_allocation = calculate_target_allocation(agent.assets, agent.returns, agent.cov_matrix, agent.initial_investment, prices_df, 0)
    print(f"Agent {agent.id}, Target Allocation: {agent.target_allocation}")

# Check if the allocations were set correctly
for agent in agents:
    if not agent.allocation:
        print(f"Error: Unable to set allocation for agent {agent.id}")

# Run the MARL system for a certain number of time steps

prev_roi = {agent.id: [] for agent in env.agents}
num_time_steps = len(weekly_prices)
prev_sum_allocation = {agent.id: 0 for agent in env.agents}
for t in range(num_time_steps):
    prices = {ticker: weekly_prices[ticker].iloc[t] for ticker in ticker_list}
    

    states, actions, rewards = env.step(prices_df)
    
    # Check if there are any agents
    if not env.agents:
        print("No agents found.")
        break

    # Print state, action, and reward for each agent
    for i, agent in enumerate(env.agents):
        

        # Use to calculate ROI
        prev_prices = {ticker: weekly_prices[ticker].iloc[num_time_steps-1] for ticker in ticker_list}
        prev_allocation = agent.allocation.copy()
        prev_leftover = agent.leftover
        
        # Convert the state to an integer
        state_int = agent.discretize_state(states[i])
        print(f"State Int: {state_int}")
        
        
        current_portfolio_value = sum(agent.allocation[asset] * prices[asset] for asset in agent.assets)
        prev_portfolio_value = sum(prev_allocation[asset] * prev_prices[asset] for asset in agent.assets)
        
        current_portfolio_value = round(current_portfolio_value, 2)
        prev_portfolio_value = round(prev_portfolio_value, 2)

        q_values = [agent.q_table[state_int][agent.actions.index(actions[i][asset])] for asset in agent.assets]
        rounded_allocation = {k: round(v, 2) for k, v in agent.weight_allocation.items()}

        print(f"Agent {agent.id}, Week: {t+1}, State: {state_int}, Actions: {actions[i]}, Q-values: {q_values}, Reward: {rewards[i]}")
        print(f"Agent {agent.id}, Allocation: {agent.allocation}, Weight Allocation: {rounded_allocation}, Leftover: {agent.leftover}")
        print(f"Current Portfolio Value: {current_portfolio_value}, Previous Portfolio Value: {prev_portfolio_value}\n, Starting Portfolio Value: {agent.initial_investment}\n")
    
        prev_prices = prices  # Store the current prices for the next timestamp
        prev_allocation = agent.allocation.copy()  # Store the current allocation for the next timestamp
        prev_leftover = agent.leftover  # Store the current leftover for the next timestamp
        distance = sum(abs(agent.allocation[asset] - agent.target_allocation.get(asset, 0)) for asset in agent.assets)

        # Convert the state to an integer
        state_int = agent.discretize_state(states[i])

        # Update the Q-tables for each agent
        for asset in agent.assets:
            action_index = agent.actions.index(actions[i][asset])  # Convert the action to an index
            old_value = agent.q_table[state_int][action_index]
            next_max = np.max(agent.q_table[state_int])

            new_value = (1 - agent.alpha) * old_value + agent.alpha * (rewards[i] + agent.gamma * next_max)
            agent.q_table[state_int][action_index] = new_value  # Update the Q-table using the action index


        if distance < 0.1:
            agent.set_allocation(prices_df, t)
        
        # Calculate the ROI for each agent's portfolio
        initial_allocation = agent.allocation  # Assuming the allocation doesn't change
        initial_prices = {ticker: weekly_prices[ticker].iloc[0] for ticker in ticker_list}  # Prices at the first step
        previous_prices = {ticker: weekly_prices[ticker].iloc[num_time_steps-1] for ticker in ticker_list}
        current_portfolio_value = sum(agent.allocation[asset] * prices_check[asset] for asset in agent.assets)
        
        
        

        # Ensure all assets are in the allocation and prices
        missing_assets = set(agent.assets) - set(agent.allocation.keys()) - set(prices.keys())
        for asset in missing_assets:
            agent.allocation[asset] = 0
            prices[asset] = initial_prices.get(asset, 0)

        if t > 0:
            roi = round(((current_portfolio_value - agent.initial_investment) / agent.initial_investment) * 100,2)
            previous_step_roi = calculate_roi(agent, prices, initial_allocation, previous_prices)
            print(f"Agent {agent.id}, Current Check Prices: {current_portfolio_value}")
            print(f"Agent {agent.id}, ROI: {roi} %")
            print(f"Agent {agent.id}, Previous ROI: {previous_step_roi} %\n")
            prev_roi[agent.id].append(roi)

            
        
    
    if t == num_time_steps - 1:
        for agent_id, roi in prev_roi.items():
            plt.plot(roi, label=f'Agent {agent_id}')
        plt.xlabel('Timestamp')
        plt.ylabel('ROI')
        plt.legend()
        plt.show()


avg_roi = {agent.id: [] for agent in env.agents}
max_roi = {agent.id: [] for agent in env.agents}


for time_step in range(num_time_steps):
    current_prices = prices_check
    #Average allocation 
    for agent in agents:
        agent.avg_allocation(prices_df, 0)
        current_portfolio_value = round(sum(agent.allocation[asset] * prices_check[asset] for asset in agent.assets),2)
        prev_portfolio_value = sum(agent.allocation[asset] * initial_prices[asset] for asset in agent.assets)
        roi = ((current_portfolio_value - prev_portfolio_value) / prev_portfolio_value) * 100
        if time_step == num_time_steps - 1:
            print(f"Agent {agent.id}, Allocation: {agent.allocation}")
            print(f"Agent {agent.id}, Final ROI: {roi} %")
        avg_roi[agent.id].append(roi)

    #Max allocation:
    for agent in agents:
        agent.set_max_allocation(prices_df, 0)
        current_portfolio_value = round(sum(agent.allocation[asset] * prices_check[asset] for asset in agent.assets),2)
        prev_portfolio_value = sum(agent.allocation[asset] * initial_prices[asset] for asset in agent.assets)
        roi = ((current_portfolio_value - prev_portfolio_value) / prev_portfolio_value) * 100
        if time_step == num_time_steps - 1:
            print(f"Agent {agent.id}, Allocation: {agent.allocation}")
            print(f"Agent {agent.id}, Final ROI: {roi} %")
        max_roi[agent.id].append(roi)



for agent_id in prev_roi.keys():
    plt.figure()  # Create a new figure
    plt.plot(prev_roi[agent_id], label='Markowitz ROI')
    plt.plot(avg_roi[agent_id], label='Avg ROI')
    plt.plot(max_roi[agent_id], label='Max ROI')
    plt.xlabel('Timestamp')
    plt.ylabel('ROI')
    plt.title(f'Agent {agent_id}')  # Set the title to the agent's ID
    plt.legend()
    plt.show()
# Reset the environment and agents
    

env.reset()





