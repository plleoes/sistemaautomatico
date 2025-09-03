import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Función para cargar y preparar los datos
def load_data(filename):
    data = pd.read_csv(filename)
    if 'Date' in data.columns and 'Time' in data.columns:
        data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'].astype(str).str.zfill(4), format='%m/%d/%Y %H%M')
    else:
        data['datetime'] = pd.to_datetime(data['DateTime'])
    data.set_index('datetime', inplace=True)
    data.sort_index(inplace=True)
    return data

# Cargar datos
dia_data = load_data('EURJPY.txt')
trin_data = load_data('TRIN_resampled.csv')

leverage=1

# Merge data
data = pd.merge(dia_data, trin_data, left_index=True, right_index=True, how='inner', suffixes=('_DIA', '_TRIN'))
data = data.resample('1min').last().ffill()  # Resample to 1-minute intervals
# Después de la fusión y el remuestreo
print(data.columns)

# Calcular las bandas de Bollinger para NKE
window = 20
data['SMA'] = data['Close_DIA'].rolling(window=window).mean()
data['STD'] = data['Close_DIA'].rolling(window=window).std()
data['Upper_Band'] = data['SMA'] + (data['STD'] * 2)
data['Lower_Band'] = data['SMA'] - (data['STD'] * 2)

# Calcular la distancia porcentual a las bandas
data['Distance_to_Upper'] = (data['Upper_Band'] - data['Close_DIA']) / data['Close_DIA']
data['Distance_to_Lower'] = (data['Close_DIA'] - data['Lower_Band']) / data['Close_DIA']


data['daily_low'] = data['Low'].resample('D').min()
data['daily_high'] = data['High'].resample('D').max()

# Calcular distancias para stop-loss
data['long_distance'] = data['Close_DIA'] - data['daily_low'].shift(1)
data['short_distance'] = data['daily_high'].shift(1) - data['Close_DIA']

# Calcular medias móviles de 20 días para stop-loss
data['long_stop_distance'] = data['long_distance'].rolling(window=20*1440).mean()
data['short_stop_distance'] = data['short_distance'].rolling(window=20*1440).mean()

# Seleccionar el período específico
start_date = '2008-01-01'
end_date = '2025-07-01'
data = data.loc[start_date:end_date]

# Strategy parameters
top_band = 200
low_band = 300
bollinger_threshold = 0.01  # 1% de distancia a las bandas de Bollinger
reverse = False
initial_capital = 80000*5 # Capital inicial en dólares

# Calculate signals
data['pos'] = np.where((data['Close_TRIN'] > top_band) & (data['Distance_to_Upper'] < bollinger_threshold), -1,
                       np.where((data['Close_TRIN'] < low_band) & (data['Distance_to_Lower'] < bollinger_threshold), 1, 0))
data['pos'] = data['pos'].replace(to_replace=0, method='ffill')

if reverse:
    data['pos'] = -data['pos']

# Calculate returns
data['returns'] = data['Close_DIA'].pct_change()
data['strategy_returns'] = data['pos'].shift(1) * data['returns']*5


# Calculate portfolio value
data['portfolio_value'] = initial_capital * (1 + data['strategy_returns']).cumprod()

# Backtest results
total_return = (data['portfolio_value'].iloc[-1] / initial_capital) - 1
annual_return = (1 + total_return) ** (252 / len(data)) - 1
volatility = data['strategy_returns'].std() * np.sqrt(252 * 1440)
sharpe_ratio = np.sqrt(252 * 1440) * data['strategy_returns'].mean() / data['strategy_returns'].std()

print(f"Initial Capital: ${initial_capital:,.2f}")
print(f"Final Portfolio Value: ${data['portfolio_value'].iloc[-1]:,.2f}")
print(f"Total Return: {total_return:.2%}")
print(f"Annual Return: {annual_return:.2%}")
print(f"Volatility (Annualized): {volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(data.index, data['portfolio_value'])
plt.title('Portfolio Value')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.yscale("log")
plt.show()


