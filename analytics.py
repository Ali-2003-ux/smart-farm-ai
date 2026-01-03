import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

def predict_future_yield(historical_df, months_ahead=3):
    """
    Predicts future health and yield trends using Linear Regression.
    
    Args:
        historical_df (pd.DataFrame): Dataframe with columns 'scan_date', 'avg_health', 'total_palms'.
        months_ahead (int): Number of months to forecast.
        
    Returns:
        dict: {'dates': [], 'health': [], 'yield': [], 'trend': str}
    """
    if len(historical_df) < 2:
        return None # Not enough data
        
    # Prepare Data
    df = historical_df.copy()
    df['date_ordinal'] = pd.to_datetime(df['scan_date']).apply(lambda date: date.toordinal())
    
    X = df[['date_ordinal']]
    y_health = df['avg_health']
    
    # Train Model
    model = LinearRegression()
    model.fit(X, y_health)
    
    # Predict Future
    last_date = pd.to_datetime(df['scan_date'].max())
    future_dates = []
    future_ordinals = []
    
    for i in range(1, months_ahead + 1):
        # Approx 30 days per month
        next_date = last_date + pd.Timedelta(days=30 * i)
        future_dates.append(next_date.strftime("%Y-%m-%d"))
        future_ordinals.append([next_date.toordinal()])
        
    future_health = model.predict(future_ordinals)
    
    # Yield Calculation (Simplified assumption: Yield correlates with Health * Count)
    # Let's assume Palm count stays constant for prediction unless we model that too.
    # Using last known count.
    current_count = df['total_palms'].iloc[-1]
    
    # Yield Model: 
    # 100 Health -> 80 kg/tree
    future_yield = []
    for h in future_health:
        # Clamp health 0-100
        h_clamped = max(0, min(100, h))
        # Yield formula: Count * (Health/100) * 80kg
        y_tons = (current_count * (h_clamped / 100.0) * 80) / 1000.0 # Metric Tons
        future_yield.append(round(y_tons, 2))
        
    # Determine Trend
    slope = model.coef_[0]
    if slope > 0.05: trend = "Improving 📈"
    elif slope < -0.05: trend = "Declining 📉"
    else: trend = "Stable ➡️"
    
    return {
        'dates': future_dates,
        'health': [round(h, 1) for h in future_health],
        'yield': future_yield,
        'trend': trend
    }
