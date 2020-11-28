# Throws AssertionError

# Ŷ(t+h|t) = Y(t)
def naive_forecast(time_series):
    assert len(time_series) > 0
    return time_series[-1]

# Ŷ(t+h|t) = Y(t+h-T)
# T:= Period of seasonality
def seasonal_naive_forecast(time_series, T):
    assert len(time_series) > T
    return time_series[-T]
