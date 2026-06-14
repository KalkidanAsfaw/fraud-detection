import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess import ip_to_int, add_time_features, add_velocity_features


def test_ip_to_int_valid():
    result = ip_to_int("192.168.1.1")
    assert isinstance(result, (int, float))
    assert result == 3232235777


def test_ip_to_int_float():
    # Source data stores IPs as floats — must cast to int, not return NaN.
    result = ip_to_int(732758368.79972)
    assert result == 732758368


def test_ip_to_int_invalid():
    result = ip_to_int("not_an_ip")
    assert np.isnan(result)


def test_add_time_features():
    df = pd.DataFrame({
        'signup_time': ['2021-01-01 10:00:00'],
        'purchase_time': ['2021-01-01 12:30:00'],
    })
    result = add_time_features(df)
    assert 'time_since_signup' in result.columns
    assert 'hour_of_day' in result.columns
    assert 'day_of_week' in result.columns
    assert abs(result['time_since_signup'].iloc[0] - 2.5) < 0.01
    assert result['hour_of_day'].iloc[0] == 12


def test_add_velocity_features():
    df = pd.DataFrame({
        'user_id': [1, 1, 2],
        'purchase_time': ['2021-01-01', '2021-01-02', '2021-01-01'],
    })
    result = add_velocity_features(df)
    assert 'user_tx_count' in result.columns
    user1_count = result[result['user_id'] == 1]['user_tx_count'].iloc[0]
    user2_count = result[result['user_id'] == 2]['user_tx_count'].iloc[0]
    assert user1_count == 2
    assert user2_count == 1
