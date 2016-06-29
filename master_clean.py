import pandas as pd
import numpy as np
from datetime import timedelta


def load_churn():
    df = pd.read_csv('data/churn.csv')


    # Create Churn target column
    df.last_trip_date = pd.to_datetime(df.last_trip_date)
    last = max(df.last_trip_date) - timedelta(days=30)
    df['churn'] = np.where(df.last_trip_date >= last, 0, 1)

    # Create 'Other' category for non-Android/non-iPhone
    df.phone = df.phone.fillna('Other')

    # Dummify Phone column
    phones = ['Android', 'Other', 'iPhone']
    df[phones] = pd.get_dummies(df['phone'])
    # Drop phone column
    df = df.drop('phone', axis=1)

    # Convert signup_date to datetime
    df.signup_date = pd.to_datetime(df.signup_date)


    # Dummify the city column and drop 'city'
    df[['Astapor', "King's Landing", 'Winterfell']] = pd.get_dummies(df['city'])
    df = df.drop(['city'], axis=1)

    # Drop rows that have NaN values
    df = df[df['avg_rating_by_driver'].notnull()]
    df = df[df['avg_rating_of_driver'].notnull()]

    # Convert timestamps from date times into ints (day of year)
    f = lambda x: x.timetuple()[-2]
    df['last_trip_date'] = df['last_trip_date'].map(f)
    df['signup_date'] = df['signup_date'].map(f)

    # Convert boolean to binary
    df['luxury_car_user'] = np.where(df['luxury_car_user'], 1, 0)
    
    # Drop target from feature space
    df = df.drop('last_trip_date', axis=1)

    return df
