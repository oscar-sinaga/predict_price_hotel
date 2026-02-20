import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.model_selection import train_test_split

class PriceForecast:

    def initialize_data(self, date_all, price_all, hotel_name_all, room_name_all, ota_all):
        """
        Initialize the data for price forecasting.

        Parameters:
        - date_all (list): List of date values.
        - price_all (list): List of price values.
        - hotel_name_all (list): List of hotel names.
        - room_name_all (list): List of room names.
        - ota_all (list): List of OTA (Online Travel Agency) names.

        Returns:
        - data (pd.DataFrame): Initialized DataFrame with date, price, hotel_name, room_name, and ota columns.
        """
        data = pd.DataFrame({'date': date_all, 'price': price_all, 'hotel_name': hotel_name_all, 'room_name': room_name_all, 'ota': ota_all})
        self.data = data
        return data

    def read_data(self, hotel_name, room_name, ota):
        """
        Read specific data for the given hotel, room, and OTA.

        Parameters:
        - hotel_name (str): Hotel name.
        - room_name (str): Room name.
        - ota (str): OTA (Online Travel Agency) name.

        Returns:
        - df (pd.DataFrame): Filtered DataFrame based on the provided criteria.
        """
        df = self.data
        df = df[(df.loc[:,'hotel_name'] == hotel_name) &
                (df.loc[:,'room_name'] == room_name) &
                (df.loc[:,'ota'] == ota)]
        df.loc[:,'date'] = pd.to_datetime(df['date'])
        self.df = df
        return df

    def imputation(self):
        """
        Impute missing data using linear interpolation.

        Returns:
        - df_interpolated (pd.DataFrame): DataFrame with imputed values.
        """
        df = self.df

        # Resetting the index and sorting the DataFrame by date
        df = df.reset_index()
        df = df.sort_values(by='date')

        # Initialize an empty DataFrame for interpolated values
        df_interpolated = pd.DataFrame(columns=['date', 'price'])

        # Iterate through rows in the DataFrame
        for index, row in df.iterrows():
            # Check if it's not the first row
            if index > 0:
                # Calculate the time interval between the current and previous dates
                time_interval = (row['date'] - df.iloc[index - 1]['date']).days

                # If the time interval is more than 1 day, perform linear interpolation
                if time_interval > 1:
                    # Generate new dates within the interval
                    new_dates = [df.iloc[index - 1]['date'] + pd.dateOffset(days=i) for i in range(1, time_interval)]

                    # Get left and right endpoints for interpolation
                    left_endpoint = df.iloc[index - 1]['price']
                    right_endpoint = row['price']

                    # Perform linear interpolation for each new date
                    for date in new_dates:
                        interpolated_price = left_endpoint + ((date - df.iloc[index - 1]['date']).days / time_interval) * (
                                    right_endpoint - left_endpoint)
                        df_interpolated = pd.concat([df_interpolated,
                                                    pd.DataFrame({'date': [date], 'price': [interpolated_price]})],
                                                ignore_index=True)

            # Add the original row to the interpolated DataFrame
            df_interpolated = pd.concat([df_interpolated,
                                        pd.DataFrame({'date': [row['date']], 'price': [row['price']]})],
                                    ignore_index=True)

        # Sort and reset the index of the interpolated DataFrame, and convert 'price' to integer
        df_interpolated = df_interpolated.sort_values(by='date')
        df_interpolated = df_interpolated.reset_index(drop=True)
        df_interpolated.loc[:,'price'] = df_interpolated['price'].astype(int)

        # Save the interpolated DataFrame to the class attribute
        self.df_interpolated = df_interpolated
        return df_interpolated


    def model_predict(self, days_pred=14):
        """
        Train SARIMA model and predict n days after the last day of training data.

        Parameters:
        - days_pred (int): Number of days to predict.

        Returns:
        - df_forecast (pd.DataFrame): DataFrame with forecasted values, including confidence intervals.
        """
        # Perform data imputation using linear interpolation
        df_interpolated = self.imputation()

        # Select the last 90 days for training
        train = df_interpolated[-90:]

        # Convert 'price' column to numeric
        train['price'] = pd.to_numeric(train['price'])

        # Find the best SARIMA model using pmdarima
        best_sarima = auto_arima(train['price'], seasonal=True, m=7, stepwise=True, trace=True)
        p, d, q = best_sarima.order
        P, D, Q, s = best_sarima.seasonal_order

        # Initialize and fit the SARIMAX model
        model = SARIMAX(train['price'], order=(p, d, q), seasonal_order=(P, D, Q, s))
        model_fit = model.fit()

        # Forecast prices for the next 'days_pred' days
        forecast = model_fit.forecast(steps=days_pred)

        # Get confidence intervals for the forecast
        forecast_cis = model_fit.get_forecast(steps=days_pred, alpha=0.01).conf_int()

        # Generate forecast dates
        forecast_date = self.get_forecast_date(train, days_pred)

        # Create a DataFrame with forecasted values and confidence intervals
        df_forecast = pd.DataFrame({'date': forecast_date,
                                    'price': forecast,
                                    'lower price': forecast_cis['lower price'],
                                    'upper price': forecast_cis['upper price']})

        # Convert the 'date' column to datetime format
        df_forecast['date'] = pd.to_datetime(df_forecast['date'])

        # Save the forecast DataFrame and training data to class attributes
        self.df_forecast = df_forecast
        self.train = train

        return df_forecast


    def get_forecast_date(self, train, days_pred):
        """
        Generate dates for forecasting.

        Parameters:
        - train (pd.DataFrame): DataFrame containing training data.
        - days_pred (int): Number of days to predict.

        Returns:
        - semua_tanggal (list): List of forecasted dates.
        """
        tanggal_pilihan = train['date'].iloc[-1]
        semua_tanggal = []

        for i in range(days_pred):
            tanggal_sekarang = tanggal_pilihan + timedelta(days=i + 1)
            semua_tanggal.append(tanggal_sekarang.strftime('%Y-%m-%d'))

        return semua_tanggal

    def plot_prediction(self):
        """
        Plot the predicted data with confidence intervals.
        """
        forecast = self.df_forecast
        train = self.train
        alpha = 0.1

        plt.figure(figsize=(12, 6))
        plt.plot(train['date'][-5:], train['price'][-5:], label='Train Data')
        plt.plot(forecast['date'], forecast['price'], label='SARIMA Forecast', color='red')
        plt.fill_between(forecast['date'], forecast.iloc[:, 2], forecast.iloc[:, 3], alpha=0.1,
                         label=f'{int((1 - alpha) * 100)}% Confidence Interval')

        plt.legend()
        plt.xlabel('date')
        plt.ylabel('price')
        plt.title(f'SARIMA Forecasting {self.df["hotel_name"].iloc[0]} {self.df["room_name"].iloc[0]} ')
        plt.show()

    def evaluate_model(self, days_pred=14):
        """
        Melakukan evaluasi train dan test, serta menghitung MAPE tiap window prediksi.
        """

        df_interpolated = self.imputation()

        # Memisahkan data train dan test secara acak (random split)
        train, test = train_test_split(df_interpolated, test_size=days_pred, random_state=42)

        # Mengurutkan kembali berdasarkan tanggal untuk visualisasi time-series
        train = train.sort_values(by='date').reset_index(drop=True)
        test = test.sort_values(by='date').reset_index(drop=True)

        # Mengambil 90 data terakhir dari train untuk mempercepat training
        train_subset = train[-90:].copy()
        train_subset['price'] = pd.to_numeric(train_subset['price'])
        test['price'] = pd.to_numeric(test['price'])

        # Mencari model SARIMA terbaik
        best_sarima = auto_arima(train_subset['price'], seasonal=True, m=7, stepwise=True, trace=False)
        p, d, q = best_sarima.order
        P, D, Q, s = best_sarima.seasonal_order

        # Fit model dan prediksi sejumlah test data
        model = SARIMAX(train_subset['price'], order=(p, d, q), seasonal_order=(P, D, Q, s))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=len(test))

        # Menyimpan hasil untuk plotting
        self.eval_train = train_subset
        self.eval_test = test
        self.eval_forecast = forecast

        # Menghitung MAPE untuk tiap window prediction (1 hari, 2 hari, dst)
        mape_list = []
        for i in range(len(test)):
            y_true_val = test['price'].iloc[i]
            y_pred_val = forecast.iloc[i]
            
            # Menghitung absolute percentage error per hari
            mape = np.abs((y_true_val - y_pred_val) / y_true_val) if y_true_val != 0 else 0
            mape_list.append(mape)

        self.mape_list = mape_list
        
        # Membuat DataFrame untuk melihat hasil evaluasi dengan mudah
        df_eval = pd.DataFrame({
            'window_hari': range(1, len(test) + 1),
            'date': test['date'].values,
            'actual_price': test['price'].values,
            'forecast_price': forecast.values,
            'mape': mape_list
        })
        
        return df_eval

    def plot_evaluation(self):
        """
        Plot grafik Train vs Test dan MAPE tiap window prediksi.
        """
        train = self.eval_train
        test = self.eval_test
        forecast = self.eval_forecast
        mape_list = self.mape_list

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        # Plot 1: Train vs Test vs Forecast
        ax1.plot(train['date'], train['price'], label='Train Data', color='blue', alpha=0.6)
        ax1.plot(test['date'], test['price'], label='Actual Test Data', color='green', marker='o')
        ax1.plot(test['date'], forecast, label='Forecast', color='red', linestyle='--', marker='x')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.set_title(f'Train vs Test vs Forecast\n{self.df["hotel_name"].iloc[0]}')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)

        # Plot 2: MAPE per Window Prediction
        windows = range(1, len(mape_list) + 1)
        ax2.plot(windows, mape_list, marker='o', color='purple')
        ax2.set_xlabel('Window Prediction (Hari ke-)')
        ax2.set_ylabel('MAPE')
        ax2.set_title('MAPE tiap Window Prediction')
        ax2.set_xticks(windows)
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

