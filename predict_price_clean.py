from pmdarima import auto_arima
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import warnings

# ==============================================================================
# CLASS 1: MODEL ARIMA
# ==============================================================================
class PriceForecast:
    def initialize_data(self, date_all, price_all, hotel_name_all, room_name_all, ota_all):
        price_numeric = pd.to_numeric(price_all, errors='coerce')
        date_dt = pd.to_datetime(date_all, errors='coerce')
        data = pd.DataFrame({
            'date': date_dt, 
            'price': price_numeric, 
            'hotel_name': hotel_name_all, 
            'room_name': room_name_all, 
            'ota': ota_all
        })
        self.data = data.dropna(subset=['date', 'price'])
        return self.data

    def read_data(self, hotel_name, room_name, ota):
        df = self.data[(self.data['hotel_name'] == hotel_name) &
                       (self.data['room_name'] == room_name) &
                       (self.data['ota'] == ota)].copy()
        df = df.sort_values(by='date').reset_index(drop=True)
        self.df = df
        return df

    def imputation(self):
        df = self.df.copy()
        df = df.drop_duplicates(subset=['date'])
        df = df.set_index('date')
        df = df.asfreq('D')
        df['price'] = df['price'].ffill().bfill()
        df = df.reset_index()
        df['hotel_name'] = self.df['hotel_name'].iloc[0]
        df['room_name'] = self.df['room_name'].iloc[0]
        df['ota'] = self.df['ota'].iloc[0]
        self.df_interpolated = df
        return df

    def get_forecast_date(self, train, days_pred):
        tanggal_pilihan = train['date'].iloc[-1]
        semua_tanggal = []
        for i in range(days_pred):
            tanggal_sekarang = tanggal_pilihan + timedelta(days=i + 1)
            semua_tanggal.append(tanggal_sekarang.strftime('%Y-%m-%d'))
        return semua_tanggal

    def model_predict(self, days_pred=14):
        df_interpolated = self.imputation()
        train = df_interpolated[-90:].copy()
        train['price_log'] = np.log1p(train['price'].astype(float))

        best_sarima = auto_arima(train['price_log'], seasonal=True, m=7, stepwise=True, trace=False)
        p, d, q = best_sarima.order
        P, D, Q, s = best_sarima.seasonal_order

        model = SARIMAX(train['price_log'], order=(p, d, q), seasonal_order=(P, D, Q, s))
        model_fit = model.fit(disp=False)

        forecast_log = model_fit.forecast(steps=days_pred)
        forecast_cis_log = model_fit.get_forecast(steps=days_pred).conf_int(alpha=0.01)

        forecast = np.expm1(forecast_log)
        lower_price = np.expm1(forecast_cis_log.iloc[:, 0])
        upper_price = np.expm1(forecast_cis_log.iloc[:, 1])

        forecast_date = self.get_forecast_date(train, days_pred)
        df_forecast = pd.DataFrame({
            'date': forecast_date,
            'price': forecast.values,
            'lower price': lower_price.values,
            'upper price': upper_price.values
        })
        df_forecast['date'] = pd.to_datetime(df_forecast['date'])

        self.df_forecast = df_forecast
        self.train = train
        return df_forecast

    def plot_prediction(self):
        forecast = self.df_forecast
        train = self.train
        alpha = 0.1

        plt.figure(figsize=(14, 6))
        past_plot = train.tail(90) 
        
        plt.plot(past_plot['date'], past_plot['price'], label='Historical Data (Last 90 Days)', color='blue')
        plt.plot(forecast['date'], forecast['price'], label='SARIMA Forecast', color='red', linestyle='--', marker='o')
        plt.fill_between(forecast['date'], forecast['lower price'], forecast['upper price'], alpha=0.15,
                         label=f'{int((1 - alpha) * 100)}% Confidence Interval', color='red')

        # PERBAIKAN: Fokuskan sumbu Y agar tidak terdistorsi oleh CI yang terlalu besar
        y_min = min(past_plot['price'].min(), forecast['price'].min())
        y_max = max(past_plot['price'].max(), forecast['price'].max())
        margin = (y_max - y_min) * 0.15 # Padding atas & bawah 15%
        plt.ylim(max(0, y_min - margin), y_max + margin)

        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'SARIMA Forecasting - {self.df["hotel_name"].iloc[0]}')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, days_pred=14):
        df_interpolated = self.imputation()

        train = df_interpolated.iloc[:-days_pred].copy()
        test = df_interpolated.iloc[-days_pred:].copy()

        train_subset = train[-90:].copy()
        train_subset['price_log'] = np.log1p(train_subset['price'].astype(float))

        best_sarima = auto_arima(train_subset['price_log'], seasonal=True, m=7, stepwise=True, trace=False)
        p, d, q = best_sarima.order
        P, D, Q, s = best_sarima.seasonal_order

        model = SARIMAX(train_subset['price_log'], order=(p, d, q), seasonal_order=(P, D, Q, s))
        model_fit = model.fit(disp=False)
        
        forecast_log = model_fit.forecast(steps=len(test))
        forecast_cis_log = model_fit.get_forecast(steps=len(test)).conf_int(alpha=0.01)

        forecast = np.expm1(forecast_log) 
        lower_price = np.expm1(forecast_cis_log.iloc[:, 0])
        upper_price = np.expm1(forecast_cis_log.iloc[:, 1])

        self.eval_train = train_subset
        self.eval_test = test
        self.eval_forecast = forecast.values
        self.eval_lower = lower_price.values
        self.eval_upper = upper_price.values

        mape_list = []
        for i in range(len(test)):
            y_true_val = float(test['price'].iloc[i])
            y_pred_val = float(forecast.iloc[i])
            mape = np.abs((y_true_val - y_pred_val) / y_true_val) if y_true_val != 0 else 0
            mape_list.append(mape)

        self.mape_list = mape_list
        
        df_eval = pd.DataFrame({
            'window_hari': range(1, len(test) + 1),
            'date': test['date'].dt.strftime('%Y-%m-%d').values,
            'actual_price': test['price'].values,
            'forecast_price': forecast.values,
            'mape_percentage': [f"{m:.2%}" for m in mape_list]
        })
        return df_eval

    def plot_evaluation(self):
        train = self.eval_train
        test = self.eval_test
        forecast = self.eval_forecast
        lower_price = self.eval_lower
        upper_price = self.eval_upper
        mape_list = self.mape_list

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        train_plot = train.tail(90)
        ax1.plot(train_plot['date'], train_plot['price'], label='Train Data (Last 90 days)', color='blue', alpha=0.6)
        ax1.plot(test['date'], test['price'], label='Actual Test Data', color='green', marker='o')
        ax1.plot(test['date'], forecast, label='Forecast', color='red', linestyle='--', marker='x')
        ax1.fill_between(test['date'], lower_price, upper_price, color='red', alpha=0.15, label='99% Confidence Interval')

        # PERBAIKAN: Kunci skala sumbu Y agar grafik forecast jelas
        y_min = min(train_plot['price'].min(), test['price'].min(), forecast.min())
        y_max = max(train_plot['price'].max(), test['price'].max(), forecast.max())
        margin = (y_max - y_min) * 0.15
        ax1.set_ylim(max(0, y_min - margin), y_max + margin)

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.set_title(f'Train vs Test vs Forecast (ARIMA)\n{self.df["hotel_name"].iloc[0]}')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)

        windows = range(1, len(mape_list) + 1)
        ax2.plot(windows, mape_list, marker='o', color='purple')
        ax2.set_xlabel('Window Prediction (Hari ke-)')
        ax2.set_ylabel('MAPE')
        ax2.set_title('MAPE tiap Window Prediction (ARIMA)')
        ax2.set_xticks(windows)
        
        for i, val in enumerate(mape_list):
            ax2.text(windows[i], val, f'{val:.1%}', ha='center', va='bottom', fontsize=9)

        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()


# ==============================================================================
# CLASS 2: MODEL RANDOM FOREST
# ==============================================================================
class PriceForecastRevamped:
    def initialize_data(self, date_all, price_all, hotel_name_all, room_name_all, ota_all):
        date_dt = pd.to_datetime(date_all, errors='coerce')
        price_numeric = pd.to_numeric(price_all, errors='coerce')
        self.data = pd.DataFrame({
            'date': date_dt,
            'price': price_numeric,
            'hotel_name': hotel_name_all,
            'room_name': room_name_all,
            'ota': ota_all
        })
        self.data = self.data.dropna(subset=['date', 'price'])
        return self.data

    def read_data(self, hotel_name, room_name, ota):
        df = self.data[
            (self.data['hotel_name'] == hotel_name) & 
            (self.data['room_name'] == room_name) & 
            (self.data['ota'] == ota)
        ].copy()
        df = df.sort_values('date').reset_index(drop=True)
        self.df = df
        return df

    def imputation(self):
        df = self.df.set_index('date')
        df = df.asfreq('D') 
        df['price'] = df['price'].ffill().bfill() 
        df = df.reset_index()
        df['hotel_name'] = self.df['hotel_name'].iloc[0]
        df['room_name'] = self.df['room_name'].iloc[0]
        df['ota'] = self.df['ota'].iloc[0]
        self.df_interpolated = df
        return df

    def extract_features(self, df):
        df = df.copy()
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        return df

    def train_model(self):
        df_clean = self.imputation()
        df_features = self.extract_features(df_clean)
        
        features = ['day_of_week', 'is_weekend', 'day_of_month', 'month', 'days_since_start']
        X = df_features[features]
        y = df_features['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        self.model = model
        self.df_features = df_features
        self.features = features
        self.start_date = df_clean['date'].min()
        return model

    def model_predict(self, days_pred=14):
        if not hasattr(self, 'model'):
            self.train_model()
            
        last_date = self.df_features['date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_pred + 1)]
        df_future = pd.DataFrame({'date': future_dates})
        
        df_future['day_of_week'] = df_future['date'].dt.dayofweek
        df_future['is_weekend'] = df_future['day_of_week'].isin([5, 6]).astype(int)
        df_future['day_of_month'] = df_future['date'].dt.day
        df_future['month'] = df_future['date'].dt.month
        df_future['days_since_start'] = (df_future['date'] - self.start_date).dt.days
        
        X_future = df_future[self.features]
        
        all_tree_predictions = np.stack([tree.predict(X_future.values) for tree in self.model.estimators_])
        mean_predictions = all_tree_predictions.mean(axis=0)
        std_dev = all_tree_predictions.std(axis=0)
        margin = 1.96 * std_dev
        
        df_future['predicted_price'] = mean_predictions
        df_future['lower_price'] = mean_predictions - margin
        df_future['upper_price'] = mean_predictions + margin
        
        self.df_forecast = df_future
        return df_future[['date', 'predicted_price', 'lower_price', 'upper_price']]
        
    def plot_prediction(self):
        past_data = self.df_features
        forecast = self.df_forecast
        
        plt.figure(figsize=(14, 6))
        past_plot = past_data.tail(90) 
        
        plt.plot(past_plot['date'], past_plot['price'], label='Historical Data (Last 90 Days)', color='blue')
        plt.plot(forecast['date'], forecast['predicted_price'], label='RF Forecast', color='red', linestyle='--', marker='o')
        plt.fill_between(forecast['date'], forecast['lower_price'], forecast['upper_price'], alpha=0.15,
                         label='95% Confidence Interval (Est. via Trees StdDev)', color='red')
        
        # PERBAIKAN: Fokuskan sumbu Y
        y_min = min(past_plot['price'].min(), forecast['predicted_price'].min())
        y_max = max(past_plot['price'].max(), forecast['predicted_price'].max())
        margin = (y_max - y_min) * 0.15
        plt.ylim(max(0, y_min - margin), y_max + margin)

        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'ML (Random Forest) Price Forecasting - {self.df["hotel_name"].iloc[0]}')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, days_pred=14):
        df_clean = self.imputation()
        df_features = self.extract_features(df_clean)
        
        train = df_features.iloc[:-days_pred].copy()
        test = df_features.iloc[-days_pred:].copy()
        
        features = ['day_of_week', 'is_weekend', 'day_of_month', 'month', 'days_since_start']
        X_train, y_train = train[features], train['price']
        X_test, y_test = test[features], test['price']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        all_tree_predictions = np.stack([tree.predict(X_test.values) for tree in model.estimators_])
        forecast = all_tree_predictions.mean(axis=0)
        std_dev = all_tree_predictions.std(axis=0)
        margin = 1.96 * std_dev
        
        lower_price = forecast - margin
        upper_price = forecast + margin

        self.eval_train = train
        self.eval_test = test
        self.eval_forecast = forecast
        self.eval_lower = lower_price
        self.eval_upper = upper_price
        
        mape_list = []
        for i in range(len(test)):
            y_true_val = test['price'].iloc[i]
            y_pred_val = forecast[i]
            mape = np.abs((y_true_val - y_pred_val) / y_true_val) if y_true_val != 0 else 0
            mape_list.append(mape)
            
        self.mape_list = mape_list
        
        df_eval = pd.DataFrame({
            'window_hari': range(1, len(test) + 1),
            'date': test['date'].dt.strftime('%Y-%m-%d').values,
            'actual_price': test['price'].values,
            'forecast_price': forecast,
            'mape_percentage': [f"{m:.2%}" for m in mape_list]
        })
        return df_eval

    def plot_evaluation(self):
        train = self.eval_train
        test = self.eval_test
        forecast = self.eval_forecast
        lower_price = self.eval_lower
        upper_price = self.eval_upper
        mape_list = self.mape_list

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        train_plot = train.tail(90)
        ax1.plot(train_plot['date'], train_plot['price'], label='Train Data (Last 90 days)', color='blue', alpha=0.6)
        ax1.plot(test['date'], test['price'], label='Actual Test Data', color='green', marker='o')
        ax1.plot(test['date'], forecast, label='Forecast', color='red', linestyle='--', marker='x')
        ax1.fill_between(test['date'], lower_price, upper_price, color='red', alpha=0.15, label='95% Confidence Interval')

        # PERBAIKAN: Kunci skala sumbu Y
        y_min = min(train_plot['price'].min(), test['price'].min(), forecast.min())
        y_max = max(train_plot['price'].max(), test['price'].max(), forecast.max())
        margin = (y_max - y_min) * 0.15
        ax1.set_ylim(max(0, y_min - margin), y_max + margin)

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.set_title(f'Train vs Test vs Forecast (Random Forest)\n{self.df["hotel_name"].iloc[0]}')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)

        windows = range(1, len(mape_list) + 1)
        ax2.plot(windows, mape_list, marker='o', color='purple')
        ax2.set_xlabel('Window Prediction (Hari ke-)')
        ax2.set_ylabel('MAPE')
        ax2.set_title('MAPE tiap Window Prediction (Random Forest)')
        ax2.set_xticks(windows)
        
        for i, val in enumerate(mape_list):
            ax2.text(windows[i], val, f'{val:.1%}', ha='center', va='bottom', fontsize=9)

        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()
