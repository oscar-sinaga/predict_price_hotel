import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from datetime import timedelta

class PriceForecastRevamped:
    def initialize_data(self, date_all, price_all, hotel_name_all, room_name_all, ota_all):
        self.data = pd.DataFrame({
            'date': pd.to_datetime(date_all),
            'price': price_all,
            'hotel_name': hotel_name_all,
            'room_name': room_name_all,
            'ota': ota_all
        })
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
        # ROMBAK 1: Menggunakan asfreq dan Forward Fill
        df = self.df.set_index('date')
        df = df.asfreq('D') # Memastikan setiap harinya ada di dataframe
        
        # Harga bertahan stagnan sampai ada perubahan (ffill)
        df['price'] = df['price'].ffill().bfill() 
        
        df = df.reset_index()
        # Mengembalikan metadata teks
        df['hotel_name'] = self.df['hotel_name'].iloc[0]
        df['room_name'] = self.df['room_name'].iloc[0]
        df['ota'] = self.df['ota'].iloc[0]
        
        self.df_interpolated = df
        return df

    def extract_features(self, df):
        df = df.copy()
        # ROMBAK 4: Feature Engineering dari struktur tanggal
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        
        # Fitur tren jangka panjang dari hari pertama mulai
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        return df

    def train_model(self):
        df_clean = self.imputation()
        df_features = self.extract_features(df_clean) # ROMBAK 2: Pakai seluruh data
        
        features = ['day_of_week', 'is_weekend', 'day_of_month', 'month', 'days_since_start']
        X = df_features[features]
        y = df_features['price']
        
        # ROMBAK 5: Data Splitting dilakukan secara acak (Random Split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # ROMBAK 3: Menggunakan model Machine Learning Tree-based
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        print(f"Evaluasi Model - MAPE pada Data Test: {mape:.2%}")
        
        self.model = model
        self.df_features = df_features
        self.features = features
        self.start_date = df_clean['date'].min()
        return model

    def model_predict(self, days_pred=14):
        # Memastikan model dilatih lebih dahulu
        if not hasattr(self, 'model'):
            self.train_model()
            
        last_date = self.df_features['date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_pred + 1)]
        df_future = pd.DataFrame({'date': future_dates})
        
        # Membuat fitur yang sama untuk input prediksi masa depan
        df_future['day_of_week'] = df_future['date'].dt.dayofweek
        df_future['is_weekend'] = df_future['day_of_week'].isin([5, 6]).astype(int)
        df_future['day_of_month'] = df_future['date'].dt.day
        df_future['month'] = df_future['date'].dt.month
        df_future['days_since_start'] = (df_future['date'] - self.start_date).dt.days
        
        X_future = df_future[self.features]
        predictions = self.model.predict(X_future)
        df_future['predicted_price'] = predictions
        
        self.df_forecast = df_future
        return df_future[['date', 'predicted_price']]
        
    def plot_prediction(self):
        past_data = self.df_features
        forecast = self.df_forecast
        
        plt.figure(figsize=(14, 6))
        # Hanya plot 90 hari ke belakang agar garis prediksi terlihat jelas
        past_plot = past_data.tail(90) 
        
        plt.plot(past_plot['date'], past_plot['price'], label='Historical Data (Last 90 Days)', color='blue')
        plt.plot(forecast['date'], forecast['predicted_price'], label='Forecast', color='red', linestyle='--', marker='o')
        
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'ML Price Forecasting - {self.df["hotel_name"].iloc[0]}')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
