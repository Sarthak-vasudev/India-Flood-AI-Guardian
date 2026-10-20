# India Flood AI Guardian 

## Real-Time Flood Risk Prediction and SMS Alert System

India Flood AI Guardian is an MCA final-year project that uses machine learning to classify flood risk across Indian districts. It combines geographic characteristics and live environmental data to display risk levels and support SMS-based alerts.

### Features

- **KNN-based flood risk prediction** — classifies risk as Low, Moderate, High, or Extreme.
- **Live weather integration** — uses OpenWeatherMap and Open-Meteo.
- **Interactive district map** — visualizes predicted flood-risk levels using Folium.
- **SMS alerts** — sends notifications through Fast2SMS for eligible risk levels.
- **Alert logging** — uses SQLite to track alert attempts and help prevent duplicates.
- **Model evaluation** — generates classification metrics and a confusion matrix.

### Technologies Used

- Python
- Scikit-learn
- Pandas and NumPy
- Streamlit
- Folium and Streamlit-Folium
- SQLite
- OpenWeatherMap API
- Open-Meteo API
- Fast2SMS API

### Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard and flood-risk visualization |
| `train_model.py` | Trains and evaluates the KNN classifier |
| `weather_fetcher.py` | Retrieves weather and environmental data |
| `alert_engine.py` | Processes SMS alerts and reminders |
| `db_manager.py` | Manages users and alert logs |
| `districts.csv` | District-level input dataset |
| `requirements.txt` | Python dependencies |

### Setup Instructions

1. Install Python.
2. Clone or download this repository.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a local `.env` file containing your own API credentials:

   ```dotenv
   FAST2SMS_API_KEY=your_fast2sms_api_key
   OWM_API_KEY=your_openweathermap_api_key
   ```

5. Train the model:

   ```bash
   python train_model.py
   ```

6. Run the application:

   ```bash
   python -m streamlit run app.py
   ```

### Important Notes

- Never upload API keys, personal information, or your local database.
- Live data and SMS functionality require valid API credentials.
- The model depends on training data quality and the reliability of external data sources.
- This is an academic prototype, not an official emergency-warning service.

### Author

MCA Final-Year Project
