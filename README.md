\# Predictive Maintenance System



An end-to-end AI-powered predictive maintenance application that estimates Remaining Useful Life (RUL) and detects unusual sensor behavior in turbofan engines using NASA C-MAPSS FD001 data.



\## Project Overview



Unexpected equipment failures can cause downtime and expensive maintenance. This project explores how machine learning can help monitor engine condition and estimate how many operating cycles may remain before failure.



The system combines deep learning, anomaly detection, a REST API, a PostgreSQL database, and an interactive Streamlit dashboard.



\## Key Features



\- \*\*Remaining Useful Life Prediction:\*\* LSTM model estimates remaining operating cycles using the latest 50 sensor readings.

\- \*\*Anomaly Detection:\*\* Isolation Forest identifies unusual sensor observations.

\- \*\*REST API:\*\* FastAPI provides prediction and anomaly detection endpoints.

\- \*\*Database Integration:\*\* PostgreSQL stores engine readings and prediction history.

\- \*\*Interactive Dashboard:\*\* Streamlit displays engine information, sensor trends, predictions, anomaly status, and historical logs.

\- \*\*Model Inference:\*\* Trained models are loaded locally for prediction.

## Project Screenshots

### Main Dashboard
![Main Dashboard](docs/screenshots/main-dashboard.png)

### Sensor Analytics
![Sensor Analytics](docs/screenshots/sensor-analytics.png)

### Prediction Results
![Prediction Results](docs/screenshots/prediction-results.png)

### Database History
![Database History](docs/screenshots/database-history.png)



\## Technology Stack



| Area | Technologies |

|---|---|

| Programming | Python |

| Machine Learning | PyTorch, Scikit-learn |

| Deep Learning | LSTM |

| Anomaly Detection | Isolation Forest |

| Backend | FastAPI, Uvicorn |

| Frontend | Streamlit, Plotly |

| Database | PostgreSQL |

| Data Processing | Pandas, NumPy |

| Model Storage | Joblib, PyTorch |



\## Dataset



\*\*NASA C-MAPSS FD001\*\*



The project uses the NASA Commercial Modular Aero-Propulsion System Simulation dataset.



\- 100 training engines

\- 20,631 training observations

\- 21 sensor measurements

\- 3 operational settings

\- 1 operating cycle feature



The training dataset is used for model development and the dashboard demonstration.



Dataset source: \[NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)



\## Model Development



Three approaches were evaluated for RUL prediction:



\- Random Forest

\- XGBoost

\- LSTM



The selected model is a 50-cycle LSTM with 128 hidden units and one recurrent layer.



Isolation Forest is used separately for detecting unusual sensor observations.



\### Validation Results



Selected LSTM model:



| Metric | Result |

|---|---:|

| MAE | 13.25 cycles |

| RMSE | 20.49 cycles |

| R² | 0.8522 |

| NASA Score | 36,080.34 |



These results were measured on the project's validation sequences. They should not be interpreted as guaranteed real-world engine performance.



\## System Architecture



1\. Historical engine sensor data is loaded.

2\. The latest 50 readings are prepared for RUL prediction.

3\. The LSTM estimates remaining useful life.

4\. Isolation Forest evaluates the latest sensor observation.

5\. FastAPI returns the prediction results.

6\. PostgreSQL stores readings and prediction logs.

7\. Streamlit displays the results and historical information.



\## Project Structure



```text

Predictive-Maintenance-System/

│

├── backend/

│   ├── main.py

│   ├── database.py

│   ├── models.py

│   ├── model\_loader.py

│   ├── init\_db.py

│   ├── test\_api.py

│   └── test\_inference.py

│

├── frontend/

│   └── app.py

│

├── models/

│   ├── final\_lstm\_model.pth

│   ├── final\_model\_results.json

│   ├── feature\_scaler.pkl

│   ├── isolation\_forest.pkl

│   ├── validation\_anomaly\_results.csv

│   └── engine\_anomaly\_summary.csv

│

├── data/

│   └── train\_FD001.txt

│

├── docs/

│

├── requirements.txt

├── .gitignore

└── README.md

```



\## Installation and Setup



\### 1. Clone the repository



```bash

git clone https://github.com/hamad-niaz/Predictive-Maintenance-System.git

cd Predictive-Maintenance-System

```



\### 2. Create a virtual environment



```bash

python -m venv venv

```



Activate on Windows PowerShell:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



```bash

pip install -r requirements.txt

```



\### 4. Configure PostgreSQL



Install PostgreSQL and create a database named:



```text

predictive\_maintenance

```



Configure the database connection in `backend/database.py`.



Do not commit database passwords or other secrets to GitHub.



\### 5. Initialize the database



```bash

python -m backend.init\_db

```



\### 6. Start the FastAPI backend



```bash

python -m uvicorn backend.main:app --reload

```



API base URL:



```text

http://127.0.0.1:8000

```



Interactive API documentation:



```text

http://127.0.0.1:8000/docs

```



\### 7. Start the Streamlit dashboard



Open another PowerShell terminal, activate the virtual environment, and run:



```bash

python -m streamlit run frontend/app.py

```



Dashboard URL:



```text

http://localhost:8501

```



\## API Endpoints



| Method | Endpoint | Purpose |

|---|---|---|

| GET | `/` | API information |

| GET | `/health` | Backend and model status |

| POST | `/predict/rul` | Predict remaining useful life |

| POST | `/detect/anomaly` | Detect unusual sensor readings |



\## Testing



Run the API integration tests:



```bash

python -m backend.test\_api

```



Run the model inference test:



```bash

python -m backend.test\_inference

```



The API integration test has returned HTTP 200 for both prediction endpoints, with successful database writes.



\## Important Limitations



\- This project uses historical simulated NASA C-MAPSS data, not live industrial sensor streams.

\- RUL is an estimate, not a guaranteed time to failure.

\- An anomaly indicates unusual sensor behavior, not a confirmed machine failure.

\- Validation metrics do not guarantee equivalent performance on real industrial equipment.

\- The dashboard is a portfolio demonstration, not a certified industrial monitoring system.



\## Future Improvements



\- Docker-based deployment

\- Cloud deployment

\- Live sensor data integration

\- Authentication and user management

\- Automated model monitoring

\- Expanded testing and CI/CD



\## Author



\*\*Hamad Ullah Niaz\*\*



