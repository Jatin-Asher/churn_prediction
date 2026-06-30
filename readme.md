# Customer Churn Prediction API

FastAPI service and browser interface for interacting with the customer churn prediction model.

## Run the system

Install the project dependencies in the Python environment you want to use:

```bash
pip install -r requirements.txt
```

Start the API from the project root:

```bash
uvicorn app.main:app --reload
```

Then open these URLs:

- Browser interface: <http://127.0.0.1:8000/ui>
- Interface alias: <http://127.0.0.1:8000/interface>
- System health JSON: <http://127.0.0.1:8000/>
- Swagger API docs: <http://127.0.0.1:8000/docs>
- ReDoc API docs: <http://127.0.0.1:8000/redoc>

## Prediction API

Endpoint: `POST /predict`

Example request body:

```json
{
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure_months": 12,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "Yes",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 85.7,
  "total_charges": 1028.4,
  "cltv": 3200.0
}
```

Example successful response:

```json
{
  "prediction": "Churn",
  "probability": 0.84,
  "risk_level": "High Risk"
}
```
