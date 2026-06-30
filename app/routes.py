"""API routes for Customer Churn Prediction."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schema import CustomerInput, HealthResponse, PredictionResponse


MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "preprocessor.pkl"


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Customer Churn Prediction"])


def get_risk_level(probability: float) -> str:
    """Convert churn probability to a business-friendly risk label."""

    if probability < 0.50:
        return "Low Risk"
    if probability <= 0.80:
        return "Medium Risk"
    return "High Risk"


def validate_artifacts() -> None:
    """Ensure required model artifacts exist before prediction."""

    logger.info("Checking model artifact path: %s", MODEL_PATH)
    logger.info("Checking preprocessor artifact path: %s", PREPROCESSOR_PATH)

    missing_artifacts = [
        str(path)
        for path in (MODEL_PATH, PREPROCESSOR_PATH)
        if not Path(path).exists()
    ]

    if missing_artifacts:
        logger.error("Missing required artifact(s): %s", missing_artifacts)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Required model artifact(s) are missing.",
                "missing_artifacts": missing_artifacts,
            },
        )

    logger.info("Required artifacts found")


def extract_scalar(value: Any) -> Any:
    """Extract a Python scalar from NumPy/Pandas prediction outputs."""

    if hasattr(value, "item"):
        return value.item()
    return value


def get_prediction_label(prediction_value: int) -> str:
    """Convert the model class into the API response label."""

    return "Churn" if prediction_value == 1 else "No Churn"


@router.get("/", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def home() -> HealthResponse:
    """Return project information and API health status."""

    model_exists = MODEL_PATH.exists()
    preprocessor_exists = PREPROCESSOR_PATH.exists()

    api_status = "healthy" if model_exists and preprocessor_exists else "degraded"

    return HealthResponse(
        project="Customer Churn Prediction ML API",
        api_status=api_status,
        version="1.0.0",
        model_artifact=str(MODEL_PATH),
        preprocessor_artifact=str(PREPROCESSOR_PATH),
        documentation={
            "interface": "/ui",
            "swagger_ui": "/docs",
            "redoc": "/redoc",
        },
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict_churn(customer: CustomerInput) -> PredictionResponse:
    """
    Predict whether a customer is likely to churn.

    The request model mirrors `CustomData` in the existing prediction pipeline.
    The endpoint intentionally delegates preprocessing, artifact loading, and
    prediction to `PredictPipeline` to avoid duplicating ML pipeline logic.
    """

    try:
        request_payload = customer.model_dump()
        logger.info("Incoming prediction request: %s", request_payload)

        validate_artifacts()

        from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

        custom_data = CustomData(**request_payload)
        input_dataframe = custom_data.get_data_as_dataframe()

        logger.info("Input DataFrame columns: %s", input_dataframe.columns.tolist())
        logger.info("Input DataFrame shape: %s", input_dataframe.shape)

        prediction_pipeline = PredictPipeline()
        prediction, probability = prediction_pipeline.predict(input_dataframe)

        prediction_value = int(extract_scalar(prediction[0]))
        prediction_label = get_prediction_label(prediction_value)
        churn_probability = round(float(extract_scalar(probability[0])), 4)
        risk_level = get_risk_level(churn_probability)

        logger.info(
            "Prediction completed successfully: raw_prediction=%s, prediction=%s, probability=%s, risk=%s",
            prediction_value,
            prediction_label,
            churn_probability,
            risk_level,
        )

        return PredictionResponse(
            prediction=prediction_label,
            probability=churn_probability,
            risk_level=risk_level,
        )

    except HTTPException:
        raise
    except ModuleNotFoundError as exc:
        logger.exception("Prediction dependency is missing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Prediction failed because a required Python dependency is not installed in the running environment.",
                "missing_dependency": exc.name,
                "fix": "Install project dependencies in the same environment used to run uvicorn: pip install -r requirements.txt",
            },
        ) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Prediction failed. Please verify input values and model artifacts.",
                "error": str(exc),
                "expected_request_fields": list(CustomerInput.model_fields.keys()),
            },
        ) from exc