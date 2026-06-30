"""Pydantic schemas for the Customer Churn Prediction API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SWAGGER_REQUEST_EXAMPLE = {
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
    "cltv": 3200.0,
}


class CustomerInput(BaseModel):
    """
    Request schema matching src.pipeline.prediction_pipeline.CustomData.

    The field names intentionally use snake_case because they map directly to
    the CustomData constructor arguments in the existing prediction pipeline.
    """

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={"example": SWAGGER_REQUEST_EXAMPLE},
    )

    gender: Literal["Male", "Female"] = Field(..., description="Customer gender.")
    senior_citizen: Literal[0, 1] = Field(
        ...,
        description="Senior citizen flag. Use 1 for senior citizen, otherwise 0.",
    )
    partner: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has a partner.",
    )
    dependents: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has dependents.",
    )
    tenure_months: int = Field(..., ge=0, description="Customer tenure in months.")
    phone_service: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has phone service.",
    )
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(
        ...,
        description="Customer multiple-lines subscription status.",
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ...,
        description="Customer internet service type.",
    )
    online_security: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Customer online-security subscription status.",
    )
    online_backup: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Customer online-backup subscription status.",
    )
    device_protection: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Customer device-protection subscription status.",
    )
    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Customer tech-support subscription status.",
    )
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Customer streaming-TV subscription status.",
    )
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(
        ...,
        description="Customer streaming-movies subscription status.",
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ...,
        description="Customer contract type.",
    )
    paperless_billing: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer uses paperless billing.",
    )
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Customer payment method.")
    monthly_charges: float = Field(
        ...,
        ge=0,
        description="Current monthly charges for the customer.",
    )
    total_charges: float = Field(
        ...,
        ge=0,
        description="Total charges billed to the customer.",
    )
    cltv: float = Field(..., ge=0, description="Customer lifetime value.")


class PredictionResponse(BaseModel):
    """Response schema returned by the /predict endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    prediction: Literal["Churn", "No Churn"] = Field(
        ...,
        description="Business-friendly predicted churn label.",
    )
    probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Estimated probability that the customer will churn.",
    )
    risk_level: Literal["Low Risk", "Medium Risk", "High Risk"] = Field(
        ...,
        description="Risk category derived from churn probability.",
    )


class HealthResponse(BaseModel):
    """Response schema returned by the root health endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    project: str
    api_status: str
    version: str
    model_artifact: str
    preprocessor_artifact: str
    documentation: dict[str, str]