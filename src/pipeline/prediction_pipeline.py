import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_object


logger = logging.getLogger(__name__)

MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.pkl"

PREPROCESSOR_PATH = (
    PROJECT_ROOT / "artifacts" / "preprocessor.pkl"
)


def _as_column_list(columns):
    """Return string column names from a ColumnTransformer column selector."""

    if isinstance(columns, str):
        return [columns]

    if isinstance(columns, (list, tuple)):
        return [column for column in columns if isinstance(column, str)]

    return []


def get_required_preprocessor_columns(preprocessor):
    """
    Get only the raw columns actually consumed by the fitted preprocessor.

    ColumnTransformer.feature_names_in_ contains every column present during
    fitting, including columns dropped by ``remainder='drop'``. Requiring all of
    those columns at prediction time breaks the API when the training DataFrame
    had unused columns such as location fields or labels. The transformer column
    selectors are the real prediction-time requirements.
    """

    required_columns = []

    transformers = getattr(preprocessor, "transformers_", None) or getattr(
        preprocessor,
        "transformers",
        [],
    )

    for transformer_definition in transformers:
        if len(transformer_definition) < 3:
            continue

        name, transformer, columns = transformer_definition[:3]

        if name == "remainder" or transformer == "drop" or columns is None:
            continue

        required_columns.extend(_as_column_list(columns))

    if required_columns:
        return list(dict.fromkeys(required_columns))

    if hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)

    return []


def iter_estimator_tree(estimator):
    """Yield an estimator and all nested estimators in pipelines/transformers."""

    yield estimator

    for _, step in getattr(estimator, "steps", []):
        if step in ("drop", "passthrough"):
            continue

        yield from iter_estimator_tree(step)

    for transformer_definition in getattr(estimator, "transformers_", []):
        if len(transformer_definition) < 2:
            continue

        transformer = transformer_definition[1]
        if transformer in ("drop", "passthrough"):
            continue

        yield from iter_estimator_tree(transformer)

    for _, transformer in getattr(estimator, "transformer_list", []):
        if transformer in ("drop", "passthrough"):
            continue

        yield from iter_estimator_tree(transformer)


def patch_sklearn_artifact_compatibility(estimator):
    """
    Repair known scikit-learn pickle compatibility issues in old artifacts.

    scikit-learn objects are not guaranteed to be forward-compatible across
    versions when unpickled. Some newer SimpleImputer versions expect a private
    ``_fill_dtype`` attribute that older pickled SimpleImputer objects do not
    contain, causing: ``'SimpleImputer' object has no attribute '_fill_dtype'``.
    The attribute is only used for dtype handling during transform. For these
    fitted imputers, the fitted ``statistics_`` dtype is the safest available
    fallback.
    """

    patched_imputers = 0

    for nested_estimator in iter_estimator_tree(estimator):
        if nested_estimator.__class__.__name__ != "SimpleImputer":
            continue

        if hasattr(nested_estimator, "_fill_dtype"):
            continue

        statistics = getattr(nested_estimator, "statistics_", None)
        if statistics is not None and hasattr(statistics, "dtype"):
            nested_estimator._fill_dtype = statistics.dtype
        else:
            nested_estimator._fill_dtype = object

        patched_imputers += 1

    if patched_imputers:
        logger.warning(
            "Patched %s SimpleImputer object(s) for scikit-learn artifact compatibility",
            patched_imputers,
        )

    return estimator



class PredictPipeline:

    def __init__(self):

        self.model_path = MODEL_PATH

        self.preprocessor_path = PREPROCESSOR_PATH

    def predict(self, features):

        logger.info("Starting prediction pipeline")
        logger.info("Prediction input DataFrame columns: %s", features.columns.tolist())
        logger.info("Prediction input DataFrame shape: %s", features.shape)
        logger.info("Loading model from: %s", self.model_path)

        model = load_object(
            self.model_path
        )

        logger.info("Model loaded successfully: %s", type(model).__name__)
        logger.info("Loading preprocessor from: %s", self.preprocessor_path)

        preprocessor = load_object(
            self.preprocessor_path
        )
        preprocessor = patch_sklearn_artifact_compatibility(preprocessor)

        logger.info("Preprocessor loaded successfully: %s", type(preprocessor).__name__)

        expected_columns = get_required_preprocessor_columns(preprocessor)

        if expected_columns:
            logger.info("Preprocessor required raw columns: %s", expected_columns)

            missing_columns = [
                column for column in expected_columns if column not in features.columns
            ]
            extra_columns = [
                column for column in features.columns if column not in expected_columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Input DataFrame is missing required column(s): {missing_columns}. "
                    f"Received columns: {features.columns.tolist()}"
                )

            if extra_columns:
                logger.warning(
                    "Input DataFrame contains extra column(s) not used by preprocessor: %s",
                    extra_columns,
                )

            features = features[expected_columns]
            logger.info("Input DataFrame reordered to expected columns")

        data_scaled = preprocessor.transform(
            features
        )

        logger.info("Transformed feature matrix shape: %s", data_scaled.shape)

        prediction = model.predict(
            data_scaled
        )

        logger.info("Prediction result: %s", prediction)

        probability = model.predict_proba(
            data_scaled
        )[:, 1]

        logger.info("Probability result: %s", probability)

        return prediction, probability
    
class CustomData:

    def __init__(
        self,
        gender,
        senior_citizen,
        partner,
        dependents,
        tenure_months,
        phone_service,
        multiple_lines,
        internet_service,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies,
        contract,
        paperless_billing,
        payment_method,
        monthly_charges,
        total_charges,
        cltv
    ):

        self.gender = gender

        self.senior_citizen = senior_citizen

        self.partner = partner

        self.dependents = dependents

        self.tenure_months = tenure_months

        self.phone_service = phone_service

        self.multiple_lines = multiple_lines

        self.internet_service = internet_service

        self.online_security = online_security

        self.online_backup = online_backup

        self.device_protection = device_protection

        self.tech_support = tech_support

        self.streaming_tv = streaming_tv

        self.streaming_movies = streaming_movies

        self.contract = contract

        self.paperless_billing = paperless_billing

        self.payment_method = payment_method

        self.monthly_charges = monthly_charges

        self.total_charges = total_charges

        self.cltv = cltv

    def get_data_as_dataframe(self):

        import pandas as pd

        custom_data_input_dict = {

            "Gender": [self.gender],

            "Senior Citizen": [self.senior_citizen],

            "Partner": [self.partner],

            "Dependents": [self.dependents],

            "Tenure Months": [self.tenure_months],

            "Phone Service": [self.phone_service],

            "Multiple Lines": [self.multiple_lines],

            "Internet Service": [self.internet_service],

            "Online Security": [self.online_security],

            "Online Backup": [self.online_backup],

            "Device Protection": [self.device_protection],

            "Tech Support": [self.tech_support],

            "Streaming TV": [self.streaming_tv],

            "Streaming Movies": [self.streaming_movies],

            "Contract": [self.contract],

            "Paperless Billing": [self.paperless_billing],

            "Payment Method": [self.payment_method],

            "Monthly Charges": [self.monthly_charges],

            "Total Charges": [self.total_charges],

            "CLTV": [self.cltv]
        }

        dataframe = pd.DataFrame(
            custom_data_input_dict
        )

        logger.info("Created CustomData DataFrame columns: %s", dataframe.columns.tolist())
        logger.info("Created CustomData DataFrame shape: %s", dataframe.shape)

        return dataframe