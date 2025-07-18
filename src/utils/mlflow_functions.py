import mlflow
from src.utils.config_loader import get_config_value

def setup_mlflow(config, exp_name):
    """Setup MLflow tracking based on configuration."""
    mlflow_config = get_config_value(config, 'mlflow', {})
    mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'file:./mlruns'))
    mlflow.set_experiment(exp_name)