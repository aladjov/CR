from pathlib import Path
from typing import Dict, List

from .pipeline_spec import PipelineSpec


class GenericSpecGenerator:
    def __init__(self, output_dir: str = "./generated_pipelines"):
        self.output_dir = output_dir

    def generate_python_pipeline(self, spec: PipelineSpec) -> str:
        lines = [
            '"""',
            f'Auto-generated pipeline: {spec.name}',
            f'Version: {spec.version}',
            f'Description: {spec.description}',
            '"""',
            '',
            'import pandas as pd',
            'import numpy as np',
            'from sklearn.model_selection import train_test_split',
            'from sklearn.preprocessing import StandardScaler, OneHotEncoder',
            'from sklearn.compose import ColumnTransformer',
            'from sklearn.pipeline import Pipeline',
            'from sklearn.ensemble import GradientBoostingClassifier',
            'from sklearn.metrics import classification_report, roc_auc_score',
            '',
            '',
            f'def load_data(path: str = "{spec.sources[0].path if spec.sources else "data.csv"}") -> pd.DataFrame:',
            '    """Load and return the dataset."""',
        ]
        if spec.sources and spec.sources[0].format == "csv":
            lines.append('    return pd.read_csv(path)')
        elif spec.sources and spec.sources[0].format == "parquet":
            lines.append('    return pd.read_parquet(path)')
        else:
            lines.append('    return pd.read_csv(path)')
        lines.extend([
            '',
            '',
            'def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:',
            '    """Apply preprocessing transformations."""',
            '    df = df.copy()',
        ])
        for transform in spec.silver_transforms[:5]:
            if transform.transform_type == "standard_scaling":
                lines.append(f'    # Scale {transform.input_columns}')
            elif transform.transform_type == "one_hot_encoding":
                lines.append(f'    # Encode {transform.input_columns}')
        lines.extend([
            '    return df',
            '',
            '',
        ])
        target_col = spec.model_config.target_column if spec.model_config else "target"
        lines.extend([
            'def build_pipeline():',
            '    """Build the ML pipeline."""',
            '    numeric_features = []',
            '    categorical_features = []',
        ])
        if spec.schema:
            for col in spec.schema.columns[:5]:
                if col.semantic_type in ["numeric_continuous", "numeric_discrete"]:
                    lines.append(f'    numeric_features.append("{col.name}")')
                elif col.semantic_type in ["categorical_nominal", "categorical_ordinal"]:
                    lines.append(f'    categorical_features.append("{col.name}")')
        lines.extend([
            '',
            '    preprocessor = ColumnTransformer(',
            '        transformers=[',
            '            ("num", StandardScaler(), numeric_features),',
            '            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)',
            '        ]',
            '    )',
            '',
            '    pipeline = Pipeline([',
            '        ("preprocessor", preprocessor),',
            '        ("classifier", GradientBoostingClassifier())',
            '    ])',
            '',
            '    return pipeline',
            '',
            '',
            f'def train_model(df: pd.DataFrame, target_column: str = "{target_col}"):',
            '    """Train the model."""',
            '    X = df.drop(columns=[target_column])',
            '    y = df[target_column]',
            '',
            '    X_train, X_test, y_train, y_test = train_test_split(',
            '        X, y, test_size=0.2, random_state=42',
            '    )',
            '',
            '    pipeline = build_pipeline()',
            '    pipeline.fit(X_train, y_train)',
            '',
            '    y_pred = pipeline.predict(X_test)',
            '    y_proba = pipeline.predict_proba(X_test)[:, 1]',
            '',
            '    print("Classification Report:")',
            '    print(classification_report(y_test, y_pred))',
            '    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")',
            '',
            '    return pipeline',
            '',
            '',
            'if __name__ == "__main__":',
            '    df = load_data()',
            '    df = preprocess_data(df)',
            '    model = train_model(df)',
        ])
        return '\n'.join(lines)

    def generate_airflow_dag(self, spec: PipelineSpec) -> str:
        dag_id = spec.name.replace(" ", "_").lower()
        lines = [
            '"""',
            f'Airflow DAG: {spec.name}',
            'Auto-generated pipeline DAG',
            '"""',
            '',
            'from datetime import datetime, timedelta',
            'from airflow import DAG',
            'from airflow.operators.python import PythonOperator',
            'from airflow.operators.dummy import DummyOperator',
            '',
            '',
            'default_args = {',
            '    "owner": "data_team",',
            '    "depends_on_past": False,',
            '    "email_on_failure": True,',
            '    "retries": 1,',
            '    "retry_delay": timedelta(minutes=5),',
            '}',
            '',
            '',
            'with DAG(',
            f'    dag_id="{dag_id}",',
            '    default_args=default_args,',
            '    description="Customer retention ML pipeline",',
            '    schedule_interval="@daily",',
            '    start_date=datetime(2024, 1, 1),',
            '    catchup=False,',
            '    tags=["ml", "retention"],',
            ') as dag:',
            '',
            '    start = DummyOperator(task_id="start")',
            '',
            '    def load_data_task():',
            '        """Load data from source."""',
            '        pass',
            '',
            '    load_data = PythonOperator(',
            '        task_id="load_data",',
            '        python_callable=load_data_task,',
            '    )',
            '',
            '    def transform_data_task():',
            '        """Apply data transformations."""',
            '        pass',
            '',
            '    transform_data = PythonOperator(',
            '        task_id="transform_data",',
            '        python_callable=transform_data_task,',
            '    )',
            '',
            '    def train_model_task():',
            '        """Train ML model."""',
            '        pass',
            '',
            '    train_model = PythonOperator(',
            '        task_id="train_model",',
            '        python_callable=train_model_task,',
            '    )',
            '',
            '    def evaluate_model_task():',
            '        """Evaluate model performance."""',
            '        pass',
            '',
            '    evaluate_model = PythonOperator(',
            '        task_id="evaluate_model",',
            '        python_callable=evaluate_model_task,',
            '    )',
            '',
            '    end = DummyOperator(task_id="end")',
            '',
            '    start >> load_data >> transform_data >> train_model >> evaluate_model >> end',
        ]
        return '\n'.join(lines)

    def generate_prefect_flow(self, spec: PipelineSpec) -> str:
        flow_name = spec.name.replace(" ", "_").lower()
        lines = [
            '"""',
            f'Prefect Flow: {spec.name}',
            'Auto-generated pipeline flow',
            '"""',
            '',
            'from prefect import flow, task',
            'import pandas as pd',
            '',
            '',
            '@task(name="load_data")',
            f'def load_data(path: str = "{spec.sources[0].path if spec.sources else "data.csv"}"):',
            '    """Load data from source."""',
            '    return pd.read_csv(path)',
            '',
            '',
            '@task(name="transform_data")',
            'def transform_data(df: pd.DataFrame) -> pd.DataFrame:',
            '    """Apply data transformations."""',
            '    return df',
            '',
            '',
            '@task(name="train_model")',
            'def train_model(df: pd.DataFrame):',
            '    """Train ML model."""',
            '    pass',
            '',
            '',
            '@task(name="evaluate_model")',
            'def evaluate_model(model, df: pd.DataFrame):',
            '    """Evaluate model performance."""',
            '    pass',
            '',
            '',
            f'@flow(name="{flow_name}")',
            'def main_flow():',
            '    """Main pipeline flow."""',
            '    df = load_data()',
            '    df = transform_data(df)',
            '    model = train_model(df)',
            '    evaluate_model(model, df)',
            '',
            '',
            'if __name__ == "__main__":',
            '    main_flow()',
        ]
        return '\n'.join(lines)

    def generate_docker_compose(self, spec: PipelineSpec) -> str:
        lines = [
            'version: "3.8"',
            '',
            'services:',
            f'  {spec.name.replace(" ", "_").lower()}:',
            '    build: .',
            '    volumes:',
            '      - ./data:/app/data',
            '      - ./models:/app/models',
            '    environment:',
            '      - PYTHONUNBUFFERED=1',
            '    command: python pipeline.py',
            '',
            '  jupyter:',
            '    image: jupyter/scipy-notebook:latest',
            '    ports:',
            '      - "8888:8888"',
            '    volumes:',
            '      - .:/home/jovyan/work',
            '    environment:',
            '      - JUPYTER_ENABLE_LAB=yes',
        ]
        return '\n'.join(lines)

    def generate_requirements(self, spec: PipelineSpec) -> str:
        requirements = [
            'pandas>=2.0.0',
            'numpy>=1.24.0',
            'scikit-learn>=1.3.0',
            'scipy>=1.11.0',
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
            'plotly>=5.15.0',
            'pyyaml>=6.0',
        ]
        if any("gradient" in (spec.model_config.model_type or "").lower() if spec.model_config else False for _ in [1]):
            pass
        return '\n'.join(requirements)

    def generate_readme(self, spec: PipelineSpec) -> str:
        lines = [
            f'# {spec.name}',
            '',
            f'{spec.description}',
            '',
            '## Overview',
            '',
            f'- **Version:** {spec.version}',
            f'- **Created:** {spec.created_at}',
            '',
            '## Data Sources',
            '',
        ]
        for source in spec.sources:
            lines.append(f'- **{source.name}:** `{source.path}` ({source.format})')
        lines.extend([
            '',
            '## Schema',
            '',
            '| Column | Type | Semantic Type | Nullable |',
            '|--------|------|---------------|----------|',
        ])
        if spec.schema:
            for col in spec.schema.columns[:10]:
                lines.append(f'| {col.name} | {col.data_type} | {col.semantic_type} | {col.nullable} |')
        lines.extend([
            '',
            '## Pipeline Stages',
            '',
            '### Bronze (Raw)',
            f'- {len(spec.bronze_transforms)} transforms',
            '',
            '### Silver (Cleaned)',
            f'- {len(spec.silver_transforms)} transforms',
            '',
            '### Gold (Features)',
            f'- {len(spec.gold_transforms)} transforms',
            f'- {len(spec.feature_definitions)} feature definitions',
            '',
            '## Model',
            '',
        ])
        if spec.model_config:
            lines.extend([
                f'- **Type:** {spec.model_config.model_type}',
                f'- **Target:** {spec.model_config.target_column}',
                f'- **Features:** {len(spec.model_config.feature_columns)} columns',
            ])
        lines.extend([
            '',
            '## Quality Gates',
            '',
        ])
        for gate in spec.quality_gates:
            lines.append(f'- **{gate.name}:** {gate.gate_type} (threshold: {gate.threshold})')
        lines.extend([
            '',
            '## Usage',
            '',
            '```bash',
            'pip install -r requirements.txt',
            'python pipeline.py',
            '```',
        ])
        return '\n'.join(lines)

    def generate_all(self, spec: PipelineSpec) -> Dict[str, str]:
        return {
            "pipeline.py": self.generate_python_pipeline(spec),
            "dag.py": self.generate_airflow_dag(spec),
            "flow.py": self.generate_prefect_flow(spec),
            "docker-compose.yml": self.generate_docker_compose(spec),
            "requirements.txt": self.generate_requirements(spec),
            "README.md": self.generate_readme(spec)
        }

    def save_all(self, spec: PipelineSpec) -> List[str]:
        files = self.generate_all(spec)
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved = []
        for filename, content in files.items():
            file_path = output_path / filename
            file_path.write_text(content)
            saved.append(filename)
        return saved
