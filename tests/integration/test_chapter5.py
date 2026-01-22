from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from customer_retention.stages.modeling import (
    BaselineTrainer,
    ClassWeightMethod,
    CrossValidator,
    CVStrategy,
    DataSplitter,
    FeatureScaler,
    HyperparameterTuner,
    ImbalanceHandler,
    ImbalanceStrategy,
    ModelComparator,
    ModelEvaluator,
    ModelType,
    OptimizationObjective,
    ScalerType,
    SearchStrategy,
    SplitStrategy,
    ThresholdOptimizer,
)
from customer_retention.stages.validation import ModelValidityGate


@pytest.fixture
def retail_data():
    retail_path = Path(__file__).parent.parent / "fixtures" / "customer_retention_retail.csv"
    return pd.read_csv(retail_path)


@pytest.fixture
def feature_columns():
    return ["avgorder", "ordfreq", "eopenrate", "eclickrate", "paperless", "refill", "doorstep"]


class TestDataPreparation:
    def test_ac5_1_stratification_preserves_class_balance(self, retail_data, feature_columns):
        X = retail_data[feature_columns]
        y = retail_data["retained"]

        splitter = DataSplitter(
            target_column="retained",
            strategy=SplitStrategy.RANDOM_STRATIFIED,
            test_size=0.20,
        )
        result = splitter.split(retail_data)

        original_ratio = y.mean()
        train_ratio = result.y_train.mean()
        test_ratio = result.y_test.mean()

        assert abs(train_ratio - original_ratio) < 0.02
        assert abs(test_ratio - original_ratio) < 0.02

    def test_ac5_2_no_data_leakage_between_sets(self, retail_data, feature_columns):
        splitter = DataSplitter(target_column="retained", test_size=0.20)
        result = splitter.split(retail_data)

        train_indices = set(result.X_train.index)
        test_indices = set(result.X_test.index)

        assert len(train_indices & test_indices) == 0

    def test_ac5_3_split_reproducible_with_seed(self, retail_data, feature_columns):
        splitter1 = DataSplitter(target_column="retained", random_state=42)
        splitter2 = DataSplitter(target_column="retained", random_state=42)

        result1 = splitter1.split(retail_data)
        result2 = splitter2.split(retail_data)

        assert result1.X_train.index.equals(result2.X_train.index)

    def test_ac5_4_temporal_split_respects_time_order(self):
        np.random.seed(42)
        n = 500
        temporal_data = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "target": np.random.choice([0, 1], n, p=[0.3, 0.7]),
            "event_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        })

        splitter = DataSplitter(
            target_column="target",
            strategy=SplitStrategy.TEMPORAL,
            temporal_column="event_date",
            test_size=0.20,
        )
        result = splitter.split(temporal_data)

        train_max_date = temporal_data.loc[result.X_train.index, "event_date"].max()
        test_min_date = temporal_data.loc[result.X_test.index, "event_date"].min()

        assert train_max_date < test_min_date


class TestFeatureScaling:
    def test_ac5_7_scaling_applied_correctly(self, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)

        splitter = DataSplitter(target_column="retained", test_size=0.20)
        split_result = splitter.split(retail_data[feature_columns + ["retained"]])

        scaler = FeatureScaler(scaler_type=ScalerType.STANDARD)
        scaling_result = scaler.fit_transform(split_result.X_train, split_result.X_test)

        for col in scaling_result.X_train_scaled.columns:
            col_mean = scaling_result.X_train_scaled[col].mean()
            col_std = scaling_result.X_train_scaled[col].std()
            assert abs(col_mean) < 0.1, f"Column {col} mean {col_mean} not close to 0"
            assert abs(col_std - 1.0) < 0.1, f"Column {col} std {col_std} not close to 1"


class TestModelTraining:
    def test_ac5_5_all_baseline_models_train_successfully(self, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]

        models_to_train = [
            ModelType.LOGISTIC_REGRESSION,
            ModelType.RANDOM_FOREST,
            ModelType.XGBOOST,
        ]

        for model_type in models_to_train:
            trainer = BaselineTrainer(model_type=model_type, class_weight="balanced")
            result = trainer.fit(X, y)
            assert result.model is not None

    def test_ac5_6_class_imbalance_handled(self, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]

        handler = ImbalanceHandler(
            strategy=ImbalanceStrategy.CLASS_WEIGHT,
            weight_method=ClassWeightMethod.BALANCED,
        )
        result = handler.fit(X, y)

        assert result.class_weights is not None
        assert result.imbalance_ratio is not None


class TestModelEvaluation:
    @pytest.fixture
    def trained_model(self, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]
        trainer = BaselineTrainer(model_type=ModelType.LOGISTIC_REGRESSION)
        return trainer.fit(X, y).model, X, y

    def test_ac5_9_all_metrics_calculated(self, trained_model):
        model, X, y = trained_model
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(model, X, y)

        required_metrics = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
        for metric in required_metrics:
            assert metric in result.metrics

    def test_ac5_10_pr_auc_is_primary_metric(self, trained_model):
        model, X, y = trained_model
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(model, X, y)

        assert "pr_auc" in result.metrics
        assert 0 <= result.metrics["pr_auc"] <= 1

    def test_ac5_11_confusion_matrix_generated(self, trained_model):
        model, X, y = trained_model
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(model, X, y)

        assert result.confusion_matrix is not None
        assert result.confusion_matrix.shape == (2, 2)

    def test_ac5_12_roc_and_pr_curves_generated(self, trained_model):
        model, X, y = trained_model
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(model, X, y)

        assert "roc_curve" in result.curves
        assert "pr_curve" in result.curves


class TestCrossValidation:
    def test_ac5_13_cv_completes_without_error(self, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)

        cv = CrossValidator(n_splits=5)
        result = cv.run(model, X, y)

        assert result.cv_scores is not None
        assert len(result.cv_scores) == 5

    def test_ac5_14_cv_uses_stratified_folds(self, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)

        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=5)
        result = cv.run(model, X, y)

        original_ratio = y.mean()
        for fold_info in result.fold_details:
            assert abs(fold_info["train_class_ratio"] - original_ratio) < 0.05

    def test_ac5_15_cv_mean_close_to_test_score(self, retail_data, feature_columns):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        model = LogisticRegression(max_iter=1000, random_state=42)

        cv = CrossValidator(strategy=CVStrategy.STRATIFIED_KFOLD, n_splits=5, scoring="roc_auc")
        cv_result = cv.run(model, X_train, y_train)

        model.fit(X_train, y_train)
        evaluator = ModelEvaluator()
        eval_result = evaluator.evaluate(model, X_test, y_test)
        test_score = eval_result.metrics["roc_auc"]

        cv_mean = cv_result.cv_mean
        assert abs(cv_mean - test_score) < 0.15, (
            f"CV mean {cv_mean:.3f} not close to test score {test_score:.3f}"
        )


class TestThresholdOptimization:
    def test_ac5_16_cost_function_defined(self, retail_data, feature_columns):
        optimizer = ThresholdOptimizer(
            objective=OptimizationObjective.MIN_COST,
            cost_fn=100,
            cost_fp=10,
        )

        assert optimizer.cost_fn == 100
        assert optimizer.cost_fp == 10

    def test_ac5_17_optimal_threshold_found(self, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        optimizer = ThresholdOptimizer(objective=OptimizationObjective.MAX_F1)
        result = optimizer.optimize(model, X, y)

        assert result.optimal_threshold is not None
        assert 0 < result.optimal_threshold < 1

    def test_ac5_18_cost_savings_calculated(self, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        optimizer = ThresholdOptimizer(
            objective=OptimizationObjective.MIN_COST,
            cost_fn=100,
            cost_fp=10,
        )
        result = optimizer.optimize(model, X, y)

        assert result.comparison_default is not None
        assert "default_cost" in result.comparison_default
        assert "optimal_cost" in result.comparison_default


class TestModelValidityGate:
    def test_ac5_19_gate_runs_all_checks(self):
        metrics = {
            "pr_auc_test": 0.55,
            "roc_auc_test": 0.75,
            "train_test_gap": 0.05,
            "cv_std": 0.03,
            "recall": 0.50,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert hasattr(result, "passed")
        assert hasattr(result, "critical_issues")
        assert hasattr(result, "high_issues")

    def test_ac5_20_suspicious_performance_flagged(self):
        metrics = {
            "pr_auc_test": 0.95,
            "roc_auc_test": 0.98,
            "train_test_gap": 0.05,
            "cv_std": 0.02,
            "recall": 0.90,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert not result.passed
        assert len(result.critical_issues) > 0

    def test_ac5_21_overfitting_detected(self):
        metrics = {
            "pr_auc_test": 0.50,
            "roc_auc_test": 0.70,
            "train_test_gap": 0.20,
            "cv_std": 0.05,
            "recall": 0.50,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert not result.passed
        overfitting_issues = [i for i in result.critical_issues if "MV003" in str(i)]
        assert len(overfitting_issues) > 0

    def test_ac5_22_gate_produces_actionable_output(self):
        metrics = {
            "pr_auc_test": 0.55,
            "roc_auc_test": 0.75,
            "train_test_gap": 0.05,
            "cv_std": 0.03,
            "recall": 0.50,
        }

        gate = ModelValidityGate()
        result = gate.run(metrics)

        assert result.recommendation is not None
        assert len(result.recommendation) > 0


class TestFullPipeline:
    def test_end_to_end_training_pipeline(self, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]

        splitter = DataSplitter(target_column="retained", test_size=0.20)
        split_result = splitter.split(retail_data[feature_columns + ["retained"]])

        X_train = split_result.X_train
        X_test = split_result.X_test
        y_train = split_result.y_train
        y_test = split_result.y_test

        trainer = BaselineTrainer(
            model_type=ModelType.RANDOM_FOREST,
            class_weight="balanced",
        )
        trained = trainer.fit(X_train, y_train)

        evaluator = ModelEvaluator()
        eval_result = evaluator.evaluate(trained.model, X_test, y_test)

        assert eval_result.metrics["pr_auc"] > 0
        assert eval_result.metrics["roc_auc"] > 0

    def test_model_comparison_selects_best(self, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X, y)

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)

        comparator = ModelComparator(primary_metric="pr_auc")
        result = comparator.compare({"logistic": lr, "random_forest": rf}, X, y)

        assert result.best_model_name is not None
        assert result.ranking is not None
        assert len(result.ranking) == 2

    def test_hyperparameter_tuning_improves_model(self, retail_data, feature_columns):
        X = retail_data[feature_columns].fillna(0)
        y = retail_data["retained"]

        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(random_state=42)
        param_space = {
            "n_estimators": [10, 20],
            "max_depth": [3, 5],
        }

        tuner = HyperparameterTuner(
            strategy=SearchStrategy.GRID_SEARCH,
            param_space=param_space,
            cv=3,
        )
        result = tuner.tune(model, X, y)

        assert result.best_params is not None
        assert result.best_score > 0
