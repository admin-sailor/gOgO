import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import logging
from typing import Dict, List, Tuple
from config import LR_MODEL_PATH, NN_MODEL_PATH, MODEL_DIR
import os
import math
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(MODEL_DIR, exist_ok=True)

class BTTSPredictor:
    """Machine Learning models for BTTS prediction"""
    
    def __init__(self):
        self.lr_model = None
        self.nn_model = None
        self.scaler = StandardScaler()
        self.agg_rf_model = None
        self.agg_xgb_model = None
        self.feature_names = [
            'home_scoring_rate', 'home_conceding_rate',
            'away_scoring_rate', 'away_conceding_rate',
            'home_dfi', 'away_dfi',
            'home_clean_sheet_freq', 'away_clean_sheet_freq',
            'home_win_rate', 'away_win_rate',
            'home_matches_played', 'away_matches_played',
        ]
        self.load_models()
    
    def load_models(self):
        """Load trained models from disk"""
        if os.path.exists(LR_MODEL_PATH):
            try:
                self.lr_model = joblib.load(LR_MODEL_PATH)
                logger.info("Logistic Regression model loaded")
            except Exception as e:
                logger.warning(f"Could not load LR model: {e}")
        
        if os.path.exists(NN_MODEL_PATH):
            try:
                self.nn_model = keras.models.load_model(NN_MODEL_PATH)
                logger.info("Neural Network model loaded")
            except Exception as e:
                logger.warning(f"Could not load NN model: {e}")
        # Aggregated dataset models (optional)
        rf_path = os.path.join(MODEL_DIR, 'aggregated_rf.pkl')
        xgb_path = os.path.join(MODEL_DIR, 'aggregated_xgb.json')
        if os.path.exists(rf_path):
            try:
                self.agg_rf_model = joblib.load(rf_path)
                logger.info("Aggregated RandomForest model loaded")
            except Exception as e:
                logger.warning(f"Could not load aggregated RF model: {e}")
        if XGB_AVAILABLE and os.path.exists(xgb_path):
            try:
                xgb = XGBClassifier()
                xgb.load_model(xgb_path)
                self.agg_xgb_model = xgb
                logger.info("Aggregated XGBoost model loaded")
            except Exception as e:
                logger.warning(f"Could not load aggregated XGB model: {e}")
    
    def prepare_features(self, home_stats: Dict, away_stats: Dict) -> np.ndarray:
        """Prepare feature vector for prediction"""
        features = [
            home_stats.get('goals_per_game', 0),
            home_stats.get('goals_conceded_per_game', 0),
            away_stats.get('goals_per_game', 0),
            away_stats.get('goals_conceded_per_game', 0),
            home_stats.get('defensive_fragility_index', 0),
            away_stats.get('defensive_fragility_index', 0),
            home_stats.get('clean_sheet_frequency', 0),
            away_stats.get('clean_sheet_frequency', 0),
            home_stats.get('win_rate', 0),
            away_stats.get('win_rate', 0),
            home_stats.get('matches_played', 0),
            away_stats.get('matches_played', 0),
        ]
        return np.array(features).reshape(1, -1)
    
    def train_logistic_regression(self, X: np.ndarray, y: np.ndarray):
        """Train Logistic Regression model"""
        X_scaled = self.scaler.fit_transform(X)
        self.lr_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        self.lr_model.fit(X_scaled, y)
        joblib.dump(self.lr_model, LR_MODEL_PATH)
        logger.info("Logistic Regression model trained and saved")
    
    def train_neural_network(self, X: np.ndarray, y: np.ndarray):
        """Train Neural Network model"""
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        self.nn_model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.nn_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.nn_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            verbose=0
        )
        
        self.nn_model.save(NN_MODEL_PATH)
        logger.info("Neural Network model trained and saved")
    
    def predict_btts_logistic(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """Predict BTTS probability using Logistic Regression"""
        if not self.lr_model:
            return self.predict_btts_heuristic(home_stats, away_stats, model_name='Logistic Regression (Heuristic)')
        
        try:
            features = self.prepare_features(home_stats, away_stats)
            # Handle scaler if not fitted (fallback to raw features if scaler fails)
            try:
                features_scaled = self.scaler.transform(features)
            except:
                features_scaled = features
            
            probability = self.lr_model.predict_proba(features_scaled)[0][1]
            prediction = self.lr_model.predict(features_scaled)[0]
            
            return {
                'model': 'Logistic Regression',
                'btts_probability': float(probability),
                'btts_prediction': bool(prediction),
                'confidence': max(probability, 1 - probability)
            }
        except Exception as e:
            logger.error(f"LR Prediction error: {e}")
            return self.predict_btts_heuristic(home_stats, away_stats)
    
    def predict_btts_neural(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """Predict BTTS probability using Neural Network"""
        if not self.nn_model:
             return self.predict_btts_heuristic(home_stats, away_stats, model_name='Neural Network (Heuristic)')
        
        try:
            features = self.prepare_features(home_stats, away_stats)
            try:
                features_scaled = self.scaler.transform(features)
            except:
                features_scaled = features
            
            probability = float(self.nn_model.predict(features_scaled, verbose=0)[0][0])
            prediction = probability > 0.5
            
            return {
                'model': 'Neural Network',
                'btts_probability': probability,
                'btts_prediction': bool(prediction),
                'confidence': max(probability, 1 - probability)
            }
        except Exception as e:
            logger.error(f"NN Prediction error: {e}")
            return self.predict_btts_heuristic(home_stats, away_stats)

    def predict_btts_heuristic(self, home_stats: Dict, away_stats: Dict, model_name: str = 'Heuristic') -> Dict:
        """
        Robust Poisson-based heuristic for BTTS probability.
        Uses expected goals to calculate probability of both teams scoring > 0.
        """
        # Calculate expected goals for home team (Home Attack vs Away Defense)
        home_attack = home_stats.get('goals_per_game', 1.0)
        away_defense = away_stats.get('goals_conceded_per_game', 1.0)
        # Average them (simple model)
        lambda_home = (home_attack + away_defense) / 2.0
        
        # Calculate expected goals for away team (Away Attack vs Home Defense)
        away_attack = away_stats.get('goals_per_game', 1.0)
        home_defense = home_stats.get('goals_conceded_per_game', 1.0)
        lambda_away = (away_attack + home_defense) / 2.0
        
        # Poisson probability of scoring > 0 goals: P(X > 0) = 1 - P(X = 0) = 1 - e^(-lambda)
        prob_home_score = 1 - math.exp(-lambda_home)
        prob_away_score = 1 - math.exp(-lambda_away)
        
        # BTTS Probability (assuming independence)
        btts_prob = prob_home_score * prob_away_score
        
        # Adjust for clean sheet tendencies (if teams are good at keeping clean sheets, reduce prob)
        home_cs_freq = home_stats.get('clean_sheet_frequency', 0)
        away_cs_freq = away_stats.get('clean_sheet_frequency', 0)
        
        # Damping factor based on clean sheets
        # If both keep clean sheets 50% of time, reduce probability significantly
        cs_factor = 1.0 - ((home_cs_freq + away_cs_freq) / 4.0) # Conservative reduction
        btts_prob = btts_prob * cs_factor
        
        # Clamp between 0.05 and 0.95
        btts_prob = max(0.05, min(0.95, btts_prob))
        
        return {
            'model': model_name,
            'btts_probability': float(btts_prob),
            'btts_prediction': btts_prob > 0.5,
            'confidence': max(btts_prob, 1 - btts_prob)
        }
    
    def predict_ensemble(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """Ensemble prediction from both models"""
        lr_result = self.predict_btts_logistic(home_stats, away_stats)
        nn_result = self.predict_btts_neural(home_stats, away_stats)
        
        # If both are heuristic, average them (they might be slightly different if we added noise, 
        # but here they are same, so it's fine)
        
        avg_probability = (lr_result['btts_probability'] + nn_result['btts_probability']) / 2
        ensemble_prediction = avg_probability > 0.5
        
        return {
            'model': 'Ensemble',
            'btts_probability': avg_probability,
            'btts_prediction': bool(ensemble_prediction),
            'confidence': max(avg_probability, 1 - avg_probability),
            'lr_probability': lr_result['btts_probability'],
            'nn_probability': nn_result['btts_probability'],
        }

    def train_aggregated_models(self, df: pd.DataFrame):
        """
        Train models using aggregated dataset.
        Expects engineered columns present:
        Home_xG, Away_xG, Possession_home_pct, ShotsOnTarget_home, ShotsOnTarget_away,
        Corners_home, Corners_away, PassAcc_home_pct, PassAcc_away_pct,
        League_norm, Venue_norm, Referee_norm, BTTS_Target, season_year
        """
        try:
            # Drop rows with missing target
            df_train = df.dropna(subset=['BTTS_Target']).copy()
            # Features
            num_cols = [
                'Home_xG', 'Away_xG',
                'Possession_home_pct', 'ShotsOnTarget_home', 'ShotsOnTarget_away',
                'Corners_home', 'Corners_away',
                'PassAcc_home_pct', 'PassAcc_away_pct',
            ]
            cat_cols = ['League_norm', 'Venue_norm', 'Referee_norm']
            # One-hot encode categorical
            df_enc = pd.get_dummies(df_train[cat_cols], dummy_na=True)
            X = pd.concat([df_train[num_cols].astype(float), df_enc], axis=1)
            y = df_train['BTTS_Target'].astype(int)
            # Split by season: latest season as test
            latest_season = int(df_train['season_year'].max())
            X_train = X[df_train['season_year'] < latest_season]
            y_train = y[df_train['season_year'] < latest_season]
            # Train RF
            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced_subsample'
            )
            rf.fit(X_train, y_train)
            self.agg_rf_model = rf
            joblib.dump(rf, os.path.join(MODEL_DIR, 'aggregated_rf.pkl'))
            logger.info("Aggregated RF model trained and saved")
            # Train XGB if available
            if XGB_AVAILABLE:
                xgb = XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=42
                )
                xgb.fit(X_train, y_train)
                xgb.save_model(os.path.join(MODEL_DIR, 'aggregated_xgb.json'))
                self.agg_xgb_model = xgb
                logger.info("Aggregated XGB model trained and saved")
            return True
        except Exception as e:
            logger.error(f"Failed training aggregated models: {e}")
            return False

    def predict_btts_aggregated(self, features: np.ndarray) -> Dict:
        """
        Predict BTTS using aggregated models.
        Features should match order used in app aggregated endpoint.
        Falls back to RF if XGB unavailable.
        """
        try:
            if self.agg_xgb_model is not None:
                prob = float(self.agg_xgb_model.predict_proba(features)[0][1])
                return {'model': 'XGBoost', 'probability': prob}
            elif self.agg_rf_model is not None:
                prob = float(self.agg_rf_model.predict_proba(features)[0][1])
                return {'model': 'RandomForest', 'probability': prob}
            else:
                return {}
        except Exception as e:
            logger.error(f"Aggregated prediction error: {e}")
            return {}
