from flask import Flask, jsonify
from flask_cors import CORS
from config import CORS_ORIGINS
from data_client import FootballDataClient, FeatureEngineer, AggregatedDataSource
from ml_models import BTTSPredictor
from mysql_db import MySQLDB
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    CORS(app, origins=CORS_ORIGINS)
    football_client = FootballDataClient()
    feature_engineer = FeatureEngineer()
    predictor = BTTSPredictor()
    db = MySQLDB()
    aggregated_source = AggregatedDataSource(os.path.join(os.path.dirname(__file__), 'football_data.aggregated_data.json'))
    app.config['football_client'] = football_client
    app.config['feature_engineer'] = feature_engineer
    app.config['predictor'] = predictor
    app.config['db'] = db
    app.config['aggregated_source'] = aggregated_source
    try:
        from .blueprints.api import api_bp
    except ImportError:
        from blueprints.api import api_bp
    app.register_blueprint(api_bp)
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        return jsonify({'error': 'Internal server error'}), 500
    return app
