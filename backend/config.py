import os
from dotenv import load_dotenv

load_dotenv()

# Football-data.org API
FOOTBALL_API_KEY = os.getenv('FOOTBALL_API_KEY')
FOOTBALL_API_BASE_URL = 'https://api.football-data.org/v4'

# Supabase
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# MySQL
MYSQL_HOST = os.getenv('MYSQL_HOST', '127.0.0.1')
MYSQL_PORT = os.getenv('MYSQL_PORT', '3306')
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
MYSQL_DB = os.getenv('MYSQL_DB', 'btts_db')

# Flask
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
DEBUG = FLASK_ENV == 'development'

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
LR_MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_btts.pkl')
NN_MODEL_PATH = os.path.join(MODEL_DIR, 'neural_network_btts.h5')

# Seasons for data aggregation
SEASONS = ['2023', '2024', '2025']

# CORS settings
_cors_env = os.getenv('CORS_ORIGINS', '')
if _cors_env:
    CORS_ORIGINS = [x.strip() for x in _cors_env.split(',') if x.strip()]
else:
    CORS_ORIGINS = ['*']
