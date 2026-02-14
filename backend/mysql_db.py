import logging
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from mysql.connector import pooling, Error, connect
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MySQLDB:
    """MySQL database operations"""

    def __init__(self):
        try:
            self._ensure_schema()
            self.pool = pooling.MySQLConnectionPool(
                pool_name="btts_pool",
                pool_size=5,
                host=MYSQL_HOST or "127.0.0.1",
                port=int(MYSQL_PORT or 3306),
                user=MYSQL_USER or "root",
                password=MYSQL_PASSWORD or "",
                database=MYSQL_DB or "btts_db",
                charset="utf8mb4",
                autocommit=True
            )
            logger.info("MySQL connection pool initialized")
        except Error as e:
            logger.error(f"Failed to initialize MySQL pool: {e}")
            self.pool = None
            try:
                self._ensure_schema()
                self.pool = pooling.MySQLConnectionPool(
                    pool_name="btts_pool",
                    pool_size=5,
                    host=MYSQL_HOST or "127.0.0.1",
                    port=int(MYSQL_PORT or 3306),
                    user=MYSQL_USER or "root",
                    password=MYSQL_PASSWORD or "",
                    database=MYSQL_DB or "btts_db",
                    charset="utf8mb4",
                    autocommit=True
                )
                logger.info("MySQL schema ensured and connection pool re-initialized")
            except Error as e2:
                logger.error(f"Failed to ensure schema or reinitialize pool: {e2}")
                self.pool = None

    def _get_conn(self):
        if not self.pool:
            raise RuntimeError("MySQL pool not initialized")
        return self.pool.get_connection()

    def _ensure_schema(self):
        """Ensure database and tables exist by executing setup SQL script"""
        try:
            conn = connect(
                host=MYSQL_HOST or "127.0.0.1",
                port=int(MYSQL_PORT or 3306),
                user=MYSQL_USER or "root",
                password=MYSQL_PASSWORD or "",
                charset="utf8mb4",
                autocommit=True
            )
            cur = conn.cursor()
            base_dir = os.path.dirname(os.path.dirname(__file__))
            sql_path = os.path.join(base_dir, 'scripts', 'setup-mysql.sql')
            if os.path.exists(sql_path):
                with open(sql_path, 'r', encoding='utf-8') as f:
                    sql = f.read()
                for stmt in [s.strip() for s in sql.split(';') if s.strip()]:
                    try:
                        cur.execute(stmt)
                    except Error as e_stmt:
                        if getattr(e_stmt, 'errno', None) == 1061:
                            logger.info(f"Ignoring duplicate index: {e_stmt}")
                            continue
                        raise
                logger.info("Executed setup-mysql.sql to ensure schema")
            cur.close()
            conn.close()
        except Error as e:
            logger.error(f"Error ensuring schema: {e}")

    def cache_team_stats(self, team_id: int, season: str, stats: Dict):
        """Cache team statistics"""
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO team_statistics (team_id, season, stats, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE stats = VALUES(stats), updated_at = NOW()
                """,
                (team_id, season, json.dumps(stats))
            )
            cur.close()
            conn.close()
            logger.info(f"Cached stats for team {team_id} season {season}")
        except Error as e:
            logger.error(f"Error caching team stats: {e}")

    def upsert_team(self, team_id: int, name: Optional[str] = None,
                    short_name: Optional[str] = None, crest: Optional[str] = None,
                    competition_code: Optional[str] = None):
        """Ensure a team exists in the teams table"""
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO teams (id, name, short_name, crest, competition_code)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  name = VALUES(name),
                  short_name = VALUES(short_name),
                  crest = VALUES(crest),
                  competition_code = VALUES(competition_code)
                """,
                (team_id, name, short_name, crest, competition_code)
            )
            cur.close()
            conn.close()
        except Error as e:
            logger.error(f"Error upserting team {team_id}: {e}")

    def get_cached_team_stats(self, team_id: int, season: str) -> Optional[Dict]:
        """Get cached team statistics"""
        try:
            conn = self._get_conn()
            cur = conn.cursor(dictionary=True)
            cur.execute(
                """
                SELECT stats FROM team_statistics
                WHERE team_id = %s AND season = %s
                """,
                (team_id, season)
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row and row.get('stats'):
                return row['stats'] if isinstance(row['stats'], dict) else json.loads(row['stats'])
            return None
        except Error as e:
            logger.error(f"Error fetching cached stats: {e}")
            return None

    def store_prediction(self, home_team_id: int, away_team_id: int,
                         fixture_id: Optional[int], prediction: Dict, odds: Optional[float] = None):
        """Store prediction in database; upsert on fixture_id when provided"""
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            fixture_val = fixture_id if fixture_id else None
            cur.execute(
                """
                INSERT INTO predictions
                  (fixture_id, home_team_id, away_team_id, btts_probability, btts_prediction,
                   confidence, model_type, odds, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON DUPLICATE KEY UPDATE
                   btts_probability = VALUES(btts_probability),
                   btts_prediction = VALUES(btts_prediction),
                   confidence = VALUES(confidence),
                   model_type = VALUES(model_type),
                   odds = VALUES(odds),
                   updated_at = NOW()
                """,
                (
                    fixture_val,
                    home_team_id,
                    away_team_id,
                    float(prediction.get('btts_probability') or 0),
                    1 if prediction.get('btts_prediction') else 0,
                    float(prediction.get('confidence') or 0),
                    str(prediction.get('model') or 'ensemble'),
                    odds
                )
            )
            cur.close()
            conn.close()
            logger.info(f"Stored prediction (fixture_id={fixture_val})")
        except Error as e:
            logger.error(f"Error storing prediction: {e}")

    def get_team_by_name(self, name: str) -> Optional[Dict]:
        """Get team info by name"""
        try:
            conn = self._get_conn()
            cur = conn.cursor(dictionary=True)
            like = f"%{name}%"
            cur.execute(
                """
                SELECT * FROM teams
                WHERE name LIKE %s
                LIMIT 1
                """,
                (like,)
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
            return row
        except Error as e:
            logger.error(f"Error fetching team: {e}")
            return None

    def cache_upcoming_fixtures(self, fixtures: List[Dict], competition_code: Optional[str] = None):
        """Cache upcoming fixtures"""
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            for fixture in fixtures:
                home = fixture.get('homeTeam', {}) or {}
                away = fixture.get('awayTeam', {}) or {}
                try:
                    if home.get('id'):
                        self.upsert_team(
                            home.get('id'),
                            home.get('name'), None, home.get('crest'), competition_code
                        )
                    if away.get('id'):
                        self.upsert_team(
                            away.get('id'),
                            away.get('name'), None, away.get('crest'), competition_code
                        )
                except Exception as e:
                    logger.info(f"Skipping team upsert during fixtures cache: {e}")
                cur.execute(
                    """
                    INSERT INTO fixtures (fixture_id, competition_code, home_team_id, away_team_id, utc_date, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON DUPLICATE KEY UPDATE
                      competition_code = VALUES(competition_code),
                      home_team_id = VALUES(home_team_id),
                      away_team_id = VALUES(away_team_id),
                      utc_date = VALUES(utc_date),
                      status = VALUES(status)
                    """,
                    (
                        fixture.get('id'),
                        competition_code,
                        fixture.get('homeTeam', {}).get('id'),
                        fixture.get('awayTeam', {}).get('id'),
                        datetime.fromisoformat(fixture.get('utcDate').replace('Z', '+00:00')) if fixture.get('utcDate') else None,
                        'SCHEDULED'
                    )
                )
            cur.close()
            conn.close()
            logger.info(f"Cached {len(fixtures)} upcoming fixtures")
        except Error as e:
            logger.error(f"Error caching fixtures: {e}")

    def get_prediction_history(self, limit: int = 100) -> List[Dict]:
        """Get prediction history"""
        try:
            conn = self._get_conn()
            cur = conn.cursor(dictionary=True)
            cur.execute(
                """
                SELECT id, fixture_id, home_team_id, away_team_id,
                       btts_probability, btts_prediction, confidence,
                       model_type, odds, created_at
                FROM predictions
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (int(limit),)
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return rows or []
        except Error as e:
            logger.error(f"Error fetching prediction history: {e}")
            return []

    def get_prediction_count(self) -> int:
        """Get total predictions count"""
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM predictions")
            (count,) = cur.fetchone()
            cur.close()
            conn.close()
            return int(count or 0)
        except Error as e:
            logger.error(f"Error fetching prediction count: {e}")
            return 0
