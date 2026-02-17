import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from config import FOOTBALL_API_KEY, FOOTBALL_API_BASE_URL, SEASONS
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballDataClient:
    """Client for football-data.org API"""
    
    def __init__(self):
        self.base_url = FOOTBALL_API_BASE_URL
        self.headers = {'X-Auth-Token': FOOTBALL_API_KEY}
    
    def _get(self, endpoint: str, params: Dict = None) -> Dict:
        """Make GET request to API"""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API Error: {e}")
            return {}
    
    def get_team_matches(self, team_id: int, season: str = None, status: str = 'FINISHED') -> List[Dict]:
        """Get team matches for a season"""
        params = {'status': status}
        if season:
            params['season'] = season
        
        data = self._get(f'teams/{team_id}/matches', params)
        return data.get('matches', [])
    
    def get_recent_matches(self, team_id: int, limit: int = 5) -> List[Dict]:
        """Get most recent finished matches for a team regardless of season"""
        params = {'status': 'FINISHED', 'limit': limit}
        # Intentionally no season parameter to get the absolute latest
        data = self._get(f'teams/{team_id}/matches', params)
        return data.get('matches', [])
    
    def get_team_info(self, team_id: int) -> Dict:
        """Get team information including logo"""
        data = self._get(f'teams/{team_id}')
        return {
            'id': data.get('id'),
            'name': data.get('name'),
            'shortName': data.get('shortName'),
            'crest': data.get('crest'),
            'founded': data.get('founded'),
            'venue': data.get('venue'),
            'website': data.get('website'),
        }
    
    def get_competition_teams(self, competition_code: str = 'PL') -> List[Dict]:
        """Get all teams from a competition"""
        data = self._get(f'competitions/{competition_code}/teams')
        teams = []
        for team in data.get('teams', []):
            teams.append({
                'id': team.get('id'),
                'name': team.get('name'),
                'crest': team.get('crest'),
            })
        return teams
    
    def get_upcoming_fixtures(self, competition_code: str = 'PL', days: int = 30) -> List[Dict]:
        """Get upcoming fixtures"""
        date_from = datetime.now().strftime('%Y-%m-%d')
        date_to = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
        
        params = {'dateFrom': date_from, 'dateTo': date_to, 'status': 'SCHEDULED'}
        data = self._get(f'competitions/{competition_code}/matches', params)
        
        fixtures = []
        for match in data.get('matches', []):
            fixtures.append({
                'id': match.get('id'),
                'utcDate': match.get('utcDate'),
                'homeTeam': {
                    'id': match.get('homeTeam', {}).get('id'),
                    'name': match.get('homeTeam', {}).get('name'),
                    'crest': match.get('homeTeam', {}).get('crest'),
                },
                'awayTeam': {
                    'id': match.get('awayTeam', {}).get('id'),
                    'name': match.get('awayTeam', {}).get('name'),
                    'crest': match.get('awayTeam', {}).get('crest'),
                },
                'status': match.get('status'),
            })
        return fixtures
    
    def get_head_to_head(self, team1_id: int, team2_id: int) -> List[Dict]:
        """Get head-to-head matches between two teams"""
        # Fetch last 50 matches for team1 to find H2H games
        # We manually filter because API doesn't support 'opponents' filter reliably
        data = self._get(f'teams/{team1_id}/matches', 
                        {'status': 'FINISHED', 'limit': 50})
        
        all_matches = data.get('matches', [])
        h2h_matches = []
        
        for match in all_matches:
            home_id = match.get('homeTeam', {}).get('id')
            away_id = match.get('awayTeam', {}).get('id')
            
            # Check if match involves both teams
            if (home_id == team1_id and away_id == team2_id) or \
               (home_id == team2_id and away_id == team1_id):
                h2h_matches.append(match)
                
        return h2h_matches

    def get_standings(self, competition_code: str) -> List[Dict]:
        """Get league standings/table for a competition"""
        data = self._get(f'competitions/{competition_code}/standings')
        standings = []
        for group in data.get('standings', []):
            group_type = group.get('type', '')
            group_name = group.get('group', '')
            if group_type == 'TOTAL':
                for entry in group.get('table', []):
                    team = entry.get('team', {})
                    standings.append(self._standings_entry(entry, team, group_name))
                break
            elif group_type == 'GROUP':
                for entry in group.get('table', []):
                    team = entry.get('team', {})
                    standings.append(self._standings_entry(entry, team, group_name))
        return standings

    def _standings_entry(self, entry: Dict, team: Dict, group: str = '') -> Dict:
        return {
            'position': entry.get('position'),
            'group': group,
            'team_id': team.get('id'),
            'team_name': team.get('name'),
            'crest': team.get('crest'),
            'played': entry.get('playedGames', 0),
            'won': entry.get('won', 0),
            'draw': entry.get('draw', 0),
            'lost': entry.get('lost', 0),
            'goals_for': entry.get('goalsFor', 0),
            'goals_against': entry.get('goalsAgainst', 0),
            'goal_difference': entry.get('goalDifference', 0),
            'points': entry.get('points', 0),
        }


class FeatureEngineer:
    """Calculate advanced football statistics and features"""
    
    @staticmethod
    def calculate_team_stats(matches: List[Dict], team_id: int, is_home: bool = None) -> Dict:
        """
        Calculate comprehensive team statistics
        team_id: ID of team to calculate stats for
        is_home: None=all, True=home only, False=away only
        """
        if not matches:
            return {}
        
        filtered_matches = []
        for match in matches:
            if is_home is None:
                filtered_matches.append(match)
            elif is_home and match.get('homeTeam', {}).get('id') == team_id:
                filtered_matches.append(match)
            elif not is_home and match.get('awayTeam', {}).get('id') == team_id:
                filtered_matches.append(match)
        
        if not filtered_matches:
            return {}
        
        # Extract goals
        goals_for = []
        goals_against = []
        shots_on_target = []
        corners = []
        clean_sheets = 0
        wins = 0
        
        for match in filtered_matches:
            is_team_home = match.get('homeTeam', {}).get('id') == team_id
            
            if is_team_home:
                gf = match.get('score', {}).get('fullTime', {}).get('home', 0)
                ga = match.get('score', {}).get('fullTime', {}).get('away', 0)
            else:
                gf = match.get('score', {}).get('fullTime', {}).get('away', 0)
                ga = match.get('score', {}).get('fullTime', {}).get('home', 0)
            
            goals_for.append(gf)
            goals_against.append(ga)
            
            if ga == 0:
                clean_sheets += 1
            if gf > ga:
                wins += 1
        
        matches_played = len(filtered_matches)
        
        return {
            'matches_played': matches_played,
            'goals_for': sum(goals_for),
            'goals_against': sum(goals_against),
            'avg_goals_for': np.mean(goals_for),
            'avg_goals_against': np.mean(goals_against),
            'goals_per_game': sum(goals_for) / matches_played if matches_played > 0 else 0,
            'goals_conceded_per_game': sum(goals_against) / matches_played if matches_played > 0 else 0,
            'clean_sheets': clean_sheets,
            'clean_sheet_frequency': clean_sheets / matches_played if matches_played > 0 else 0,
            'defensive_fragility_index': sum(goals_against) / max(clean_sheets, 1),
            'goals_conceded': sum(goals_against),
            'win_rate': wins / matches_played if matches_played > 0 else 0,
        }
    
    @staticmethod
    def calculate_btts_likelihood(home_stats: Dict, away_stats: Dict) -> Dict:
        """Calculate BTTS (Both Teams To Score) probability indicators"""
        
        home_scoring = home_stats.get('goals_per_game', 0)
        home_conceding = home_stats.get('goals_conceded_per_game', 0)
        
        away_scoring = away_stats.get('goals_per_game', 0)
        away_conceding = away_stats.get('goals_conceded_per_game', 0)
        
        # DFI (Defensive Fragility Index): higher = more goals conceded
        home_dfi = home_stats.get('defensive_fragility_index', 0)
        away_dfi = away_stats.get('defensive_fragility_index', 0)
        
        # BTTS indicators
        btts_indicators = {
            'home_scoring_rate': home_scoring,
            'home_conceding_rate': home_conceding,
            'away_scoring_rate': away_scoring,
            'away_conceding_rate': away_conceding,
            'home_dfi': home_dfi,
            'away_dfi': away_dfi,
            'combined_dfi': (home_dfi + away_dfi) / 2,
            'home_clean_sheet_freq': home_stats.get('clean_sheet_frequency', 0),
            'away_clean_sheet_freq': away_stats.get('clean_sheet_frequency', 0),
            'expected_total_goals': (home_scoring + away_scoring + home_conceding + away_conceding) / 2,
        }
        
        return btts_indicators


class AggregatedDataSource:
    """Load and engineer features from aggregated JSON dataset"""
    def __init__(self, path: str):
        self.path = path
        self.df = self._load_json(path)
        # Ensure parsed numeric columns exist
        self._prepare_base_columns()
        # Precompute BTTS target
        self.df['BTTS_Target'] = self.df['Score'].apply(self._btts_from_score)
        # Attach season_year for splitting
        self.df['season_year'] = self.df['season'].astype(str).str.extract(r'(\d{4})').fillna('0').astype(int)
        # Compute rolling BTTS form per team
        self._compute_form_btts()

    def _load_json(self, path: str) -> pd.DataFrame:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            # Normalize nested dicts to make access easier
            df['team_stats'] = df.get('team_stats', {}).where(df['team_stats'].notnull(), None)
            df['team_stats_extra'] = df.get('team_stats_extra', {}).where(df['team_stats_extra'].notnull(), None)
            return df
        except Exception as e:
            logger.error(f"Failed to load aggregated dataset: {e}")
            return pd.DataFrame()

    @staticmethod
    def _btts_from_score(score: str) -> bool:
        try:
            # Score like "0–3" (en dash)
            parts = score.replace('-', '–').split('–')
            home = int(parts[0].strip())
            away = int(parts[1].strip())
            return (home > 0) and (away > 0)
        except Exception:
            return False

    @staticmethod
    def _parse_percent(val: str) -> float:
        try:
            if val is None or val == '':
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            # extract percentage number in string like "35%" or "313 of 395 79%"
            if '%' in val:
                # take the last percentage occurrence
                nums = [s for s in val.split() if '%' in s]
                if nums:
                    return float(nums[-1].replace('%', '').strip())
            # plain number
            return float(val)
        except Exception:
            return np.nan

    @staticmethod
    def _parse_count_of(val: str) -> Dict[str, float]:
        """
        Parse strings like "47% 8 of 17" -> {'count': 8, 'total': 17, 'percent': 47}
        """
        try:
            if val is None or val == '':
                return {'count': np.nan, 'total': np.nan, 'percent': np.nan}
            if isinstance(val, (int, float)):
                return {'count': float(val), 'total': np.nan, 'percent': np.nan}
            percent = AggregatedDataSource._parse_percent(val)
            # find "x of y"
            tokens = val.replace('%', '').split()
            if 'of' in tokens:
                idx = tokens.index('of')
                count = float(tokens[idx - 1])
                total = float(tokens[idx + 1])
            else:
                count, total = np.nan, np.nan
            return {'count': count, 'total': total, 'percent': percent}
        except Exception:
            return {'count': np.nan, 'total': np.nan, 'percent': np.nan}

    def _prepare_base_columns(self):
        if self.df.empty:
            return
        # xG numeric
        self.df['Home_xG'] = pd.to_numeric(self.df.get('Home_xG'), errors='coerce')
        self.df['Away_xG'] = pd.to_numeric(self.df.get('Away_xG'), errors='coerce')
        # Odds numeric
        self.df['Odd_Home'] = pd.to_numeric(self.df.get('Odd_Home'), errors='coerce')
        self.df['Odd_Away'] = pd.to_numeric(self.df.get('Odd_Away'), errors='coerce')
        self.df['Odds_Draw'] = pd.to_numeric(self.df.get('Odds_Draw'), errors='coerce')
        # Date to datetime
        self.df['Date'] = pd.to_datetime(self.df.get('Date'), errors='coerce')
        # Possession %, Passing Accuracy %
        poss_home = []
        poss_away = []
        passacc_home = []
        passacc_away = []
        sot_home = []
        sot_away = []
        corners_home = []
        corners_away = []
        saves_home = []
        saves_away = []
        tackles_home = []
        tackles_away = []
        interceptions_home = []
        interceptions_away = []
        clearances_home = []
        clearances_away = []
        touches_home = []
        touches_away = []
        crosses_home = []
        crosses_away = []
        leagues = []
        venues = []
        referees = []
        for _, row in self.df.iterrows():
            ts = row.get('team_stats') or {}
            tse = row.get('team_stats_extra') or {}
            # Possession
            ph = self._parse_percent((ts.get('Possession') or {}).get('home_value'))
            pa = self._parse_percent((ts.get('Possession') or {}).get('away_value'))
            # Passing Accuracy
            pash = self._parse_percent((ts.get('Passing Accuracy') or {}).get('home_value'))
            pasa = self._parse_percent((ts.get('Passing Accuracy') or {}).get('away_value'))
            # Shots on Target (count)
            soth = self._parse_count_of((ts.get('Shots on Target') or {}).get('home_value')).get('count')
            sota = self._parse_count_of((ts.get('Shots on Target') or {}).get('away_value')).get('count')
            # Corners
            ch = pd.to_numeric((tse.get('Corners') or {}).get('home_value'), errors='coerce')
            ca = pd.to_numeric((tse.get('Corners') or {}).get('away_value'), errors='coerce')
            # Saves (count)
            savh = self._parse_count_of((ts.get('Saves') or {}).get('home_value')).get('count')
            sava = self._parse_count_of((ts.get('Saves') or {}).get('away_value')).get('count')
            # Defensive actions
            th = pd.to_numeric((tse.get('Tackles') or {}).get('home_value'), errors='coerce')
            ta = pd.to_numeric((tse.get('Tackles') or {}).get('away_value'), errors='coerce')
            ih = pd.to_numeric((tse.get('Interceptions') or {}).get('home_value'), errors='coerce')
            ia = pd.to_numeric((tse.get('Interceptions') or {}).get('away_value'), errors='coerce')
            clh = pd.to_numeric((tse.get('Clearances') or {}).get('home_value'), errors='coerce')
            cla = pd.to_numeric((tse.get('Clearances') or {}).get('away_value'), errors='coerce')
            # Touches (proxy for final third presence)
            toh = pd.to_numeric((tse.get('Touches') or {}).get('home_value'), errors='coerce')
            toa = pd.to_numeric((tse.get('Touches') or {}).get('away_value'), errors='coerce')
            # Crosses
            crh = pd.to_numeric((tse.get('Crosses') or {}).get('home_value'), errors='coerce')
            cra = pd.to_numeric((tse.get('Crosses') or {}).get('away_value'), errors='coerce')
            poss_home.append(ph)
            poss_away.append(pa)
            passacc_home.append(pash)
            passacc_away.append(pasa)
            sot_home.append(soth)
            sot_away.append(sota)
            corners_home.append(ch)
            corners_away.append(ca)
            saves_home.append(savh)
            saves_away.append(sava)
            tackles_home.append(th)
            tackles_away.append(ta)
            interceptions_home.append(ih)
            interceptions_away.append(ia)
            clearances_home.append(clh)
            clearances_away.append(cla)
            touches_home.append(toh)
            touches_away.append(toa)
            crosses_home.append(crh)
            crosses_away.append(cra)
            leagues.append(row.get('League') or row.get('league') or '')
            venues.append(row.get('Venue') or '')
            referees.append(row.get('Referee') or '')
        # assign columns
        self.df['Possession_home_pct'] = pd.Series(poss_home)
        self.df['Possession_away_pct'] = pd.Series(poss_away)
        self.df['PassAcc_home_pct'] = pd.Series(passacc_home)
        self.df['PassAcc_away_pct'] = pd.Series(passacc_away)
        self.df['ShotsOnTarget_home'] = pd.Series(sot_home)
        self.df['ShotsOnTarget_away'] = pd.Series(sot_away)
        self.df['Corners_home'] = pd.Series(corners_home)
        self.df['Corners_away'] = pd.Series(corners_away)
        self.df['Saves_home'] = pd.Series(saves_home)
        self.df['Saves_away'] = pd.Series(saves_away)
        self.df['Tackles_home'] = pd.Series(tackles_home)
        self.df['Tackles_away'] = pd.Series(tackles_away)
        self.df['Interceptions_home'] = pd.Series(interceptions_home)
        self.df['Interceptions_away'] = pd.Series(interceptions_away)
        self.df['Clearances_home'] = pd.Series(clearances_home)
        self.df['Clearances_away'] = pd.Series(clearances_away)
        self.df['Touches_home'] = pd.Series(touches_home)
        self.df['Touches_away'] = pd.Series(touches_away)
        self.df['Crosses_home'] = pd.Series(crosses_home)
        self.df['Crosses_away'] = pd.Series(crosses_away)
        self.df['League_norm'] = pd.Series(leagues).astype(str)
        self.df['Venue_norm'] = pd.Series(venues).astype(str)
        self.df['Referee_norm'] = pd.Series(referees).astype(str)
        # Derived features
        self.df['xG_Combined'] = (self.df['Home_xG'] + self.df['Away_xG']).astype(float)
        # Defensive Weakness: 1 - (Saves / Opponent SOT)
        self.df['DefWeak_home'] = 1.0 - (self.df['Saves_home'] / self.df['ShotsOnTarget_away']).replace([np.inf, -np.inf], np.nan)
        self.df['DefWeak_away'] = 1.0 - (self.df['Saves_away'] / self.df['ShotsOnTarget_home']).replace([np.inf, -np.inf], np.nan)
        # Pressure Index (simple weighted sum, scaled)
        raw_home = self.df['Corners_home'].fillna(0) + self.df['Crosses_home'].fillna(0) + (self.df['Touches_home'].fillna(0) / 10.0)
        raw_away = self.df['Corners_away'].fillna(0) + self.df['Crosses_away'].fillna(0) + (self.df['Touches_away'].fillna(0) / 10.0)
        # min-max scale within dataset
        def minmax(series: pd.Series) -> pd.Series:
            mn = series.min()
            mx = series.max()
            if pd.isna(mn) or pd.isna(mx) or mx == mn:
                return pd.Series(np.zeros(len(series)))
            return (series - mn) / (mx - mn)
        self.df['PressureIndex_home'] = minmax(raw_home)
        self.df['PressureIndex_away'] = minmax(raw_away)
        self.df['Attacking_Pressure'] = (self.df['PressureIndex_home'] + self.df['PressureIndex_away']) / 2.0

    def _compute_form_btts(self):
        if self.df.empty:
            return
        # sort by Date
        self.df = self.df.sort_values(by=['Date'])
        # compute rolling BTTS per team
        for side in ['Home', 'Away']:
            team_col = side
            form_col = f'Form_BTTS_{side.lower()}'
            vals = []
            # Use a dict of past outcomes per team
            history = {}
            for _, row in self.df.iterrows():
                team = row.get(team_col)
                btts = bool(row.get('BTTS_Target'))
                # current rolling average BEFORE including this match
                past = history.get(team, [])
                if len(past) == 0:
                    vals.append(np.nan)
                else:
                    vals.append(np.mean(past[-5:]))
                # update history
                past.append(btts)
                history[team] = past
            self.df[form_col] = pd.Series(vals)
        # Combined Form_BTTS
        self.df['Form_BTTS'] = (self.df['Form_BTTS_home'] + self.df['Form_BTTS_away']) / 2.0

    @staticmethod
    def poisson_btts_prob(home_xg: float, away_xg: float) -> float:
        try:
            if home_xg is None or away_xg is None:
                return 0.5
            # P(team scores >=1) = 1 - e^{-xG}
            p_home = 1 - np.exp(-float(home_xg))
            p_away = 1 - np.exp(-float(away_xg))
            prob = p_home * p_away
            # clamp
            return float(max(0.05, min(0.95, prob)))
        except Exception:
            return 0.5

    @staticmethod
    def implied_prob(odds: float) -> float:
        try:
            if not odds or odds <= 0:
                return np.nan
            return 1.0 / float(odds)
        except Exception:
            return np.nan

    def value_gaps(self, row: pd.Series, btts_prob: float) -> Dict[str, float]:
        draw_p = self.implied_prob(row.get('Odds_Draw'))
        home_p = self.implied_prob(row.get('Odd_Home'))
        away_p = self.implied_prob(row.get('Odd_Away'))
        return {
            'gap_vs_draw': (btts_prob - draw_p) if not np.isnan(draw_p) else np.nan,
            'gap_vs_home': (btts_prob - home_p) if not np.isnan(home_p) else np.nan,
            'gap_vs_away': (btts_prob - away_p) if not np.isnan(away_p) else np.nan,
        }

    def volatility_multiplier(self, row: pd.Series) -> float:
        """
        Volatility Score based on team_stats_extra:
        - Increase if tackles and interceptions are low for both teams
        - Increase if clearances are high and possession is skewed
        - Increase if both xG > 1.2
        Produces multiplier in [0.9, 1.2]
        """
        try:
            mult = 1.0
            # defensive organization low
            t_low = (row.get('Tackles_home', 0) < 12) and (row.get('Tackles_away', 0) < 12)
            i_low = (row.get('Interceptions_home', 0) < 6) and (row.get('Interceptions_away', 0) < 6)
            if t_low and i_low:
                mult += 0.05
            # siege scenario: clearances high and possession skewed
            clear_high = (row.get('Clearances_home', 0) > 25) or (row.get('Clearances_away', 0) > 25)
            poss_skew = abs((row.get('Possession_home_pct', 50) - row.get('Possession_away_pct', 50))) > 20
            if clear_high and poss_skew:
                mult += 0.05
            # high expected goals
            if (row.get('Home_xG', 0) > 1.2) and (row.get('Away_xG', 0) > 1.2):
                mult += 0.05
            return float(max(0.9, min(1.2, mult)))
        except Exception:
            return 1.0

    def get_match_by_unique_id(self, unique_id: str) -> Optional[pd.Series]:
        if self.df.empty:
            return None
        try:
            row = self.df.loc[self.df['unique_id'] == unique_id]
            if row.empty:
                return None
            return row.iloc[0]
        except Exception:
            return None

    def team_recent_average(self, team_name: str, last_n: int = 10) -> Dict[str, float]:
        """
        Average key stats from last N matches for given team
        """
        if self.df.empty or not team_name:
            return {}
        # Matches where team is Home or Away
        df_team = self.df[(self.df['Home'] == team_name) | (self.df['Away'] == team_name)].sort_values('Date', ascending=False).head(last_n)
        if df_team.empty:
            return {}
        # Compute averages for core features
        metrics = {
            'Home_xG': df_team['Home_xG'].mean(),
            'Away_xG': df_team['Away_xG'].mean(),
            'Possession_home_pct': df_team['Possession_home_pct'].mean(),
            'Possession_away_pct': df_team['Possession_away_pct'].mean(),
            'ShotsOnTarget_home': df_team['ShotsOnTarget_home'].mean(),
            'ShotsOnTarget_away': df_team['ShotsOnTarget_away'].mean(),
            'Corners_home': df_team['Corners_home'].mean(),
            'Corners_away': df_team['Corners_away'].mean(),
            'PassAcc_home_pct': df_team['PassAcc_home_pct'].mean(),
            'PassAcc_away_pct': df_team['PassAcc_away_pct'].mean(),
            'PressureIndex_home': df_team['PressureIndex_home'].mean(),
            'PressureIndex_away': df_team['PressureIndex_away'].mean(),
            'Form_BTTS': df_team['Form_BTTS'].mean(),
        }
        return {k: (0.0 if (v is None or np.isnan(v)) else float(v)) for k, v in metrics.items()}

    def latest_season(self) -> int:
        if self.df.empty:
            return 0
        return int(self.df['season_year'].max())
