from flask import Blueprint, jsonify, request, current_app
from config import FOOTBALL_API_KEY
import logging
import numpy as np

logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__)

def _clients():
    app = current_app
    return (
        app.config['football_client'],
        app.config['feature_engineer'],
        app.config['predictor'],
        app.config['db'],
        app.config['aggregated_source'],
        app.config['ai_analyzer'],
    )

def get_ai_analyzer():
    return current_app.config.get('ai_analyzer')

@api_bp.route('/api/test', methods=['GET'])
def test():
    try:
        if not FOOTBALL_API_KEY or FOOTBALL_API_KEY == 'your_api_key_here':
            return jsonify({
                'status': 'FAILED',
                'message': 'FOOTBALL_API_KEY is not configured or is still a placeholder',
                'instruction': 'Please update /backend/.env with your actual API key from https://www.football-data.org/client'
            }), 400
        football_client, _, _, _, _ = _clients()
        test_response = football_client._get('competitions')
        if test_response:
            return jsonify({
                'status': 'SUCCESS',
                'message': 'API key is valid and connected to football-data.org',
                'competitions_available': len(test_response.get('competitions', []))
            })
        else:
            return jsonify({
                'status': 'FAILED',
                'message': 'API key might be invalid or API is unreachable'
            }), 400
    except Exception as e:
        return jsonify({
            'status': 'ERROR',
            'message': str(e)
        }), 500

@api_bp.route('/api/teams', methods=['GET'])
def get_teams():
    competition = request.args.get('competition', 'PL')
    try:
        if not FOOTBALL_API_KEY or FOOTBALL_API_KEY == 'your_api_key_here':
            return jsonify({
                'error': 'FOOTBALL_API_KEY not configured',
                'message': 'Please set FOOTBALL_API_KEY in /backend/.env',
                'teams': []
            }), 400
        football_client, _, _, _, _ = _clients()
        logger.info(f"Fetching teams for competition: {competition}")
        teams = football_client.get_competition_teams(competition)
        logger.info(f"Successfully fetched {len(teams)} teams")
        return jsonify({'teams': teams, 'count': len(teams)})
    except Exception as e:
        logger.error(f"Error fetching teams: {e}")
        return jsonify({'error': str(e), 'teams': []}), 400

@api_bp.route('/api/team/<int:team_id>/stats', methods=['GET'])
def get_team_stats(team_id):
    try:
        season = request.args.get('season', '2024')
        football_client, feature_engineer, _, db, _ = _clients()
        cached = db.get_cached_team_stats(team_id, season)
        if cached:
            return jsonify({'stats': cached, 'source': 'cache'})
        team_info = football_client.get_team_info(team_id)
        matches = football_client.get_team_matches(team_id, season)
        all_stats = feature_engineer.calculate_team_stats(matches, team_id, is_home=None)
        home_stats = feature_engineer.calculate_team_stats(matches, team_id, is_home=True)
        away_stats = feature_engineer.calculate_team_stats(matches, team_id, is_home=False)
        stats = {
            'team_id': team_id,
            'team_name': team_info.get('name'),
            'crest': team_info.get('crest'),
            'season': season,
            'all_matches': all_stats,
            'home_matches': home_stats,
            'away_matches': away_stats,
        }
        db.cache_team_stats(team_id, season, stats)
        return jsonify({'stats': stats, 'source': 'api'})
    except Exception as e:
        logger.error(f"Error fetching team stats: {e}")
        return jsonify({'error': str(e)}), 400

@api_bp.route('/api/predict/btts', methods=['POST'])
def predict_btts():
    try:
        data = request.json
        home_team_id = data.get('home_team_id')
        away_team_id = data.get('away_team_id')
        fixture_id = data.get('fixture_id')
        season = data.get('season', '2024')
        model_type = data.get('model', 'ensemble')
        user_id = data.get('user_id')
        football_client, feature_engineer, predictor, db, _ = _clients()
        home_matches = football_client.get_team_matches(home_team_id, season)
        away_matches = football_client.get_team_matches(away_team_id, season)
        home_stats = feature_engineer.calculate_team_stats(home_matches, home_team_id, is_home=None)
        away_stats = feature_engineer.calculate_team_stats(away_matches, away_team_id, is_home=None)
        home_recent = football_client.get_recent_matches(home_team_id, limit=5)
        away_recent = football_client.get_recent_matches(away_team_id, limit=5)
        home_last_5 = sorted(home_recent, key=lambda x: x['utcDate'], reverse=True)
        away_last_5 = sorted(away_recent, key=lambda x: x['utcDate'], reverse=True)
        h2h_matches = football_client.get_head_to_head(home_team_id, away_team_id)
        h2h_last_5 = sorted(h2h_matches, key=lambda x: x['utcDate'], reverse=True)[:5]
        btts_indicators = feature_engineer.calculate_btts_likelihood(home_stats, away_stats)
        if model_type == 'lr':
            prediction = predictor.predict_btts_logistic(home_stats, away_stats)
        elif model_type == 'nn':
            prediction = predictor.predict_btts_neural(home_stats, away_stats)
        else:
            prediction = predictor.predict_ensemble(home_stats, away_stats)
        if prediction.get('lr_probability') is None and model_type == 'lr':
            prediction['lr_probability'] = prediction.get('btts_probability')
        if prediction.get('nn_probability') is None and model_type == 'nn':
            prediction['nn_probability'] = prediction.get('btts_probability')
        prediction['model'] = model_type
        try:
            home_info = football_client.get_team_info(home_team_id)
            away_info = football_client.get_team_info(away_team_id)
            db.upsert_team(home_team_id, home_info.get('name'), home_info.get('shortName'), home_info.get('crest'), None)
            db.upsert_team(away_team_id, away_info.get('name'), away_info.get('shortName'), away_info.get('crest'), None)
        except Exception as e:
            logger.info(f"Team upsert skipped: {e}")
        db.store_prediction(home_team_id, away_team_id, fixture_id, prediction, user_id=user_id)
        return jsonify({
            'prediction': prediction,
            'indicators': btts_indicators,
            'home_stats': home_stats,
            'away_stats': away_stats,
            'home_last_5': home_last_5,
            'away_last_5': away_last_5,
            'h2h_matches': h2h_last_5
        })
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 400

@api_bp.route('/api/fixtures/upcoming', methods=['GET'])
def get_upcoming_fixtures():
    try:
        competition = request.args.get('competition', 'PL')
        days = int(request.args.get('days', 30))
        football_client, _, _, db, _ = _clients()
        fixtures = football_client.get_upcoming_fixtures(competition, days)
        db.cache_upcoming_fixtures(fixtures, competition)
        return jsonify({'fixtures': fixtures, 'count': len(fixtures)})
    except Exception as e:
        logger.error(f"Error fetching fixtures: {e}")
        return jsonify({'error': str(e)}), 400

@api_bp.route('/api/predictions/history', methods=['GET'])
def get_predictions_history():
    try:
        limit = int(request.args.get('limit', 100))
        user_id = request.args.get('user_id')
        football_client, _, _, db, _ = _clients()
        predictions = db.get_prediction_history(limit, user_id=user_id) or []
        enriched = []
        for row in predictions:
            home_id = row.get('home_team_id')
            away_id = row.get('away_team_id')
            hname = row.get('home_team_name')
            aname = row.get('away_team_name')
            hcrest = row.get('home_crest')
            acrest = row.get('away_crest')
            if not hname or not hcrest:
                try:
                    info = football_client.get_team_info(home_id)
                    hname = hname or info.get('name')
                    hcrest = hcrest or info.get('crest')
                    try:
                        db.upsert_team(home_id, info.get('name'), info.get('shortName'), info.get('crest'), None)
                    except Exception:
                        pass
                except Exception:
                    pass
            if not aname or not acrest:
                try:
                    info = football_client.get_team_info(away_id)
                    aname = aname or info.get('name')
                    acrest = acrest or info.get('crest')
                    try:
                        db.upsert_team(away_id, info.get('name'), info.get('shortName'), info.get('crest'), None)
                    except Exception:
                        pass
                except Exception:
                    pass
            row['home_team_name'] = hname or row.get('home_team_name') or f"Team {home_id}"
            row['away_team_name'] = aname or row.get('away_team_name') or f"Team {away_id}"
            row['home_crest'] = hcrest or row.get('home_crest')
            row['away_crest'] = acrest or row.get('away_crest')
            enriched.append(row)
        return jsonify({'predictions': enriched, 'count': len(enriched)})
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        return jsonify({'error': str(e)}), 400

@api_bp.route('/api/standings', methods=['GET'])
def get_standings():
    try:
        competition = request.args.get('competition', 'PL')
        football_client, _, _, _, _ = _clients()
        standings = football_client.get_standings(competition)
        return jsonify({'standings': standings, 'competition': competition})
    except Exception as e:
        logger.error(f"Error fetching standings: {e}")
        return jsonify({'error': str(e), 'standings': []}), 400

@api_bp.route('/api/head-to-head', methods=['GET'])
def get_head_to_head():
    try:
        team1_id = int(request.args.get('team1_id'))
        team2_id = int(request.args.get('team2_id'))
        football_client, _, _, _, _ = _clients()
        h2h = football_client.get_head_to_head(team1_id, team2_id)
        return jsonify({'matches': h2h, 'count': len(h2h)})
    except Exception as e:
        logger.error(f"Error fetching H2H: {e}")
        return jsonify({'error': str(e)}), 400

@api_bp.route('/api/aggregated/leagues', methods=['GET'])
def get_aggregated_leagues():
    try:
        _, _, _, _, aggregated_source = _clients()
        if aggregated_source.df.empty:
            return jsonify({'leagues': []})
        def normalize(name: str) -> str:
            if not name:
                return ''
            s = str(name).replace('-', ' ').strip()
            return ' '.join([w.capitalize() for w in s.split()])
        raw = aggregated_source.df.get('League_norm')
        if raw is None:
            leagues_series = aggregated_source.df['League'].fillna(aggregated_source.df['league'])
        else:
            leagues_series = raw
        league_names = set([normalize(x) for x in leagues_series.dropna().unique().tolist()])
        code_map = {
            'Premier League': 'PL',
            'Bundesliga': 'BL1',
            'Eredivisie': 'DED',
            'La Liga': 'PD',
            'Champions League': 'CL',
            'Championship': 'ELC',
        }
        leagues = []
        excluded = {'Serie A', 'Ligue 1'}
        for name in sorted(league_names):
            if name in excluded:
                continue
            code = code_map.get(name)
            if code:
                leagues.append({'code': code, 'name': name})
        return jsonify({'leagues': leagues})
    except Exception as e:
        logger.error(f"Error fetching aggregated leagues: {e}")
        return jsonify({'leagues': []}), 200

@api_bp.route('/api/aggregated/predict', methods=['POST'])
def aggregated_predict():
    try:
        data = request.json or {}
        unique_id = data.get('unique_id')
        if not unique_id:
            return jsonify({'error': 'unique_id is required'}), 400
        _, _, predictor, _, aggregated_source = _clients()
        row = aggregated_source.get_match_by_unique_id(unique_id)
        if row is None or row.empty:
            return jsonify({'error': 'Match not found in aggregated dataset'}), 404
        match_name = ' '.join(row.get('match', [])) if isinstance(row.get('match'), list) else (row.get('match') or f"{row.get('Home')} vs {row.get('Away')}")
        base_prob = aggregated_source.poisson_btts_prob(row.get('Home_xG'), row.get('Away_xG'))
        mult = aggregated_source.volatility_multiplier(row)
        final_prob = max(0.05, min(0.95, base_prob * mult))
        gaps = aggregated_source.value_gaps(row, final_prob)
        ml_prob = None
        try:
            features = [
                float(row.get('Home_xG') or 0),
                float(row.get('Away_xG') or 0),
                float(row.get('Possession_home_pct') or 0),
                float(row.get('ShotsOnTarget_home') or 0),
                float(row.get('ShotsOnTarget_away') or 0),
                float(row.get('Corners_home') or 0),
                float(row.get('Corners_away') or 0),
                float(row.get('PassAcc_home_pct') or 0),
                float(row.get('PassAcc_away_pct') or 0),
            ]
            ml_result = predictor.predict_btts_aggregated(np.array(features).reshape(1, -1))
            if ml_result and 'probability' in ml_result:
                ml_prob = float(ml_result['probability'])
                final_prob = (final_prob * 0.6) + (ml_prob * 0.4)
        except Exception as e:
            logger.info(f"Aggregated ML prediction not available: {e}")
        confidence = max(final_prob, 1 - final_prob)
        return jsonify({
            'match': match_name,
            'BTTS_Probability': float(final_prob),
            'Recommended_Bet': 'Yes' if final_prob >= 0.52 else 'No',
            'Confidence_Interval': float(confidence),
            'base_poisson': float(base_prob),
            'volatility_multiplier': float(mult),
            'ml_probability': ml_prob,
            'value_gaps': gaps
        })
    except Exception as e:
        logger.error(f"Error in aggregated prediction: {e}")
        return jsonify({'error': str(e)}), 500
@api_bp.route('/api/ai/analyze-btts', methods=['POST'])
def analyze_btts_ai():
    """Generate AI-powered BTTS analysis using Google Generative AI"""
    try:
        data = request.json
        home_team_id = data.get('home_team_id')
        away_team_id = data.get('away_team_id')
        season = data.get('season', '2024')
        
        football_client, feature_engineer, predictor, db, _, ai_analyzer = _clients()
        
        # Get team info
        home_info = football_client.get_team_info(home_team_id)
        away_info = football_client.get_team_info(away_team_id)
        
        home_name = home_info.get('name', f'Team {home_team_id}')
        away_name = away_info.get('name', f'Team {away_team_id}')
        
        # Get team stats
        home_matches = football_client.get_team_matches(home_team_id, season)
        away_matches = football_client.get_team_matches(away_team_id, season)
        
        home_stats = feature_engineer.calculate_team_stats(home_matches, home_team_id, is_home=None)
        away_stats = feature_engineer.calculate_team_stats(away_matches, away_team_id, is_home=None)
        
        # Get recent form (last 5 matches)
        home_recent = football_client.get_recent_matches(home_team_id, limit=5)
        away_recent = football_client.get_recent_matches(away_team_id, limit=5)
        
        home_recent_sorted = sorted(home_recent, key=lambda x: x.get('utcDate', ''), reverse=True)
        away_recent_sorted = sorted(away_recent, key=lambda x: x.get('utcDate', ''), reverse=True)
        
        # Calculate form string (wins/draws/losses in last 5)
        def get_form_string(matches, team_id):
            results = []
            for match in matches[:5]:
                is_home = match.get('homeTeam', {}).get('id') == team_id
                if is_home:
                    home_goals = match.get('score', {}).get('fullTime', {}).get('home', 0)
                    away_goals = match.get('score', {}).get('fullTime', {}).get('away', 0)
                else:
                    home_goals = match.get('score', {}).get('fullTime', {}).get('home', 0)
                    away_goals = match.get('score', {}).get('fullTime', {}).get('away', 0)
                
                if is_home:
                    if home_goals > away_goals:
                        results.append('W')
                    elif home_goals == away_goals:
                        results.append('D')
                    else:
                        results.append('L')
                else:
                    if away_goals > home_goals:
                        results.append('W')
                    elif home_goals == away_goals:
                        results.append('D')
                    else:
                        results.append('L')
            return '-'.join(results[:5]) if results else 'No Data'
        
        home_form = get_form_string(home_recent_sorted, home_team_id)
        away_form = get_form_string(away_recent_sorted, away_team_id)
        
        # Get standings for position
        try:
            standings = football_client.get_standings('PL')  # Assuming PL for now
            home_pos = next((s for s in standings if s.get('team_id') == home_team_id), {}).get('position', 'N/A')
            away_pos = next((s for s in standings if s.get('team_id') == away_team_id), {}).get('position', 'N/A')
        except:
            home_pos = 'N/A'
            away_pos = 'N/A'
        
        # Make prediction
        prediction = predictor.predict_ensemble(home_stats, away_stats)
        expected_goals = (home_stats.get('goals_per_game', 0) + away_stats.get('goals_per_game', 0))
        btts_probability = prediction.get('btts_probability', 0)
        confidence = max(btts_probability, 1 - btts_probability) if btts_probability else 0
        
        # Generate AI analysis
        ai_result = ai_analyzer.generate_btts_analysis(
            home_name=home_name,
            away_name=away_name,
            home_stats=home_stats,
            away_stats=away_stats,
            expected_goals=expected_goals,
            btts_probability=btts_probability,
            confidence=confidence,
            home_form=home_form,
            away_form=away_form,
            home_pos=f'Position {home_pos}',
            away_pos=f'Position {away_pos}'
        )
        
        return jsonify({
            'success': ai_result['success'],
            'analysis': ai_result['analysis'],
            'error': ai_result.get('error'),
            'home_name': home_name,
            'away_name': away_name,
            'btts_probability': float(btts_probability),
            'home_stats': home_stats,
            'away_stats': away_stats
        })
        
    except Exception as e:
        logger.error(f"Error in AI BTTS analysis: {e}")
        return jsonify({
            'success': False,
            'analysis': '',
            'error': str(e)
        }), 500

@api_bp.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({'status': 'ok'}), 200
