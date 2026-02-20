import google.generativeai as genai
import logging
from typing import Dict, Optional
from config import GOOGLE_API_KEY

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """Generate AI-powered football betting analysis using Google Generative AI"""
    
    def __init__(self):
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None
            logger.warning("GOOGLE_API_KEY not configured")
    
    def generate_btts_analysis(self, 
                              home_name: str,
                              away_name: str,
                              home_stats: Dict,
                              away_stats: Dict,
                              expected_goals: float,
                              btts_probability: float,
                              confidence: float,
                              home_form: str = "Form",
                              away_form: str = "Form",
                              home_pos: str = "Position",
                              away_pos: str = "Position") -> Optional[Dict]:
        """
        Generate AI-powered betting analysis for BTTS prediction
        
        Returns dict with:
        - success: bool
        - analysis: str (the AI response)
        - error: str (if any)
        """
        if not self.model:
            return {
                'success': False,
                'analysis': '',
                'error': 'Google API not configured'
            }
        
        try:
            # Format numbers for display
            def fmt(val, decimals=2):
                if val is None:
                    return '--'
                try:
                    return f"{float(val):.{decimals}f}"
                except:
                    return str(val)
            
            # Build the prompt using the provided template
            prompt = (
                "ACT AS: A Senior Football Quantitative Analyst and Professional Betting Consultant. "
                "TONE: High-conviction, professional, and slightly cynical. Use technical football jargon "
                "(e.g., 'low block', 'high press', 'transitional vulnerability', 'squad depth').\n\n"
                
                "OBJECTIVE: Synthesize the provided internal model metrics with your worldwide 2026 knowledge "
                "of team tactics, key player availability, and historical match-up trends to determine the "
                "true value of a BTTS (GG/NG) outcome.\n\n"
                
                "STRUCTURE YOUR RESPONSE INTO THESE 4 SECTIONS:\n"
                "1. TACTICAL NARRATIVE: How will these two specific styles clash? Identify the 'Confluence'—"
                "the point where stats and tactical reality meet (e.g., Home team's high line vs Away team's pace on the break).\n"
                
                "2. THE BTTS SEAL: Don't just give a %; explain WHY. Look for 'Game-changers' like key striker "
                "injuries or defensive suspensions in 2026. Is the defense 'vulnerable' or just 'unlucky'?\n"
                
                "3. MARKET CONFLUENCE & ODDS: Estimate what the professional market odds should be for GG/NG "
                "based on this data. Highlight if the 2026 model probability offers 'Value' compared to common bookie lines.\n"
                
                "4. FINAL VERDICT & RED FLAGS: Provide a final suggestion. Mention one 'Red Flag' that could "
                "ruin the prediction (e.g., weather, rotation, or recent managerial changes).\n\n"
                
                "DATA INPUTS:\n"
                f"MATCHUP: {home_name} vs {away_name}\n"
                f"HOME METRICS: Goals: {fmt(home_stats.get('goals_per_game'))}, Conceded: {fmt(home_stats.get('goals_conceded_per_game'))}, Clean Sheet: {fmt((home_stats.get('clean_sheet_frequency') or 0)*100,1)}%. Form: {home_form}.\n"
                f"AWAY METRICS: Goals: {fmt(away_stats.get('goals_per_game'))}, Conceded: {fmt(away_stats.get('goals_conceded_per_game'))}, Clean Sheet: {fmt((away_stats.get('clean_sheet_frequency') or 0)*100,1)}%. Form: {away_form}.\n"
                f"POSITIONAL CONTEXT: Home {home_pos}, Away {away_pos}.\n"
                f"INTERNAL MODEL: xG {fmt(expected_goals)}, BTTS Prob {fmt((btts_probability or 0)*100,1)}%, Confidence {fmt((confidence or 0)*100,1)}%."
            )
            
            logger.info(f"Generating AI analysis for {home_name} vs {away_name}")
            
            # Generate the response
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                return {
                    'success': True,
                    'analysis': response.text,
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'analysis': '',
                    'error': 'Empty response from AI'
                }
                
        except Exception as e:
            logger.error(f"Error generating AI analysis: {e}")
            return {
                'success': False,
                'analysis': '',
                'error': str(e)
            }
