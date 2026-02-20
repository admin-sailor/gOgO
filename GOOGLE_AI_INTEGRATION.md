# Google Generative AI Integration Guide

## Overview
Your gOgO football betting prediction app now includes AI-powered analysis using Google's Generative AI (Gemini). After generating a BTTS prediction, users can click "Generate AI Analysis" to receive expert-level tactical and statistical insights about the matchup.

## How It Works

### Backend Flow
1. **Data Collection**: When a user requests AI analysis, the backend collects:
   - Team statistics (goals per game, goals conceded, clean sheet percentage)
   - Recent form (last 5 matches: W/D/L pattern)
   - League position
   - Model-generated predictions (xG, BTTS probability, confidence)

2. **Prompt Construction**: The collected data is formatted into a detailed prompt that instructs the Gemini model to act as a "Senior Football Quantitative Analyst and Professional Betting Consultant"

3. **AI Analysis**: Gemini generates analysis structured into 4 key sections:
   - **Tactical Narrative**: How the teams' playing styles will clash
   - **BTTS Seal**: Probability and reasoning for both/both teams scoring
   - **Market Confluence & Odds**: Estimated betting odds and value assessment
   - **Final Verdict & Red Flags**: Recommendation with key risk factors

### Frontend Integration
- New "AI-Powered Analysis" section appears below prediction results
- "Generate AI Analysis" button triggers the API call
- Loading spinner shows while generating
- Formatted response displays with proper sections and styling

## Architecture

### Backend Files Added/Modified

**New File: `/backend/ai_analyzer.py`**
- `AIAnalyzer` class: Manages Google Generative AI integration
- `generate_btts_analysis()`: Creates formatted prompt and calls Gemini API
- Handles error cases and logging

**Modified: `/backend/__init__.py`**
- Imports and initializes `AIAnalyzer`
- Adds AI analyzer to Flask app config

**Modified: `/backend/config.py`**
- Added `GOOGLE_API_KEY` configuration variable

**Modified: `/backend/blueprints/api.py`**
- New endpoint: `POST /api/ai/analyze-btts`
- Collects team stats and feeds them to AI analyzer
- Returns formatted AI analysis to frontend

**Modified: `/backend/requirements.txt`**
- Added `google-generativeai==0.3.0` dependency

### Frontend Files Modified

**Modified: `/frontend/index.html`**
- Added "AI-Powered Analysis" section with button and result container
- Integrated into the existing prediction results layout

**Modified: `/frontend/js/app.js`**
- `generateAIAnalysis()`: Handles button click and API call
- `formatAIAnalysis()`: Formats Gemini response with HTML styling
- Event listener integration for the new button

**Modified: `/frontend/js/api.js`**
- New method: `analyzeBTTSWithAI()`: Client-side API wrapper

**Modified: `/frontend/css/styles.css`**
- Complete AI section styling with animations
- Spinner and loading states
- Responsive design for mobile/tablet
- Dark and light theme support

## Environment Setup

### Required: Google API Key
1. Create a project at [Google AI Studio](https://aistudio.google.com)
2. Generate an API key for Gemini
3. Add to your environment variables:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

The key is already configured in your Vercel environment.

## API Endpoint Reference

### POST /api/ai/analyze-btts
Generates AI-powered BTTS analysis using team statistics and model predictions.

**Request Body:**
```json
{
  "home_team_id": 57,
  "away_team_id": 61,
  "season": "2024"
}
```

**Response:**
```json
{
  "success": true,
  "analysis": "...",
  "home_name": "Arsenal",
  "away_name": "Chelsea",
  "btts_probability": 0.68,
  "home_stats": { ... },
  "away_stats": { ... }
}
```

## Usage Example

1. User selects Home and Away teams
2. Clicks "Predict BTTS" to get initial prediction
3. Views prediction results and form analysis
4. Clicks "Generate AI Analysis" button
5. Waits for Gemini to analyze the matchup
6. Reads comprehensive tactical and betting analysis

## Model Configuration

The integration uses **Gemini 2.5 Flash** for:
- Fast response times (~1-3 seconds)
- Cost-effective API calls
- Reliable performance for tactical analysis

The prompt is designed to extract:
- Deep tactical insights
- Probability assessment reasoning
- Betting value identification
- Risk factors and red flags

## Error Handling

- Missing API key → Error message, button remains enabled for retry
- API rate limiting → Graceful error with retry option
- Network failures → User-friendly error toast notification
- Invalid team selection → Validation before API call

## Future Enhancements

Possible improvements:
- Caching previous analyses to reduce API calls
- Support for other prediction types (over/under, handicaps)
- Extended analysis with injury reports
- Historical accuracy tracking
- User feedback integration for model refinement

## Testing the Integration

1. Ensure `GOOGLE_API_KEY` is set in environment
2. Navigate to the app and select two teams
3. Make a prediction
4. Click "Generate AI Analysis"
5. Review the formatted response

## Troubleshooting

**"Google API not configured"**
- Check that `GOOGLE_API_KEY` environment variable is set
- Restart the backend server
- Verify API key is valid at Google AI Studio

**Analysis not generating**
- Check backend logs for API errors
- Verify both teams are different
- Ensure prediction was made first (both teams selected)
- Check internet connectivity

**Timeout errors**
- Gemini API may take 2-3 seconds
- Increase client timeout if needed
- Check Google Cloud API quota limits

---

**Created**: 2026-02-20  
**Version**: 1.0  
**Status**: Production Ready
