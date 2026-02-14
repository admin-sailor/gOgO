const API_BASE_URL = 'http://localhost:5000/api';

class API {
    static async request(endpoint, options = {}) {
        try {
            const url = `${API_BASE_URL}${endpoint}`;
            console.log(' API Request:', url);
            
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                },
                mode: 'cors',
                ...options,
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error(' API Error Response:', errorText);
                throw new Error(`API Error: ${response.status} ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log(' API Response:', data);
            return data;
        } catch (error) {
            console.error(' API Request Failed:', error.message);
            throw error;
        }
    }

    static async getTeams(competition = 'PL') {
        return this.request(`/teams?competition=${competition}`);
    }

    static async getTeamStats(teamId, season = '2024') {
        return this.request(`/team/${teamId}/stats?season=${season}`);
    }

    static async predictBTTS(homeTeamId, awayTeamId, fixtureId = null, season = '2024', modelType = 'ensemble', signal = undefined) {
        return this.request('/predict/btts', {
            method: 'POST',
            body: JSON.stringify({
                home_team_id: homeTeamId,
                away_team_id: awayTeamId,
                fixture_id: fixtureId,
                season: season,
                model: modelType,
            }),
            signal
        });
    }

    static async getUpcomingFixtures(competition = 'PL', days = 30) {
        return this.request(`/fixtures/upcoming?competition=${competition}&days=${days}`);
    }

    static async getPredictionsHistory(limit = 100) {
        return this.request(`/predictions/history?limit=${limit}`);
    }

    static async getHeadToHead(team1Id, team2Id) {
        return this.request(`/head-to-head?team1_id=${team1Id}&team2_id=${team2Id}`);
    }

    static async getStandings(competition = 'PL') {
        return this.request(`/standings?competition=${competition}`);
    }

    static async getAggregatedLeagues() {
        return this.request('/aggregated/leagues');
    }
}
// Expose API to global window to ensure availability across scripts
if (typeof window !== 'undefined') {
    window.API = API;
}
