const DEFAULT_API_BASE = 'http://localhost:5000/api';
function resolveApiBase() {
    try {
        const w = typeof window !== 'undefined' ? window : {};
        if (w.__API_BASE_URL) return String(w.__API_BASE_URL).replace(/\/$/, '');
        const m = typeof document !== 'undefined' ? document.querySelector('meta[name="api-base-url"]') : null;
        if (m && m.content) return String(m.content).replace(/\/$/, '');
        const o = typeof location !== 'undefined' ? location.origin : '';
        if (o) return `${o}/api`;
        return DEFAULT_API_BASE;
    } catch (_) {
        return DEFAULT_API_BASE;
    }
}
const API_BASE_URL = resolveApiBase();
function getClientId() {
    try {
        const k = 'client_id';
        const existing = localStorage.getItem(k);
        if (existing) return existing;
        const gen = (crypto && crypto.randomUUID) ? crypto.randomUUID() : (`cid_${Math.random().toString(36).slice(2)}_${Date.now()}`);
        localStorage.setItem(k, gen);
        return gen;
    } catch (_) {
        return `cid_${Math.random().toString(36).slice(2)}_${Date.now()}`;
    }
}
const CLIENT_ID = getClientId();

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
                user_id: CLIENT_ID,
            }),
            signal
        });
    }

    static async getUpcomingFixtures(competition = 'PL', days = 30) {
        return this.request(`/fixtures/upcoming?competition=${competition}&days=${days}`);
    }

    static async getPredictionsHistory(limit = 100) {
        const q = encodeURIComponent(CLIENT_ID);
        return this.request(`/predictions/history?limit=${limit}&user_id=${q}`);
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
