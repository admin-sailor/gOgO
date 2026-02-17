let teams = [];
let currentPrediction = null;
let dashboardData = null;
let selectedLeague = 'PL'; // Track selected league
let pendingFixtureId = null; // Set when predicting from fixtures page
let activeFixtureBtn = null; // Track active fixture predict button
let teamsLoading = false;
let currentPredictController = null;
let predictionCanceled = false;
let predictTimeoutId = null;
let teamsLoadingToastEl = null;
let teamsLoadingTimer = null;

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    console.log(' Initializing BTTS Predictor...');
    initializeTheme();
    initializeEventListeners();
    initializeSectionFromStorage();
    initializeImageProtection();
    const predictBtn = document.getElementById('predictBtn');
    if (predictBtn) predictBtn.disabled = true;
    await populateLeaguesFromDataset();
    await loadTeams(selectedLeague);
    await loadDashboard();
    initializeLoader();
    startNewsLoader();
});

function initializeEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', switchSection);
    });
    const brand = document.querySelector('.brand-icon');
    if (brand) {
        brand.style.cursor = 'pointer';
        brand.addEventListener('click', () => {
            switchSectionManually('dashboard');
        });
    }

    // League selector
    document.getElementById('league').addEventListener('change', (e) => {
        selectedLeague = e.target.value || 'PL';
        console.log(' League changed to:', selectedLeague);
        loadTeams(selectedLeague);
    });

    // Predictions
    document.getElementById('predictBtn').addEventListener('click', makePrediction);
    const cancelBtn = document.getElementById('cancelPredictBtn');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', cancelPrediction);
    }

    // Fixtures
    document.getElementById('loadFixturesBtn').addEventListener('click', loadUpcomingFixtures);
    const themeBtn = document.getElementById('themeToggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', toggleTheme);
    }

    // History
    document.getElementById('loadHistoryBtn').addEventListener('click', loadPredictionHistory);
    document.getElementById('exportHistoryBtn').addEventListener('click', exportHistoryCSV);

    enhanceSelectWithScroll('league', 10);
    enhanceSelectWithScroll('homeTeam', 10);
    enhanceSelectWithScroll('awayTeam', 10);
    enhanceSelectWithScroll('season', 6);
    enhanceSelectWithScroll('competitionFilter', 10);
}

async function populateLeaguesFromDataset() {
    try {
        // Prefer API.getAggregatedLeagues if available, else fallback to direct fetch
        let leaguesResp = null;
        if (typeof window !== 'undefined' && window.API && typeof window.API.getAggregatedLeagues === 'function') {
            leaguesResp = await window.API.getAggregatedLeagues();
        } else {
            const res = await fetch('http://localhost:5000/api/aggregated/leagues', {
                headers: { 'Content-Type': 'application/json' },
                mode: 'cors'
            });
            if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
            }
            leaguesResp = await res.json();
        }
        const { leagues } = leaguesResp || {};
        const select = document.getElementById('league');
        if (!select) return;
        const options = ['<option value="">Select a league...</option>'].concat(
            (leagues || []).map(l => `<option value="${l.code}">${l.name}</option>`)
        );
        select.innerHTML = options.join('');
        // Set default if PL available
        const hasPL = (leagues || []).some(l => l.code === 'PL');
        selectedLeague = hasPL ? 'PL' : ((leagues[0] && leagues[0].code) || '');
        if (selectedLeague) {
            select.value = selectedLeague;
        }
    } catch (e) {
        console.warn('Failed to populate leagues from dataset:', e && e.message ? e.message : e);
    }
}

function enhanceSelectWithScroll(id, size = 8) {
    const el = document.getElementById(id);
    if (!el) return;
    const root = el.parentElement || el;
    let listEl = root.querySelector('.dropdown-list');
    if (!listEl) {
        listEl = document.createElement('div');
        listEl.className = 'dropdown-list';
        root.appendChild(listEl);
    }
    const isOpen = () => listEl.style.display === 'block';
    const close = () => {
        listEl.style.display = 'none';
        el.classList.remove('select-expanded');
    };
    const open = () => {
        const opts = Array.from(el.options || []).filter(o => String(o.value) !== '');
        const maxRows = Math.max(4, Math.min(size, opts.length));
        listEl.innerHTML = opts.map(o => `<div class="dropdown-option" data-val="${o.value}">${o.text}</div>`).join('');
        listEl.style.display = 'block';
        listEl.style.maxHeight = `${maxRows * 32}px`;
        el.classList.add('select-expanded');
        listEl.querySelectorAll('.dropdown-option').forEach(optEl => {
            optEl.addEventListener('mousedown', (e) => {
                e.preventDefault();
                const v = optEl.getAttribute('data-val');
                el.value = v;
                const ev = new Event('change');
                el.dispatchEvent(ev);
                close();
            }, { once: true });
        });
    };
    el.addEventListener('mousedown', (e) => {
        e.preventDefault();
        if (isOpen()) {
            close();
        } else {
            open();
        }
    });
    document.addEventListener('mousedown', (e) => {
        if (isOpen()) {
            const inside = e.target === el || listEl.contains(e.target);
            if (!inside) close();
        }
    });
    el.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') close();
    });
    el.addEventListener('blur', () => {
        close();
    });
}

function initializeImageProtection() {
    try {
        const brand = document.querySelector('.brand-icon');
        if (brand) brand.classList.add('interactive-img');
        const themeImg = document.querySelector('#themeToggle img');
        if (themeImg) themeImg.classList.add('interactive-img');
        document.querySelectorAll('img').forEach(img => {
            img.setAttribute('draggable', 'false');
        });
        document.addEventListener('contextmenu', (e) => {
            const t = e.target;
            if (t && t.tagName === 'IMG' && !t.classList.contains('interactive-img')) {
                e.preventDefault();
            }
        });
        document.addEventListener('dragstart', (e) => {
            const t = e.target;
            if (t && t.tagName === 'IMG' && !t.classList.contains('interactive-img')) {
                e.preventDefault();
            }
        });
        document.addEventListener('click', (e) => {
            const path = e.composedPath ? e.composedPath() : [];
            const img = path.find(el => el && el.tagName === 'IMG');
            if (img && !img.classList.contains('interactive-img')) {
                const anchor = path.find(el => el && el.tagName === 'A');
                if (anchor) {
                    e.preventDefault();
                }
            }
        }, true);
    } catch (_) {}
}

function switchSection(e) {
    const sectionName = e.target.dataset.section;
    
    // Update navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    e.target.classList.add('active');

    // Update sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(sectionName).classList.add('active');
    try { localStorage.setItem('currentSection', sectionName); } catch (_) {}
    if (sectionName !== 'predictions') {
        stopTeamsLoadingToast(false);
    }
}

async function loadTeams(leagueCode) {
    try {
        console.log(' Loading teams for league:', leagueCode);
        teamsLoading = true;
        const predictBtn = document.getElementById('predictBtn');
        if (predictBtn) predictBtn.disabled = true;
        if (document.getElementById('predictions').classList.contains('active')) {
            startTeamsLoadingToast('Fetching teams');
        }
        if (!leagueCode) {
            document.getElementById('homeTeam').innerHTML = '<option value="">Select home team...</option>';
            document.getElementById('awayTeam').innerHTML = '<option value="">Select away team...</option>';
            teams = [];
            teamsLoading = false;
            if (predictBtn) predictBtn.disabled = false;
            stopTeamsLoadingToast(false);
            return;
        }

        console.log(' Fetching teams for league:', leagueCode);
        const response = await API.getTeams(leagueCode);
        teams = response.teams || [];
        console.log(' Teams loaded:', teams.length);

        // Populate dropdowns
        const teamOptions = teams.map(team => 
            `<option value="${team.id}">${team.name}</option>`
        ).join('');

        document.getElementById('homeTeam').innerHTML = '<option value="">Select home team...</option>' + teamOptions;
        document.getElementById('awayTeam').innerHTML = '<option value="">Select away team...</option>' + teamOptions;

        // Prefetch crests
        try {
            const limit = Math.min(20, teams.length);
            for (let i = 0; i < limit; i++) {
                const u = teams[i]?.crest;
                if (u) {
                    const img = new Image();
                    img.decoding = 'async';
                    img.loading = 'eager';
                    img.referrerPolicy = 'no-referrer';
                    img.src = u;
                }
            }
        } catch (_) {}
        // Show team count
        const teamCountEl = document.getElementById('teamCount');
        if (teamCountEl) {
            teamCountEl.textContent = teams.length + ' teams';
        }
        teamsLoading = false;
        if (predictBtn) predictBtn.disabled = false;
        stopTeamsLoadingToast(true);
    } catch (error) {
        console.error(' Failed to load teams:', error);
        document.getElementById('homeTeam').innerHTML = `<option value="">Error loading teams: ${error.message}</option>`;
        document.getElementById('awayTeam').innerHTML = `<option value="">Error loading teams: ${error.message}</option>`;
        teamsLoading = false;
        const predictBtn = document.getElementById('predictBtn');
        if (predictBtn) predictBtn.disabled = false;
        stopTeamsLoadingToast(false);
    }
}

function getTeamName(teamId) {
    const team = teams.find(t => t.id === teamId || t.id === parseInt(teamId));
    return team?.name || `Team ${teamId}`;
}

async function loadDashboard() {
    try {
        console.log(' Loading dashboard...');
        
        // Top Leagues List
        const topLeagues = [
            { code: 'PL', name: 'Premier League', country: 'England', crest: 'https://cdn.worldvectorlogo.com/logos/premier-league-1.svg' },
            { code: 'BL1', name: 'Bundesliga', country: 'Germany', crest: 'https://cdn.worldvectorlogo.com/logos/bundesliga-2.svg' },
            { code: 'DED', name: 'Eredivisie', country: 'Netherlands', crest: 'https://cdn.worldvectorlogo.com/logos/eredivisie.svg' },
            { code: 'CL', name: 'Champions League', country: 'Europe', crest: 'https://cdn.worldvectorlogo.com/logos/uefa-champions-league-1.svg' }
        ];

        const topLeaguesHtml = topLeagues.map(league => `
            <div class="league-item" onclick="selectLeague('${league.code}')">
                <img src="${league.crest}" alt="${league.name}" class="league-crest" loading="lazy" decoding="async" fetchpriority="low" onerror="this.src='logo-ico.png'">
                <div class="league-info">
                    <div class="league-name">${league.name}</div>
                    <div class="league-country">${league.country}</div>
                </div>
            </div>
        `).join('');
        const topLeaguesList = document.getElementById('topLeaguesList');
        if (topLeaguesList) {
            topLeaguesList.innerHTML = topLeaguesHtml;
        }

        // Fetch predictions
        let predList = [];
        try {
            const predictions = await API.getPredictionsHistory(100);
            predList = predictions.predictions || [];
            console.log(' Predictions loaded:', predList.length);
        } catch (e) {
            console.warn(' Predictions not available:', e.message);
        }

        // Recent Predictions (show all, scrollbar provided via CSS)
        const recentList = predList;
    if (recentList.length === 0) {
        document.getElementById('recentPredictions').innerHTML = `
            <div id="recentLoader" class="recent-loader"></div>
            <p class="loading">No predictions yet. Make one to see it here.</p>
        `;
        startRecentLoader();
    } else {
        stopRecentLoader();
        const recentPredictionsHtml = recentList.map(pred => `
            <div class="prediction-item">
                <div class="team-info">
                    <div class="team-name">${pred.home_team_name || getTeamName(pred.home_team_id)} vs ${pred.away_team_name || getTeamName(pred.away_team_id)}</div>
                    <div class="team-stat">BTTS: ${pred.btts_prediction ? 'YES' : 'NO'} (${((pred.btts_probability || 0) * 100).toFixed(1)}%)</div>
                </div>
            </div>
        `).join('');
        document.getElementById('recentPredictions').innerHTML = recentPredictionsHtml;
    }

    } catch (error) {
        console.error(' Dashboard loading error:', error);
    }
}

// Helper to select league from dashboard
function selectLeague(code) {
    const select = document.getElementById('league');
    if (select) {
        select.value = code;
        // Trigger change event manually
        const event = new Event('change');
        select.dispatchEvent(event);
        // Switch to predictions tab
        document.querySelector('[data-section="predictions"]').click();
    }
}

async function makePrediction() {
    const homeTeamId = parseInt(document.getElementById('homeTeam').value);
    const awayTeamId = parseInt(document.getElementById('awayTeam').value);
    const season = document.getElementById('season').value;
    const modelType = document.getElementById('modelType').value;

    if (predictionCanceled) {
        predictionCanceled = false;
        stopLoader();
        const btn = document.getElementById('predictBtn');
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Predict BTTS';
        }
        if (activeFixtureBtn) {
            activeFixtureBtn.disabled = false;
            activeFixtureBtn.textContent = 'Predict BTTS';
            activeFixtureBtn = null;
        }
        return;
    }
    if (!homeTeamId || !awayTeamId) {
        showToast('Please select both teams', 'error');
        return;
    }

    if (homeTeamId === awayTeamId) {
        showToast('Home and away teams must be different', 'error');
        return;
    }

    try {
        document.getElementById('predictBtn').disabled = true;
        document.getElementById('predictBtn').textContent = 'Analysing...';
        startLoader();
        currentPredictController = new AbortController();

        const result = await API.predictBTTS(homeTeamId, awayTeamId, pendingFixtureId, season, modelType, currentPredictController.signal);
        pendingFixtureId = null; // Clear after use

        currentPrediction = result;
        await displayPredictionResult(result);

        // Refresh dashboard so Recent Predictions, Top Teams, and charts update
        loadDashboard();

        document.getElementById('predictBtn').disabled = false;
        document.getElementById('predictBtn').textContent = 'Predict BTTS';
        stopLoader();
        currentPredictController = null;
        if (activeFixtureBtn) {
            activeFixtureBtn.disabled = false;
            activeFixtureBtn.textContent = 'Predict BTTS';
            activeFixtureBtn = null;
        }
    } catch (error) {
        const isAbort = error && (error.name === 'AbortError' || error.message.includes('AbortError'));
        if (!isAbort) {
            console.error('Error making prediction:', error);
            showToast('Error making prediction', 'error');
        }
        pendingFixtureId = null;
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('predictBtn').textContent = 'Predict BTTS';
        stopLoader();
        currentPredictController = null;
        if (activeFixtureBtn) {
            activeFixtureBtn.disabled = false;
            activeFixtureBtn.textContent = 'Predict BTTS';
            activeFixtureBtn = null;
        }
    }
}

function safeNum(val, decimals = 1) {
    const n = parseFloat(val);
    return (typeof n === 'number' && !isNaN(n)) ? n.toFixed(decimals) : '--';
}

async function displayPredictionResult(result) {
    const homeTeamId = parseInt(document.getElementById('homeTeam').value);
    const awayTeamId = parseInt(document.getElementById('awayTeam').value);
    const homeTeam = teams.find(t => t.id === homeTeamId);
    const awayTeam = teams.find(t => t.id === awayTeamId);

    const homeStats = result.home_stats || {};
    const awayStats = result.away_stats || {};
    const prediction = result.prediction || {};
    const indicators = result.indicators || {};

    // Update team info with safe formatting
    document.getElementById('homeTeamName').textContent = homeTeam?.name || 'Home';
    document.getElementById('homeTeamCrest').src = homeTeam?.crest || '';
    document.getElementById('homeGoalsPerGame').textContent = safeNum(homeStats.goals_per_game, 2);
    document.getElementById('homeGoalsConceded').textContent = safeNum(homeStats.goals_conceded_per_game, 2);
    (function () {
        const val = parseFloat(homeStats.defensive_fragility_index) || 0;
        const el = document.getElementById('homeDFI');
        const note = document.getElementById('homeDFINote');
        el.textContent = safeNum(val, 2);
        el.classList.remove('dfi-green', 'dfi-yellow', 'dfi-red');
        let status = '';
        if (val < 4.9) {
            el.classList.add('dfi-green');
            status = 'Strong';
        } else if (val >= 5.0 && val <= 7.4) {
            el.classList.add('dfi-yellow');
            status = 'Moderate';
        } else if (val >= 7.5) {
            el.classList.add('dfi-red');
            status = 'Weak';
        }
        if (note) note.textContent = status;
    })();
    document.getElementById('homeCleanSheets').textContent = homeStats.clean_sheet_frequency != null && !isNaN(homeStats.clean_sheet_frequency)
        ? ((homeStats.clean_sheet_frequency * 100).toFixed(1) + '%') : '-';

    document.getElementById('awayTeamName').textContent = awayTeam?.name || 'Away';
    document.getElementById('awayTeamCrest').src = awayTeam?.crest || '';
    document.getElementById('awayGoalsPerGame').textContent = safeNum(awayStats.goals_per_game, 2);
    document.getElementById('awayGoalsConceded').textContent = safeNum(awayStats.goals_conceded_per_game, 2);
    (function () {
        const val = parseFloat(awayStats.defensive_fragility_index) || 0;
        const el = document.getElementById('awayDFI');
        const note = document.getElementById('awayDFINote');
        el.textContent = safeNum(val, 2);
        el.classList.remove('dfi-green', 'dfi-yellow', 'dfi-red');
        let status = '';
        if (val < 4.9) {
            el.classList.add('dfi-green');
            status = 'Strong';
        } else if (val >= 5.0 && val <= 7.4) {
            el.classList.add('dfi-yellow');
            status = 'Moderate';
        } else if (val >= 7.5) {
            el.classList.add('dfi-red');
            status = 'Weak';
        }
        if (note) note.textContent = status;
    })();
    document.getElementById('awayCleanSheets').textContent = awayStats.clean_sheet_frequency != null && !isNaN(awayStats.clean_sheet_frequency)
        ? ((awayStats.clean_sheet_frequency * 100).toFixed(1) + '%') : '-';

    (function annotateStats() {
        function setNote(val, elValue, elNote, thresholds, texts) {
            elValue.classList.remove('stat-green', 'stat-yellow', 'stat-red');
            if (!elNote) return;
            if (val == null || isNaN(val)) {
                elNote.textContent = '';
                return;
            }
            let cls = 'stat-yellow', txt = texts[1];
            if (val >= thresholds.high.min && val < thresholds.high.max) { cls = 'stat-green'; txt = texts[0]; }
            else if (val >= thresholds.mid.min && val < thresholds.mid.max) { cls = 'stat-yellow'; txt = texts[1]; }
            else { cls = 'stat-red'; txt = texts[2]; }
            elValue.classList.add(cls);
            elNote.textContent = txt;
        }
        const hg = parseFloat(homeStats.goals_per_game);
        const hc = parseFloat(homeStats.goals_conceded_per_game);
        const hcsPct = (homeStats.clean_sheet_frequency || 0) * 100;
        setNote(hg,
            document.getElementById('homeGoalsPerGame'),
            document.getElementById('homeGoalsNote'),
            { high: {min: 1.8, max: Infinity}, mid: {min: 1.2, max: 1.8} },
            ['High enough', 'Balanced', 'Low output']
        );
        setNote(hc,
            document.getElementById('homeGoalsConceded'),
            document.getElementById('homeConcedeNote'),
            { high: {min: -Infinity, max: 0.8}, mid: {min: 0.8, max: 1.4} },
            ['Solid defense', 'Fragile defense', 'Leaky defense']
        );
        setNote(hcsPct,
            document.getElementById('homeCleanSheets'),
            document.getElementById('homeCleanNote'),
            { high: {min: 40, max: Infinity}, mid: {min: 20, max: 40} },
            ['Excellent', 'Average', 'Low']
        );
        const ag = parseFloat(awayStats.goals_per_game);
        const ac = parseFloat(awayStats.goals_conceded_per_game);
        const acsPct = (awayStats.clean_sheet_frequency || 0) * 100;
        setNote(ag,
            document.getElementById('awayGoalsPerGame'),
            document.getElementById('awayGoalsNote'),
            { high: {min: 1.8, max: Infinity}, mid: {min: 1.2, max: 1.8} },
            ['High enough', 'Balanced', 'Low output']
        );
        setNote(ac,
            document.getElementById('awayGoalsConceded'),
            document.getElementById('awayConcedeNote'),
            { high: {min: -Infinity, max: 0.8}, mid: {min: 0.8, max: 1.4} },
            ['Solid defense', 'Fragile defense', 'Leaky defense']
        );
        setNote(acsPct,
            document.getElementById('awayCleanSheets'),
            document.getElementById('awayCleanNote'),
            { high: {min: 40, max: Infinity}, mid: {min: 20, max: 40} },
            ['Excellent', 'Average', 'Low']
        );
    })();
    // League table positions
    const leagueCode = document.getElementById('league').value || selectedLeague;
    const standingsMap = {};
    try {
        const standingsData = await API.getStandings(leagueCode);
        (standingsData.standings || []).forEach(s => {
            standingsMap[s.team_id] = s.position;
        });
    } catch (_) {}
    const homePos = standingsMap[homeTeamId];
    const awayPos = standingsMap[awayTeamId];
    document.getElementById('homeTeamPosition').textContent = homePos ? `· ${ordinal(homePos)}` : '';
    document.getElementById('awayTeamPosition').textContent = awayPos ? `· ${ordinal(awayPos)}` : '';

    // Update prediction - safe formatting to prevent NaN
    const bttsProb = safeNum((prediction.btts_probability || 0) * 100, 1);
    const confidence = safeNum((prediction.confidence || 0) * 100, 1);
    document.getElementById('bttsProb').textContent = bttsProb;
    const yesNoEl = document.getElementById('bttsYesNo');
    yesNoEl.textContent = prediction.btts_prediction ? 'YES ✓' : 'NO ✗';
    yesNoEl.classList.toggle('yes', !!prediction.btts_prediction);
    yesNoEl.classList.toggle('no', !prediction.btts_prediction);
    document.getElementById('bttsConfidence').textContent = confidence + '%';
    document.getElementById('expectedGoals').textContent = safeNum(indicators.expected_total_goals, 2);
    const confVal = Math.max(0, Math.min(100, (parseFloat(prediction.confidence || 0) * 100)));
    const confEl = document.getElementById('confidenceFill');
    if (confEl) confEl.style.width = confVal + '%';

    // Model comparison - always set (-- when not available)
    document.getElementById('lrProb').textContent = prediction.lr_probability != null
        ? safeNum(prediction.lr_probability * 100, 1) + '%' : '--';
    document.getElementById('nnProb').textContent = prediction.nn_probability != null
        ? safeNum(prediction.nn_probability * 100, 1) + '%' : '--';

    // BTTS analysis
    const homeScoring = homeStats.goals_per_game || 0;
    const awayScoring = awayStats.goals_per_game || 0;
    const homeConceding = homeStats.goals_conceded_per_game || 0;
    const awayConceding = awayStats.goals_conceded_per_game || 0;
    const attackingThreat = (homeScoring + awayScoring) / 2;
    const defensiveWeakness = (homeConceding + awayConceding) / 2;
    document.getElementById('analysisAttacking').textContent = attackingThreat >= 1.5
        ? `High (${safeNum(attackingThreat, 2)} avg goals)` : attackingThreat >= 1
            ? `Moderate (${safeNum(attackingThreat, 2)} avg)` : `Low (${safeNum(attackingThreat, 2)} avg)`;
    document.getElementById('analysisDefensive').textContent = defensiveWeakness >= 1.5
        ? `High (${safeNum(defensiveWeakness, 2)} conceded)` : defensiveWeakness >= 1
            ? `Moderate (${safeNum(defensiveWeakness, 2)} conceded)` : `Low (${safeNum(defensiveWeakness, 2)} conceded)`;
    const probVal = parseFloat(prediction.btts_probability) || 0;
    let insight = probVal >= 0.6
        ? 'Both teams score regularly and concede often — strong BTTS opportunity.'
        : probVal >= 0.45
            ? 'Balanced match — BTTS possible but not guaranteed.'
            : 'Defensive setups or low-scoring trends — BTTS less likely.';
    document.getElementById('analysisInsight').textContent = insight;

    // Circle color
    const circle = document.getElementById('predictionCircle');
    const probPct = Math.max(0, Math.min(100, (prediction.btts_probability || 0) * 100));
    const angle = probPct * 3.6; // 100% = 360deg
    let ringColor = '#f56565'; // red low
    if (probPct > 65) ringColor = '#48bb78'; // green high
    else if (probPct > 40) ringColor = '#ecc94b'; // yellow mid
    const restColor = getComputedStyle(document.documentElement).getPropertyValue('--bg-tertiary') || '#2d3748';
    circle.style.background = `conic-gradient(${ringColor} 0deg, ${ringColor} ${angle}deg, ${restColor} ${angle}deg, ${restColor} 360deg)`;

    // removed crest overlay on BTTS card per request

    // Render Recent Form & H2H
    renderForm(result.home_last_5, homeTeamId, 'homeLast5');
    renderForm(result.away_last_5, awayTeamId, 'awayLast5');
    renderH2H(result.h2h_matches, homeTeamId, 'h2hMatches');

    document.getElementById('predictionResult').classList.remove('hidden');
    const cards = document.getElementById('resultCards');
    if (cards) {
        cards.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function renderForm(matches, teamId, elementId) {
    const container = document.getElementById(elementId);
    if (!container) return;
    
    if (!matches || matches.length === 0) {
        container.innerHTML = '<p class="team-stat">No recent matches</p>';
        return;
    }

    const html = matches.map(match => {
        const isHome = match.homeTeam.id === teamId;
        const homeScore = match.score.fullTime.home;
        const awayScore = match.score.fullTime.away;
        const opponent = isHome ? match.awayTeam.name : match.homeTeam.name;
        const opponentId = isHome ? match.awayTeam.id : match.homeTeam.id;
        const date = new Date(match.utcDate).toLocaleDateString(undefined, {month:'short', day:'numeric'});

        let resultClass = 'draw';
        if (homeScore !== null && awayScore !== null) {
            if (isHome) {
                if (homeScore > awayScore) resultClass = 'win';
                else if (homeScore < awayScore) resultClass = 'loss';
            } else {
                if (awayScore > homeScore) resultClass = 'win';
                else if (awayScore < homeScore) resultClass = 'loss';
            }
        }

        const leftCrest = `url('https://crests.football-data.org/${match.homeTeam.id}.png')`;
        const rightCrest = `url('https://crests.football-data.org/${match.awayTeam.id}.png')`;
        return `
            <div class="match-item ${resultClass}" style="--left-crest:${leftCrest}; --right-crest:${rightCrest};">
                <div>
                    <div class="match-score">${homeScore ?? '-'} - ${awayScore ?? '-'}</div>
                    <div class="match-date">${date}</div>
                </div>
                <div class="match-opponent" title="${opponent}">vs ${opponent}</div>
            </div>
        `;
    }).join('');

    container.innerHTML = html;
}

function renderH2H(matches, homeTeamId, elementId) {
    const container = document.getElementById(elementId);
    if (!container) return;

    if (!matches || matches.length === 0) {
        container.innerHTML = '<p class="team-stat">No recent head-to-head</p>';
        return;
    }

    const html = matches.map(match => {
        const homeScore = match.score.fullTime.home;
        const awayScore = match.score.fullTime.away;
        const date = new Date(match.utcDate).toLocaleDateString(undefined, {year:'numeric', month:'short', day:'numeric'});
        const homeName = match.homeTeam.shortName || match.homeTeam.name;
        const awayName = match.awayTeam.shortName || match.awayTeam.name;
        
        // Determine winner relative to home team
        let resultClass = 'draw';
        if (homeScore !== null && awayScore !== null) {
            if (match.homeTeam.id === homeTeamId) {
                 if (homeScore > awayScore) resultClass = 'win';
                 else if (homeScore < awayScore) resultClass = 'loss';
            } else {
                 // Home team is away in this match
                 if (awayScore > homeScore) resultClass = 'win';
                 else if (awayScore < homeScore) resultClass = 'loss';
            }
        }

        const leftCrest = `url('https://crests.football-data.org/${match.homeTeam.id}.png')`;
        const rightCrest = `url('https://crests.football-data.org/${match.awayTeam.id}.png')`;
        return `
            <div class="match-item ${resultClass}" style="--left-crest:${leftCrest}; --right-crest:${rightCrest};">
                <div>
                    <div class="match-score">${homeScore ?? '-'} - ${awayScore ?? '-'}</div>
                    <div class="match-date">${date}</div>
                </div>
                <div class="match-opponent">
                    ${homeName} vs ${awayName}
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = html;
}

let loaderAnim = null;
function initializeLoader() {
    const overlay = document.getElementById('loaderOverlay');
    const container = document.getElementById('lottieContainer');
    if (!overlay || !container || typeof lottie === 'undefined') return;
    loaderAnim = lottie.loadAnimation({
        container,
        renderer: 'svg',
        loop: true,
        autoplay: false,
        path: './loader.json'
    });
}

function startLoader() {
    const overlay = document.getElementById('loaderOverlay');
    if (!overlay) return;
    overlay.style.display = 'flex';
    if (loaderAnim) loaderAnim.play();
}

function stopLoader() {
    const overlay = document.getElementById('loaderOverlay');
    if (!overlay) return;
    if (loaderAnim) loaderAnim.stop();
    overlay.style.display = 'none';
}

let recentLoaderAnim = null;
function startRecentLoader() {
    const container = document.getElementById('recentLoader');
    if (!container || typeof lottie === 'undefined') return;
    if (!recentLoaderAnim) {
        recentLoaderAnim = lottie.loadAnimation({
            container,
            renderer: 'svg',
            loop: true,
            autoplay: true,
            path: './soccer-ball.json'
        });
    } else {
        recentLoaderAnim.play();
    }
}

function stopRecentLoader() {
    if (recentLoaderAnim) {
        recentLoaderAnim.stop();
    }
    const container = document.getElementById('recentLoader');
    if (container) {
        container.innerHTML = '';
    }
}
let newsLoaderAnim = null;
function startNewsLoader() {
    const container = document.getElementById('newsLoader');
    if (!container || typeof lottie === 'undefined') return;
    if (!newsLoaderAnim) {
        newsLoaderAnim = lottie.loadAnimation({
            container,
            renderer: 'svg',
            loop: true,
            autoplay: true,
            path: './loader2.json'
        });
    } else {
        newsLoaderAnim.play();
    }
}

function initializeTheme() {
    const saved = localStorage.getItem('theme') || 'dark';
    const body = document.body;
    body.classList.remove('theme-dark', 'theme-light');
    body.classList.add(saved === 'light' ? 'theme-light' : 'theme-dark');
}

function toggleTheme() {
    const body = document.body;
    const isDark = body.classList.contains('theme-dark');
    body.classList.toggle('theme-dark', !isDark);
    body.classList.toggle('theme-light', isDark);
    localStorage.setItem('theme', isDark ? 'light' : 'dark');
}
function ordinal(n) {
    const s = ['th', 'st', 'nd', 'rd'];
    const v = n % 100;
    if (v >= 11 && v <= 13) return n + 'th';
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

async function loadUpcomingFixtures() {
    try {
        const competition = document.getElementById('competitionFilter').value;
        document.getElementById('loadFixturesBtn').disabled = true;
        // Changed from 'Loading...' to 'Predicting...' as requested
        document.getElementById('loadFixturesBtn').textContent = 'Fetching...';

        const data = await API.getUpcomingFixtures(competition, 30);
        const fixtures = data.fixtures || [];

        const fixturesHtml = fixtures.map(fixture => {
            const homeTeam = fixture.homeTeam || {};
            const awayTeam = fixture.awayTeam || {};
            const date = new Date(fixture.utcDate).toLocaleString();

            return `
                <div class="fixture-card">
                    <div class="fixture-date">${date}</div>
                    <div class="fixture-teams">
                        <div class="fixture-team">
                            <img src="${homeTeam.crest}" alt="${homeTeam.name}" class="fixture-crest">
                            <div class="fixture-team-name" title="${homeTeam.name}">${homeTeam.name}</div>
                        </div>
                        <div class="fixture-vs">vs</div>
                        <div class="fixture-team">
                            <img src="${awayTeam.crest}" alt="${awayTeam.name}" class="fixture-crest">
                            <div class="fixture-team-name" title="${awayTeam.name}">${awayTeam.name}</div>
                        </div>
                    </div>
                    <button id="fixture-predict-btn-${fixture.id}" class="fixture-predict-btn btn-primary" onclick="predictFromFixture(${homeTeam.id}, ${awayTeam.id}, ${fixture.id}, '${competition}', this)">
                        Predict BTTS
                    </button>
                </div>
            `;
        }).join('');

        document.getElementById('fixturesList').innerHTML = fixturesHtml || '<p class="loading">No upcoming fixtures</p>';
        document.getElementById('loadFixturesBtn').disabled = false;
        document.getElementById('loadFixturesBtn').textContent = 'Load Fixtures';
    } catch (error) {
        console.error('Error loading fixtures:', error);
        alert('Error loading fixtures');
        document.getElementById('loadFixturesBtn').disabled = false;
        document.getElementById('loadFixturesBtn').textContent = 'Load Fixtures';
    }
}

async function predictFromFixture(homeTeamId, awayTeamId, fixtureId, competition, btnEl) {
    // 1. Map fixture competition to league
    const leagueCode = (competition === 'LA') ? 'PD' : (competition || 'PL');
    
    document.getElementById('league').value = leagueCode;
    selectedLeague = leagueCode;
    
    // 2. Handle Button UI State BEFORE any async calls to match predictBtn behavior
    if (btnEl) {
        btnEl.disabled = true;
        btnEl.textContent = 'Predicting...';
        activeFixtureBtn = btnEl;
    } else {
        const fallbackBtn = document.getElementById(`fixture-predict-btn-${fixtureId}`);
        if (fallbackBtn) {
            fallbackBtn.disabled = true;
            fallbackBtn.textContent = 'Predicting...';
            activeFixtureBtn = fallbackBtn;
        }
    }

    // 3. Start the loader immediately
    startLoader();
    
    // 4. Load teams and set input values
    await loadTeams(leagueCode);
    
    document.getElementById('homeTeam').value = homeTeamId;
    document.getElementById('awayTeam').value = awayTeamId;
    
    pendingFixtureId = fixtureId; 
    
    // 5. Navigate and trigger hidden click
    switchSectionManually('predictions');
    document.getElementById('predictionResult').classList.add('hidden');
    
    predictTimeoutId = setTimeout(() => {
        // This triggers the main predictBtn which likely has its own loader logic
        document.getElementById('predictBtn').click();
        predictTimeoutId = null;
    }, 100);
}


function switchSectionManually(sectionName) {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.section === sectionName) {
            btn.classList.add('active');
        }
    });

    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(sectionName).classList.add('active');
    try { localStorage.setItem('currentSection', sectionName); } catch (_) {}
    if (sectionName !== 'predictions') {
        stopTeamsLoadingToast(false);
    }
}

function initializeSectionFromStorage() {
    try {
        const stored = localStorage.getItem('currentSection') || 'dashboard';
        switchSectionManually(stored);
    } catch (_) {}
}

async function loadPredictionHistory() {
    try {
        document.getElementById('loadHistoryBtn').disabled = true;
        document.getElementById('loadHistoryBtn').textContent = 'Loading...';

        const data = await API.getPredictionsHistory(100);
        const predictions = data.predictions || [];

        if (predictions.length === 0) {
            document.getElementById('historyTable').innerHTML = '<p class="loading">No prediction history</p>';
            return;
        }

        let html = `
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Home Team</th>
                        <th>Away Team</th>
                        <th>BTTS Prediction</th>
                        <th>Probability</th>
                        <th>Confidence</th>
                        <th>Model</th>
                    </tr>
                </thead>
                <tbody>
        `;

        predictions.forEach(pred => {
            const date = new Date(pred.created_at).toLocaleDateString();
            const bttsYesNo = pred.btts_prediction ? 'YES' : 'NO';
            const probability = ((pred.btts_probability || 0) * 100).toFixed(1);
            const confidence = ((pred.confidence || 0) * 100).toFixed(1);

            html += `
                <tr>
                    <td>${date}</td>
                    <td>${pred.home_team_name || getTeamName(pred.home_team_id)}</td>
                    <td>${pred.away_team_name || getTeamName(pred.away_team_id)}</td>
                    <td>${bttsYesNo}</td>
                    <td>${probability}%</td>
                    <td>${confidence}%</td>
                    <td>${pred.model_type}</td>
                </tr>
            `;
        });

        html += `
                </tbody>
            </table>
        `;

        document.getElementById('historyTable').innerHTML = html;
        document.getElementById('loadHistoryBtn').disabled = false;
        document.getElementById('loadHistoryBtn').textContent = 'Load History';
    } catch (error) {
        console.error('Error loading history:', error);
        alert('Error loading history');
        document.getElementById('loadHistoryBtn').disabled = false;
        document.getElementById('loadHistoryBtn').textContent = 'Load History';
    }
}

function exportHistoryCSV() {
    const table = document.querySelector('table');
    if (!table) {
        alert('No history to export');
        return;
    }

    let csv = '';
    const rows = table.querySelectorAll('tr');

    rows.forEach(row => {
        const cols = row.querySelectorAll('td, th');
        csv += Array.from(cols).map(col => `"${col.textContent}"`).join(',') + '\n';
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'btts_predictions_history.csv';
    a.click();
    window.URL.revokeObjectURL(url);
}

// Chart functions with enhanced rendering
function createBTTSChart() {
    const canvas = document.getElementById('bttsChart');
    if (!canvas) return;

    // Setup canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = 300;

    const renderer = new ChartRenderer('bttsChart');
    
    // Get probability distribution
    const distribution = dashboardData.getProbabilityDistribution();
    const labels = Object.keys(distribution);
    const values = Object.values(distribution);
    
    if (values.reduce((a, b) => a + b, 0) === 0) {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#a0aec0';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No data available', canvas.width / 2, canvas.height / 2);
        return;
    }

    const colors = ['#48bb78', '#ecc94b', '#4299e1', '#9f7aea', '#f56565'];
    renderer.drawPieChart({
        labels: labels,
        values: values,
        colors: colors,
    });
}

function createModelPerformanceChart() {
    const canvas = document.getElementById('modelPerformanceChart');
    if (!canvas) return;

    // Setup canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = 300;

    const renderer = new ChartRenderer('modelPerformanceChart');
    
    // Get model comparison
    const modelComparison = dashboardData.getModelComparison();
    const models = Object.keys(modelComparison);
    const probabilities = models.map(m => {
        const val = modelComparison[m];
        return val === 'N/A' ? 0 : parseFloat(val);
    });

    if (probabilities.reduce((a, b) => a + b, 0) === 0) {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#a0aec0';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No data available', canvas.width / 2, canvas.height / 2);
        return;
    }

    const colors = ['#9f7aea', '#4299e1', '#48bb78'];
    renderer.drawBarChart({
        labels: models.map(m => m.replace(/_/g, ' ')),
        values: probabilities,
        colors: colors,
    });
}

function showToast(message, type = 'error') {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    const t = document.createElement('div');
    t.className = 'toast ' + (type === 'success' ? 'toast-success' : 'toast-error');
    const icon = document.createElement('div');
    icon.className = 'toast-icon';
    if (type === 'success') {
        icon.innerHTML = '<svg width="16" height="16" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg"><g clip-path="url(#clip0_477_331)"><path fill-rule="evenodd" clip-rule="evenodd" d="M0 9C0 6.61305 0.948212 4.32387 2.63604 2.63604C4.32387 0.948212 6.61305 0 9 0C11.3869 0 13.6761 0.948212 15.364 2.63604C17.0518 4.32387 18 6.61305 18 9C18 11.3869 17.0518 13.6761 15.364 15.364C13.6761 17.0518 11.3869 18 9 18C6.61305 18 4.32387 17.0518 2.63604 15.364C0.948212 13.6761 0 11.3869 0 9ZM8.4864 12.852L13.668 6.3744L12.732 5.6256L8.3136 11.1468L5.184 8.5392L4.416 9.4608L8.4864 12.852Z" fill="#51DF34"/></g><defs><clipPath id="clip0_477_331"><rect width="18" height="18" fill="white"/></clipPath></defs></svg>';
    } else {
        icon.innerHTML = '<svg width="16" height="16" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg"><g clip-path="url(#clip0_456_329)"><path d="M4.8 11.25L7.5 8.55L10.2 11.25L11.25 10.2L8.54999 7.5L11.25 4.8L10.2 3.75L7.5 6.45L4.8 3.75L3.75 4.8L6.45 7.5L3.75 10.2L4.8 11.25ZM7.5 15C6.4625 15 5.4875 14.803 4.575 14.409C3.6625 14.015 2.86875 13.4807 2.19375 12.8062C1.51875 12.1317 0.9845 11.338 0.591001 10.425C0.197501 9.512 0.000500949 8.537 9.49366e-07 7.5C-0.00049905 6.463 0.196501 5.488 0.591001 4.575C0.9855 3.662 1.51975 2.86825 2.19375 2.19375C2.86775 1.51925 3.6615 0.985 4.575 0.591C5.4885 0.197 6.4635 0 7.5 0C8.5365 0 9.51149 0.197 10.425 0.591C11.3385 0.985 12.1322 1.51925 12.8062 2.19375C13.4802 2.86825 14.0147 3.662 14.4097 4.575C14.8047 5.488 15.0015 6.463 15 7.5C14.9985 8.537 14.8015 9.512 14.409 10.425C14.0165 11.338 13.4822 12.1317 12.8062 12.8062C12.1302 13.4807 11.3365 14.0152 10.425 14.4097C9.51349 14.8042 8.5385 15.001 7.5 15Z" fill="#DF4A34"/></g><defs><clipPath id="clip0_456_329"><rect width="15" height="15" fill="white"/></clipPath></defs></svg>';
    }
    const msg = document.createElement('div');
    msg.className = 'toast-message';
    msg.textContent = message;
    t.appendChild(icon);
    t.appendChild(msg);
    container.appendChild(t);
    requestAnimationFrame(() => {
        t.classList.add('enter');
    });
    setTimeout(() => {
        t.classList.remove('enter');
        t.classList.add('exit');
        setTimeout(() => {
            if (t && t.parentNode) t.parentNode.removeChild(t);
        }, 250);
    }, 3000);
}

function startTeamsLoadingToast(baseMsg) {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    const pred = document.getElementById('predictions');
    if (!pred || !pred.classList.contains('active')) return;
    if (teamsLoadingToastEl) return;
    const t = document.createElement('div');
    t.className = 'toast toast-fetch enter';
    const icon = document.createElement('div');
    icon.className = 'toast-icon';
    icon.innerHTML = '<svg width="15" height="15" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg"><g clip-path="url(#clip0_480_333)"><path d="M7.5 15C3.35775 15 0 11.6422 0 7.5C0 3.35775 3.35775 0 7.5 0C11.6422 0 15 3.3585 15 7.5C15 11.6415 11.6422 15 7.5 15ZM7.5 3.75C7.30109 3.75 7.11032 3.82902 6.96967 3.96967C6.82902 4.11032 6.75 4.30109 6.75 4.5V8.25C6.75 8.44891 6.82902 8.63968 6.96967 8.78033C7.11032 8.92098 7.30109 9 7.5 9C7.69891 9 7.88968 8.92098 8.03033 8.78033C8.17098 8.63968 8.25 8.44891 8.25 8.25V4.5C8.25 4.30109 8.17098 4.11032 8.03033 3.96967C7.88968 3.82902 7.69891 3.75 7.5 3.75ZM7.5 11.25C7.69891 11.25 7.88968 11.171 8.03033 11.0303C8.17098 10.8897 8.25 10.6989 8.25 10.5C8.25 10.3011 8.17098 10.1103 8.03033 9.96967C7.88968 9.82902 7.69891 9.75 7.5 9.75C7.30109 9.75 7.11032 9.82902 6.96967 9.96967C6.82902 10.1103 6.75 10.3011 6.75 10.5C6.75 10.6989 6.82902 10.8897 6.96967 11.0303C7.11032 11.171 7.30109 11.25 7.5 11.25Z" fill="#DFCB34"/></g><defs><clipPath id="clip0_480_333"><rect width="15" height="15" fill="white"/></clipPath></defs></svg>';
    const msg = document.createElement('div');
    msg.className = 'toast-message';
    msg.textContent = baseMsg;
    t.appendChild(icon);
    t.appendChild(msg);
    container.appendChild(t);
    teamsLoadingToastEl = t;
    let dots = 0;
    teamsLoadingTimer = setInterval(() => {
        dots = (dots + 1) % 4;
        msg.textContent = baseMsg + '.'.repeat(dots);
    }, 500);
}

function stopTeamsLoadingToast(success) {
    const predActive = !!document.getElementById('predictions') && document.getElementById('predictions').classList.contains('active');
    if (teamsLoadingTimer) {
        clearInterval(teamsLoadingTimer);
        teamsLoadingTimer = null;
    }
    if (teamsLoadingToastEl) {
        teamsLoadingToastEl.classList.remove('enter');
        teamsLoadingToastEl.classList.add('exit');
        const el = teamsLoadingToastEl;
        teamsLoadingToastEl = null;
        setTimeout(() => {
            if (el && el.parentNode) el.parentNode.removeChild(el);
            if (success && predActive) showToast('Teams loaded successfully', 'success');
        }, 250);
    } else {
        if (success && predActive) showToast('Teams loaded successfully', 'success');
    }
}
function cancelPrediction() {
    try {
        predictionCanceled = true;
        if (predictTimeoutId) {
            clearTimeout(predictTimeoutId);
            predictTimeoutId = null;
        }
        if (currentPredictController) {
            currentPredictController.abort();
        }
        stopLoader();
        const btn = document.getElementById('predictBtn');
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Predict BTTS';
        }
        if (activeFixtureBtn) {
            activeFixtureBtn.disabled = false;
            activeFixtureBtn.textContent = 'Predict BTTS';
            activeFixtureBtn = null;
        }
        pendingFixtureId = null;
        // removed toast on cancel per request
    } catch (_) {}
}
