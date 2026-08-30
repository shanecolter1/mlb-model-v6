exports.handler = async function(event) {
  const headers = {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'access-control-allow-origin': '*'
  };
  const send = (statusCode, data) => ({ statusCode, headers, body: JSON.stringify(data) });

  try {
    const apiKey = process.env.ODDS_API_KEY;
    if (!apiKey) return send(503, { error: 'ODDS_API_KEY is not configured', events: [] });

    const url = `https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey=${encodeURIComponent(apiKey)}&regions=us&markets=totals&bookmakers=draftkings&oddsFormat=american&dateFormat=iso`;
    const response = await fetch(url, {
      headers: { accept: 'application/json', 'user-agent': 'MLB-I2-Run-Environment/0.1' }
    });
    const text = await response.text();
    if (!response.ok) return send(response.status, { error: text, events: [] });

    const raw = JSON.parse(text);
    const events = [];
    for (const item of raw) {
      const book = (item.bookmakers || []).find(b => b.key === 'draftkings');
      const market = (book?.markets || []).find(m => m.key === 'totals');
      const over = (market?.outcomes || []).find(o => String(o.name).toLowerCase() === 'over');
      const under = (market?.outcomes || []).find(o => String(o.name).toLowerCase() === 'under');
      const overPoint = Number(over?.point);
      const underPoint = Number(under?.point);
      if (!Number.isFinite(overPoint) || !Number.isFinite(underPoint) || Math.abs(overPoint - underPoint) > 1e-9) continue;
      events.push({
        eventId: item.id,
        commenceTime: item.commence_time,
        awayTeam: item.away_team,
        homeTeam: item.home_team,
        fullGameTotal: overPoint,
        bookmaker: 'draftkings',
        lastUpdate: market?.last_update || book?.last_update || null
      });
    }

    return send(200, {
      capturedAt: new Date().toISOString(),
      source: 'The Odds API DraftKings totals point only',
      scope: 'FULL_GAME_TOTAL_POINT_ONLY_NO_PRICES',
      events,
      quota: {
        remaining: response.headers.get('x-requests-remaining'),
        used: response.headers.get('x-requests-used'),
        last: response.headers.get('x-requests-last')
      }
    });
  } catch (error) {
    return send(502, { error: String(error?.message || error), events: [] });
  }
};

// Operational no-logic-change trigger for 2026-08-30 I2 daily run.
