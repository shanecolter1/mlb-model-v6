exports.handler = async function() {
  const headers = {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'access-control-allow-origin': '*'
  };
  const send = (statusCode, data) => ({ statusCode, headers, body: JSON.stringify(data) });

  try {
    const apiKey = process.env.SPORTSGAMEODDS_API_KEY;
    if (!apiKey) return send(503, { error: 'SPORTSGAMEODDS_API_KEY is not configured', events: [] });

    const url = new URL('https://api.sportsgameodds.com/v2/events');
    url.searchParams.set('leagueID', 'MLB');
    url.searchParams.set('oddsAvailable', 'true');
    url.searchParams.set('started', 'false');
    url.searchParams.set('oddID', 'points-all-game-ou-over');
    url.searchParams.set('bookmakerID', 'draftkings');
    url.searchParams.set('includeAltLines', 'false');
    url.searchParams.set('includeOpenCloseOdds', 'false');
    url.searchParams.set('limit', '100');

    const response = await fetch(url, {
      headers: {
        accept: 'application/json',
        'x-api-key': apiKey,
        'user-agent': 'MLB-I2-Run-Environment/1.0'
      }
    });
    const text = await response.text();
    if (!response.ok) return send(response.status, { error: text, events: [] });

    const raw = JSON.parse(text);
    const events = [];
    for (const item of Array.isArray(raw?.data) ? raw.data : []) {
      const odd = item?.odds?.['points-all-game-ou-over'];
      const book = odd?.byBookmaker?.draftkings;
      const point = Number(book?.overUnder);
      if (!book || book.available !== true || !Number.isFinite(point)) continue;
      events.push({
        eventId: String(item.eventID || ''),
        commenceTime: item?.status?.startsAt || item?.startTime || null,
        awayTeam: item?.teams?.away?.names?.long || item?.teams?.away?.name || null,
        homeTeam: item?.teams?.home?.names?.long || item?.teams?.home?.name || null,
        fullGameTotal: point,
        bookmaker: 'draftkings',
        lastUpdate: book.lastUpdatedAt || null
      });
    }

    return send(200, {
      capturedAt: new Date().toISOString(),
      source: 'SPORTSGAMEODDS',
      scope: 'FULL_GAME_TOTAL_POINT_ONLY_NO_PRICES',
      marketIsolation: {
        phase: 'PRE_FREEZE',
        bookmaker: 'draftkings',
        oddID: 'points-all-game-ou-over',
        pricesExposedToPredictionEngine: false
      },
      events
    });
  } catch (error) {
    return send(502, { error: String(error?.message || error), events: [] });
  }
};
