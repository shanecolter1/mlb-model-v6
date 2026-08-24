exports.handler = async function(event) {
  const headers = {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    "access-control-allow-origin": "*"
  };

  const send = (statusCode, data, extraHeaders = {}) => ({
    statusCode,
    headers: { ...headers, ...extraHeaders },
    body: typeof data === "string" ? data : JSON.stringify(data)
  });
  const bullpenFetch = (url) => fetch(url, { headers: { accept: "application/json", "user-agent": "MLB-Live-Command-Center/11.14-bullpen" } });
  const iso = d => d.toISOString().slice(0,10);

  try {
    const q = event.queryStringParameters || {};

    if (q.type === "schedule") {
      const date = /^\d{4}-\d{2}-\d{2}$/.test(q.date || "") ? q.date : new Date().toISOString().slice(0,10);
      const url = `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${encodeURIComponent(date)}&hydrate=probablePitcher,team,linescore`;
      const response = await fetch(url, { headers: { accept: "application/json", "user-agent": "MLB-Live-Command-Center/11.2" } });
      return send(response.status, await response.text());
    }

    if (q.type === "feed") {
      if (!/^\d+$/.test(q.gamePk || "")) return send(400, { error: "Invalid gamePk" });
      const url = `https://statsapi.mlb.com/api/v1.1/game/${q.gamePk}/feed/live`;
      const response = await fetch(url, { headers: { accept: "application/json", "user-agent": "MLB-Live-Command-Center/11.2" } });
      return send(response.status, await response.text());
    }

    if (q.type === "rosters") {
      const teamIds = String(q.teamIds || "").split(",").filter(x => /^\d+$/.test(x)).slice(0, 30);
      const date = /^\d{4}-\d{2}-\d{2}$/.test(q.date || "") ? q.date : new Date().toISOString().slice(0,10);
      if (!teamIds.length) return send(200, { teams: {} });
      const entries = await Promise.all(teamIds.map(async teamId => {
        const url = `https://statsapi.mlb.com/api/v1/teams/${teamId}/roster?rosterType=active&date=${encodeURIComponent(date)}`;
        const response = await fetch(url, { headers: { accept: "application/json", "user-agent": "MLB-Live-Command-Center/11.2" } });
        if (!response.ok) return [teamId, { roster: [], error: response.status }];
        const data = await response.json();
        return [teamId, {
          roster: (data.roster || []).map(entry => ({
            id: entry.person?.id,
            name: entry.person?.fullName,
            positionType: entry.position?.type || "Unknown",
            positionCode: entry.position?.code || ""
          })).filter(player => player.id)
        }];
      }));
      return send(200, { date, teams: Object.fromEntries(entries) });
    }

    // Empirical bullpen context is intentionally isolated from the legacy dashboard
    // schedule/feed/roster/stats routes. It is cached and never called per pitch.
    if (q.type === "bullpenContext") {
      if (!/^\d+$/.test(q.teamId || "")) return send(400, { error: "Invalid teamId" });
      const teamId = Number(q.teamId);
      const targetDate = /^\d{4}-\d{2}-\d{2}$/.test(q.date || "") ? q.date : new Date().toISOString().slice(0,10);
      const target = new Date(`${targetDate}T12:00:00Z`);
      const end = new Date(target); end.setUTCDate(end.getUTCDate() - 1);
      const start = new Date(target); start.setUTCDate(start.getUTCDate() - 30);
      const schedUrl = `https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId=${teamId}&gameType=R&startDate=${iso(start)}&endDate=${iso(end)}`;
      const schedResp = await bullpenFetch(schedUrl);
      if (!schedResp.ok) return send(schedResp.status, await schedResp.text());
      const sched = await schedResp.json();
      const games = [];
      for (const d of (sched.dates || [])) {
        for (const g of (d.games || [])) {
          const st = g.status || {};
          if (st.abstractGameState === "Final" || ["Final","Game Over","Completed Early"].includes(st.detailedState)) games.push({ gamePk: g.gamePk, date: d.date });
        }
      }
      const feeds = await Promise.all(games.map(async g => {
        try {
          const r = await bullpenFetch(`https://statsapi.mlb.com/api/v1.1/game/${g.gamePk}/feed/live`);
          return r.ok ? { ...g, feed: await r.json() } : null;
        } catch (_) { return null; }
      }));
      const hist = new Map();
      for (const item of feeds.filter(Boolean)) {
        const feed = item.feed;
        const teams = feed.gameData?.teams || {};
        const homeId = Number(teams.home?.id || 0), awayId = Number(teams.away?.id || 0);
        const perPitcher = new Map();
        for (const play of (feed.liveData?.plays?.allPlays || [])) {
          const about = play.about || {}, matchup = play.matchup || {};
          const half = String(about.halfInning || "").toLowerCase();
          const defenseId = half === "top" ? homeId : half === "bottom" ? awayId : 0;
          if (defenseId !== teamId) continue;
          const pid = Number(matchup.pitcher?.id || 0), inning = Number(about.inning || 0);
          if (!pid || !inning) continue;
          const x = perPitcher.get(pid) || { bf: 0, firstInning: inning };
          x.bf += 1; x.firstInning = Math.min(x.firstInning, inning); perPitcher.set(pid, x);
        }
        const dayOrd = Math.floor(new Date(`${item.date}T12:00:00Z`).getTime() / 86400000);
        for (const [pid, x] of perPitcher.entries()) {
          const z = hist.get(pid) || { apps: 0, bf: 0, lateApps: 0, saveLikeApps: 0, lastDayOrd: null };
          z.apps += 1; z.bf += x.bf; z.lateApps += x.firstInning >= 7 ? 1 : 0; z.saveLikeApps += x.firstInning >= 9 ? 1 : 0;
          z.lastDayOrd = z.lastDayOrd == null ? dayOrd : Math.max(z.lastDayOrd, dayOrd); hist.set(pid, z);
        }
      }
      const rosterUrl = `https://statsapi.mlb.com/api/v1/teams/${teamId}/roster?rosterType=active&date=${encodeURIComponent(targetDate)}&hydrate=person`;
      const rosterResp = await bullpenFetch(rosterUrl);
      if (!rosterResp.ok) return send(rosterResp.status, await rosterResp.text());
      const rosterData = await rosterResp.json();
      const targetOrd = Math.floor(target.getTime() / 86400000);
      const candidates = (rosterData.roster || []).filter(x => {
        const p = x.position || {};
        return String(p.abbreviation || "").toUpperCase() === "P" || String(p.type || "").toLowerCase() === "pitcher";
      }).map(x => {
        const pid = Number(x.person?.id || 0), z = hist.get(pid) || { apps:0,bf:0,lateApps:0,saveLikeApps:0,lastDayOrd:null };
        return {
          pitcher_id: pid,
          name: x.person?.fullName || `Pitcher ${pid}`,
          throws: x.person?.pitchHand?.code || "UNK",
          prior_apps_30d: z.apps,
          prior_bf_30d: z.bf,
          prior_rest_days: z.lastDayOrd == null ? null : targetOrd - z.lastDayOrd,
          prior_late_inning_share_30d: z.apps ? z.lateApps / z.apps : 0,
          prior_save_like_share_30d: z.apps ? z.saveLikeApps / z.apps : 0
        };
      }).filter(x => x.pitcher_id);
      return send(200, {
        status: "EMPIRICAL_BULLPEN_CONTEXT",
        market_blind: true,
        teamId,
        date: targetDate,
        source: "MLB Stats API completed games strictly before target date + exact-date active roster",
        completedGamesUsed: feeds.filter(Boolean).length,
        candidates
      }, { "cache-control": "public, max-age=60, s-maxage=300, stale-while-revalidate=600" });
    }

    if (q.type === "transactions") {
      const date = /^\d{4}-\d{2}-\d{2}$/.test(q.date || "") ? q.date : new Date().toISOString().slice(0,10);
      const url = `https://statsapi.mlb.com/api/v1/transactions?startDate=${encodeURIComponent(date)}&endDate=${encodeURIComponent(date)}&sportId=1`;
      const response = await fetch(url, { headers: { accept: "application/json", "user-agent": "MLB-Live-Command-Center/11.2" } });
      if (!response.ok) return send(response.status, await response.text());
      const data = await response.json();
      return send(200, {
        date,
        transactions: (data.transactions || []).map(t => ({
          id: t.id,
          date: t.date,
          effectiveDate: t.effectiveDate,
          typeCode: t.typeCode,
          description: t.description,
          person: t.person ? { id: t.person.id, name: t.person.fullName } : null,
          fromTeam: t.fromTeam ? { id: t.fromTeam.id, name: t.fromTeam.name } : null,
          toTeam: t.toTeam ? { id: t.toTeam.id, name: t.toTeam.name } : null
        }))
      });
    }

    if (q.type === "odds") {
      const apiKey = process.env.ODDS_API_KEY;
      if (!apiKey) return send(503, { error: "ODDS_API_KEY is not configured", events: [] });
      const url = `https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey=${encodeURIComponent(apiKey)}&regions=us&markets=h2h&oddsFormat=american&dateFormat=iso`;
      const response = await fetch(url, { headers: { accept: "application/json", "user-agent": "MLB-Live-Command-Center/11.2" } });
      const text = await response.text();
      if (!response.ok) return send(response.status, text);
      return send(200, {
        events: JSON.parse(text),
        quota: {
          remaining: response.headers.get("x-requests-remaining"),
          used: response.headers.get("x-requests-used"),
          last: response.headers.get("x-requests-last")
        }
      });
    }

    if (q.type === "peopleStats") {
      const ids = String(q.ids || "").split(",").filter(x => /^\d+$/.test(x)).slice(0, 100);
      const season = /^\d{4}$/.test(q.season || "") ? q.season : String(new Date().getFullYear());
      if (!ids.length) return send(200, { players: {} });

      const url = `https://statsapi.mlb.com/api/v1/people?personIds=${ids.join(",")}&hydrate=stats(group=[hitting,pitching],type=[season],season=${season})`;
      const response = await fetch(url, { headers: { accept: "application/json", "user-agent": "MLB-Live-Command-Center/11.2" } });
      if (!response.ok) return send(response.status, await response.text());
      const data = await response.json();
      const players = {};

      for (const person of (data.people || [])) {
        const record = { name: person.fullName, hitting: {}, pitching: {} };
        for (const statBlock of (person.stats || [])) {
          const group = statBlock.group?.displayName || statBlock.group?.displayName;
          const split = statBlock.splits?.[0]?.stat || {};
          if (group === "hitting") record.hitting = split;
          if (group === "pitching") record.pitching = split;
        }
        players[person.id] = record;
      }
      return send(200, { season, players });
    }

    return send(400, { error: "Use type=schedule, type=feed, type=rosters, type=bullpenContext, type=peopleStats, type=transactions, or type=odds" });
  } catch (error) {
    return send(502, {
      error: "Unable to reach MLB Stats API",
      detail: String(error && error.message ? error.message : error)
    });
  }
};