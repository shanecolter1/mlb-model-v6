exports.handler = async function(event) {
  const headers = {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    "access-control-allow-origin": "*"
  };

  const send = (statusCode, data) => ({
    statusCode,
    headers,
    body: typeof data === "string" ? data : JSON.stringify(data)
  });

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

    return send(400, { error: "Use type=schedule, type=feed, type=rosters, type=peopleStats, type=transactions, or type=odds" });
  } catch (error) {
    return send(502, {
      error: "Unable to reach MLB Stats API",
      detail: String(error && error.message ? error.message : error)
    });
  }
};