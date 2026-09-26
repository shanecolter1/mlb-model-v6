export const LINEUP_SOURCE_PRIORITY = Object.freeze({
  MLB_CONFIRMED: 100,
  TEAM_BEAT_CONFIRMED: 95,
  ROTOWIRE_CONFIRMED: 90,
  ROTOWIRE_PROJECTED: 80,
  ROSTERRESOURCE_PROJECTED: 70,
  MLB_PREVIOUS_GAME: 40,
});

export const STARTER_SOURCE_PRIORITY = Object.freeze({
  TEAM_BEAT_CONFIRMED: 100,
  ROTOWIRE_CONFIRMED: 90,
  ROTOWIRE_PROJECTED: 80,
  ROSTERRESOURCE_PROJECTED: 70,
  MLB_PROBABLE: 50,
});

export function normalizeName(value) {
  return String(value || '')
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

export function sameName(a,b) {
  return Boolean(normalizeName(a) && normalizeName(a) === normalizeName(b));
}

export function isNine(lineup) {
  return Array.isArray(lineup) && lineup.length === 9 && new Set(lineup.map(normalizeName)).size === 9;
}

export function lineupDelta(primary, comparator) {
  if (!isNine(primary) || !isNine(comparator)) return {
    available:false,
    exactOrderMatches:null,
    top4ExactMatches:null,
    top4SamePlayers:null,
    nineManOverlap:null,
    changedSlots:[],
  };
  const p=primary.map(normalizeName), c=comparator.map(normalizeName);
  const exactOrderMatches=p.filter((x,i)=>x===c[i]).length;
  const top4ExactMatches=p.slice(0,4).filter((x,i)=>x===c[i]).length;
  const cTop=new Set(c.slice(0,4));
  const top4SamePlayers=p.slice(0,4).filter(x=>cTop.has(x)).length;
  const cAll=new Set(c);
  const nineManOverlap=p.filter(x=>cAll.has(x)).length;
  const changedSlots=p.map((x,i)=>x===c[i]?null:{slot:i+1,primary:primary[i],comparator:comparator[i]}).filter(Boolean);
  return {available:true,exactOrderMatches,top4ExactMatches,top4SamePlayers,nineManOverlap,changedSlots};
}

export function lineupConfidence({source, primary, rosterResource, previousGame}) {
  const rr=lineupDelta(primary,rosterResource);
  const prev=lineupDelta(primary,previousGame);

  if (source === 'TEAM_BEAT_CONFIRMED' || source === 'ROTOWIRE_CONFIRMED' || source === 'MLB_CONFIRMED') {
    return {level:'HIGH',reason:'Externally confirmed or MLB-confirmed batting order',rosterResourceDelta:rr,previousGameDelta:prev,needsReview:false};
  }

  if (source === 'ROTOWIRE_PROJECTED') {
    if (rr.available) {
      if (rr.top4ExactMatches === 4) return {level:'HIGH',reason:'RotoWire expected lineup agrees exactly with RosterResource in batting slots 1-4',rosterResourceDelta:rr,previousGameDelta:prev,needsReview:false};
      if (rr.top4SamePlayers === 4) return {level:'MEDIUM_HIGH',reason:'RotoWire and RosterResource agree on top-four personnel but not exact order',rosterResourceDelta:rr,previousGameDelta:prev,needsReview:false};
      if (rr.top4SamePlayers >= 3) return {level:'MEDIUM',reason:'One top-four personnel disagreement between RotoWire and RosterResource',rosterResourceDelta:rr,previousGameDelta:prev,needsReview:true};
      return {level:'LOW',reason:'Material top-four disagreement between RotoWire and RosterResource',rosterResourceDelta:rr,previousGameDelta:prev,needsReview:true};
    }
    return {level:'MEDIUM',reason:'RotoWire expected lineup available without RosterResource cross-check',rosterResourceDelta:rr,previousGameDelta:prev,needsReview:false};
  }

  if (source === 'ROSTERRESOURCE_PROJECTED') {
    return {level:'MEDIUM_LOW',reason:'RosterResource platoon projection used because no game-specific RotoWire lineup was available',rosterResourceDelta:rr,previousGameDelta:prev,needsReview:true};
  }

  if (source === 'MLB_PREVIOUS_GAME') {
    return {level:'LOW',reason:'Previous-game lineup fallback; no current third-party projection was available',rosterResourceDelta:rr,previousGameDelta:prev,needsReview:true};
  }

  return {level:'MISSING',reason:'No usable provisional lineup source',rosterResourceDelta:rr,previousGameDelta:prev,needsReview:true};
}

export function chooseLineup({teamBeat,rotowireConfirmed,rotowireProjected,rosterResource,previousGame}) {
  const candidates=[
    ['TEAM_BEAT_CONFIRMED',teamBeat],
    ['ROTOWIRE_CONFIRMED',rotowireConfirmed],
    ['ROTOWIRE_PROJECTED',rotowireProjected],
    ['ROSTERRESOURCE_PROJECTED',rosterResource],
    ['MLB_PREVIOUS_GAME',previousGame],
  ];
  for(const [source,lineup] of candidates){
    if(isNine(lineup)) return {source,lineup:[...lineup],priority:LINEUP_SOURCE_PRIORITY[source]};
  }
  return {source:'MISSING',lineup:null,priority:0};
}

export function chooseStarter({teamBeat,rotowireConfirmed,rotowireProjected,rosterResource,mlbProbable}) {
  const candidates=[
    ['TEAM_BEAT_CONFIRMED',teamBeat],
    ['ROTOWIRE_CONFIRMED',rotowireConfirmed],
    ['ROTOWIRE_PROJECTED',rotowireProjected],
    ['ROSTERRESOURCE_PROJECTED',rosterResource],
    ['MLB_PROBABLE',mlbProbable],
  ];
  for(const [source,starter] of candidates){
    if(starter?.name) return {source,starter:{...starter},priority:STARTER_SOURCE_PRIORITY[source]};
  }
  return {source:'MISSING',starter:null,priority:0};
}

function decodeHtml(s){
  return String(s||'')
    .replace(/&nbsp;/gi,' ')
    .replace(/&amp;/gi,'&')
    .replace(/&quot;/gi,'"')
    .replace(/&#39;|&apos;/gi,"'")
    .replace(/&ndash;|&#8211;/gi,'–')
    .replace(/&mdash;|&#8212;/gi,'—')
    .replace(/&aacute;/gi,'á').replace(/&eacute;/gi,'é').replace(/&iacute;/gi,'í').replace(/&oacute;/gi,'ó').replace(/&uacute;/gi,'ú')
    .replace(/&ntilde;/gi,'ñ');
}

function stripTags(s){
  return decodeHtml(String(s||'').replace(/<script[\s\S]*?<\/script>/gi,' ').replace(/<style[\s\S]*?<\/style>/gi,' ').replace(/<[^>]+>/g,' ').replace(/\s+/g,' ').trim());
}

export function parseRosterResourceProjectedLineup(html, hand='R') {
  const side=String(hand||'R').toUpperCase().startsWith('L')?'LHP':'RHP';
  const marker=new RegExp(`Go-To\\s+Starting\\s+Lineup\\s+vs\\s+${side}`,'i');
  const match=marker.exec(html);
  if(!match) return null;
  const after=html.slice(match.index);
  const nextMarker=/Go-To\s+Starting\s+Lineup\s+vs\s+(?:RHP|LHP)|>Bench</i;
  const rest=after.slice(match[0].length);
  const next=nextMarker.exec(rest);
  const section=next?rest.slice(0,next.index):rest.slice(0,120000);
  const rows=[...section.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/gi)].map(m=>m[1]);
  const result=[];
  for(const row of rows){
    const cells=[...row.matchAll(/<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/gi)].map(m=>stripTags(m[1]));
    if(cells.length<4) continue;
    const order=Number(cells[0]);
    if(!(order>=1&&order<=9)) continue;
    const links=[...row.matchAll(/<a\b[^>]*>([\s\S]*?)<\/a>/gi)].map(m=>stripTags(m[1])).filter(Boolean);
    let player=links.find(x=>/[A-Za-zÁÉÍÓÚÑáéíóúñ]/.test(x) && !/^\d+$/.test(x));
    if(!player) player=cells[3];
    if(player) result[order-1]=player;
  }
  return isNine(result)?result:null;
}

export function starterConflict(selected, mlbProbable) {
  if(!selected?.starter?.name || !mlbProbable?.name) return {conflict:false};
  const conflict=!sameName(selected.starter.name,mlbProbable.name);
  return {conflict,selected:selected.starter.name,mlbProbable:mlbProbable.name,selectedSource:selected.source};
}
