(() => {
  'use strict';

  // Display-only park context. This module MUST NOT write to any prediction/model state.
  const CITY_BY_VENUE = {
    'american family field':'Milwaukee, WI',
    'angel stadium':'Anaheim, CA',
    'busch stadium':'St. Louis, MO',
    'chase field':'Phoenix, AZ',
    'citi field':'Queens, NY',
    'citizens bank park':'Philadelphia, PA',
    'comerica park':'Detroit, MI',
    'coors field':'Denver, CO',
    'daikin park':'Houston, TX',
    'fenway park':'Boston, MA',
    'george m. steinbrenner field':'Tampa, FL',
    'globe life field':'Arlington, TX',
    'great american ball park':'Cincinnati, OH',
    'kauffman stadium':'Kansas City, MO',
    'loandepot park':'Miami, FL',
    'loan depot park':'Miami, FL',
    'nationals park':'Washington, DC',
    'oracle park':'San Francisco, CA',
    'oriole park at camden yards':'Baltimore, MD',
    'camden yards':'Baltimore, MD',
    'petco park':'San Diego, CA',
    'pnc park':'Pittsburgh, PA',
    'progressive field':'Cleveland, OH',
    'rate field':'Chicago, IL',
    'rogers centre':'Toronto, ON',
    'sutter health park':'West Sacramento, CA',
    'target field':'Minneapolis, MN',
    't-mobile park':'Seattle, WA',
    't mobile park':'Seattle, WA',
    'truist park':'Atlanta, GA',
    'tropicana field':'St. Petersburg, FL',
    'uniqlo field at dodger stadium':'Los Angeles, CA',
    'dodger stadium':'Los Angeles, CA',
    'wrigley field':'Chicago, IL',
    'yankee stadium':'Bronx, NY',
    'bristol motor speedway':'Bristol, TN',
    'journey bank ballpark':'Williamsport, PA',
    'las vegas ballpark':'Las Vegas, NV',
    'oakland coliseum':'Oakland, CA',
    'rickwood field':'Birmingham, AL',
    'field of dreams':'Dyersville, IA'
  };

  const normalize = value => String(value || '')
    .toLowerCase()
    .replace(/&/g, 'and')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();

  const ordinal = n => {
    const value = Math.max(0, Math.min(100, Math.round(Number(n) || 0)));
    const mod100 = value % 100;
    const suffix = mod100 >= 11 && mod100 <= 13 ? 'th' : ({1:'st',2:'nd',3:'rd'}[value % 10] || 'th');
    return `${value}${suffix}`;
  };

  let parkMap = new Map();
  let loaded = false;
  let loading = null;

  async function loadParkPercentiles() {
    if (loaded) return parkMap;
    if (loading) return loading;
    loading = (async () => {
      try {
        const year = new Date().getFullYear();
        const response = await fetch(`/.netlify/functions/savant?type=parkFactors&year=${year}`, { cache: 'no-store' });
        if (!response.ok) throw new Error(`park factors ${response.status}`);
        const data = await response.json();
        parkMap = new Map((data.parks || []).map(row => [normalize(row.venue), row]));
      } catch (error) {
        console.warn('Park percentile display unavailable', error);
        parkMap = new Map();
      } finally {
        loaded = true;
      }
      return parkMap;
    })();
    return loading;
  }

  function gameByPk(gamePk) {
    try {
      if (typeof cache !== 'undefined' && Array.isArray(cache)) {
        return cache.find(g => String(g?.gamePk) === String(gamePk)) || null;
      }
    } catch (_) {}
    return null;
  }

  function venueName(g) {
    return g?.gameData?.venue?.name || g?.venue?.name || g?.gameData?.teams?.home?.venue?.name || '';
  }

  function venueLocation(g, venue) {
    const loc = g?.gameData?.venue?.location || g?.venue?.location || {};
    const city = loc.city || loc.cityName || '';
    const state = loc.stateAbbrev || loc.state || loc.province || '';
    if (city) return state ? `${city}, ${state}` : city;
    return CITY_BY_VENUE[normalize(venue)] || '';
  }

  function ensureStyle() {
    if (document.getElementById('park-display-style')) return;
    const style = document.createElement('style');
    style.id = 'park-display-style';
    style.textContent = `
      .park-display-line{margin-top:3px;text-align:center;color:var(--muted);font-size:clamp(9px,2.45vw,11px);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
      .park-display-line .park-pct{color:var(--text);font-weight:750}
    `;
    document.head.appendChild(style);
  }

  function updateCard(card) {
    const gamePk = card?.dataset?.gameCard;
    if (!gamePk) return;
    const g = gameByPk(gamePk);
    if (!g) return;
    const venue = venueName(g);
    if (!venue) return;
    const location = venueLocation(g, venue);
    const row = parkMap.get(normalize(venue));

    const header = card.querySelector('.compact-game-header') || card.querySelector('.gamehead');
    if (!header) return;
    let line = header.querySelector('.park-display-line');
    if (!line) {
      line = document.createElement('div');
      line.className = 'park-display-line';
      header.appendChild(line);
    }

    const parts = [venue];
    if (location) parts.push(location);
    line.textContent = parts.join(' · ');
    if (row && Number.isFinite(Number(row.percentile))) {
      const sep = document.createTextNode(' · ');
      const span = document.createElement('span');
      span.className = 'park-pct';
      span.textContent = `Park offense ${ordinal(row.percentile)} pct`;
      line.appendChild(sep);
      line.appendChild(span);
    }
  }

  function updateAll() {
    ensureStyle();
    document.querySelectorAll('[data-game-card]').forEach(updateCard);
  }

  function startObserver() {
    updateAll();
    const observer = new MutationObserver(() => updateAll());
    observer.observe(document.body, { childList: true, subtree: true });
  }

  document.addEventListener('DOMContentLoaded', async () => {
    await loadParkPercentiles();
    startObserver();
  });
})();
