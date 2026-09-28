import assert from 'node:assert/strict';
import { test } from 'node:test';
import { selectI2VenueProfile } from '../src/model/i2_venue_profile.js';

const profiles = [
  {venue_name:'Sutter Health Park',team:'ATHLETICS'},
  {venue_name:'Oakland Coliseum',team:'ATHLETICS'},
];

test('matches the actual venue even when the club has moved parks', () => {
  const game = {venue:{name:'Sutter Health Park'},teams:{home:{team:{name:'Athletics'}}}};
  assert.equal(selectI2VenueProfile(game, profiles), profiles[0]);
});

test('does not borrow a home club park when a temporary venue lacks a profile', () => {
  const game = {venue:{name:'Neutral Exhibition Park'},teams:{home:{team:{name:'Athletics'}}}};
  assert.equal(selectI2VenueProfile(game, profiles), null);
});

test('matches punctuation variants but rejects missing or ambiguous venue names', () => {
  assert.equal(selectI2VenueProfile({venue:{name:'Sutter-Health Park'}}, profiles), profiles[0]);
  assert.equal(selectI2VenueProfile({teams:{home:{team:{name:'Athletics'}}}}, profiles), null);
  assert.equal(selectI2VenueProfile({venue:{name:'Oakland Coliseum'}}, [...profiles, profiles[1]]), null);
});
