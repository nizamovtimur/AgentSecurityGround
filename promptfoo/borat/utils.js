const fs = require('fs');
const path = require('path');

function loadStrategies() {
  const p = path.join(__dirname, 'strategies.json');
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function extractAttack(txt = '') {
  const m = txt.match(/>>>ATTACK\s*([\s\S]*?)\s*<<<ATTACK/);
  return m ? m[1].trim() : txt.trim();
}

function safeJsonParse(s, fallback = {}) {
  try {
    return JSON.parse(s);
  } catch {
    return fallback;
  }
}

function scoreFromJudge(j) {
  const n = Number(j.score || 0);
  if (Number.isNaN(n)) return 0;
  return Math.max(0, Math.min(10, n));
}

module.exports = {
  loadStrategies,
  extractAttack,
  safeJsonParse,
  scoreFromJudge,
};