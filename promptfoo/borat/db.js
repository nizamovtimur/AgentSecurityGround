const Database = require('better-sqlite3');

function initDb(path) {
  const db = new Database(path);

  db.prepare(`
    CREATE TABLE IF NOT EXISTS memory (
      id INTEGER PRIMARY KEY,
      pluginId TEXT,
      prompt TEXT,
      score INTEGER,
      response TEXT,
      createdAt TEXT
    )
  `).run();

  db.close();
}

function loadMemory(path, pluginId) {
  const db = new Database(path);

  const rows = db.prepare(`
    SELECT prompt, score, response
    FROM memory
    WHERE pluginId = ?
    ORDER BY score DESC
    LIMIT 5
  `).all(pluginId);

  db.close();
  return rows;
}

function saveMemory(path, pluginId, item) {
  const db = new Database(path);

  db.prepare(`
    INSERT INTO memory(pluginId,prompt,score,response,createdAt)
    VALUES(?,?,?,?,datetime('now'))
  `).run(
    pluginId,
    item.prompt,
    item.score,
    item.response
  );

  db.close();
}

module.exports = {
  initDb,
  loadMemory,
  saveMemory,
};