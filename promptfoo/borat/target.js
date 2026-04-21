const { callModel } = require('./llm');

async function runTarget(provider, prompt) {
  const txt = await callModel(provider, [
    { role: "user", content: prompt }
  ]);

  return txt;
}

module.exports = { runTarget };