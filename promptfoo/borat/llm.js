const OpenAI = require('openai');

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function callModel(model, messages) {
  const resp = await client.chat.completions.create({
    model,
    messages,
    temperature: 0.7,
  });

  return resp.choices[0].message.content;
}

module.exports = { callModel };