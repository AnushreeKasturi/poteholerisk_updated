const express = require('express');
const path    = require('path');
const fetch = require('node-fetch');
require('dotenv').config();

const app  = express();
const PORT = process.env.PORT || 3000;
const OWM_KEY = process.env.OPENWEATHER_API_KEY;

if (!OWM_KEY) {
  console.error('\n❌  Missing OPENWEATHER_API_KEY in .env file.\n   Copy .env.example → .env and add your key.\n');
  process.exit(1);
}

// Serve the frontend
app.use(express.static(path.join(__dirname, 'public')));
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ── Weather proxy ──────────────────────────────────────────────
// Browser calls  GET /api/weather?lat=12.97&lon=77.59
// Server injects the real API key and forwards to OpenWeather.
// The key is NEVER sent to the browser.
app.get('/api/weather', async (req, res) => {
  const lat = parseFloat(req.query.lat) || 12.9716;
  const lon = parseFloat(req.query.lon) || 77.5946;

  const url =
    `https://api.openweathermap.org/data/2.5/weather` +
    `?lat=${lat}&lon=${lon}&appid=${OWM_KEY}&units=metric`;

  try {
    const upstream = await fetch(url);
    if (!upstream.ok) {
      const body = await upstream.text();
      return res.status(upstream.status).json({ error: body });
    }
    const data = await upstream.json();
    res.json(data);
  } catch (err) {
    console.error('Weather fetch error:', err.message);
    res.status(500).json({ error: 'Failed to reach OpenWeather API' });
  }
});

app.listen(PORT, () => {
  console.log(`\n✅  PotholeRisk server running at http://localhost:${PORT}\n`);
});
