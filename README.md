# Pocket Othello

A responsive Othello/Reversi web game for iPhone, Android, and desktop browsers. It is built with plain HTML, CSS, and JavaScript and is ready for GitHub Pages.

## Play Online

https://fdtdengineer.github.io/pocket_othello/

## Features

- Standard 8×8 Othello rules, legal-move hints, pass handling, and score tracking
- CPU opponents with Easy, Normal, and Hard difficulty levels
- Local two-player mode on one device
- Online two-player matches using a six-character room code
- Direct browser-to-browser game synchronization with PeerJS/WebRTC
- Responsive mobile layout with iPhone safe-area and Safari toolbar clearance
- Installable PWA and offline CPU/local play
- No build process

## Online Matches

One player selects **Online → Create Room** and shares the six-character code. The other player opens the same site, selects **Online**, enters the code, and joins.

Online matches use PeerJS Cloud for connection signaling and WebRTC for the game data. A small number of restrictive networks or symmetric NAT configurations may require a TURN server and can fail to connect.

## Run Locally

Serve the directory over HTTP:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Tests

```bash
npm test
npm run check
```

## GitHub Pages Deployment

The repository includes `.github/workflows/pages.yml`.

After pushing to `main`, open:

```text
Settings → Pages → Build and deployment → Source → GitHub Actions
```

The published URL will be:

```text
https://fdtdengineer.github.io/pocket_othello/
```

## Project Structure

```text
.
├── index.html
├── style.css
├── app.js
├── engine.js
├── ai.js
├── online.js
├── manifest.webmanifest
├── service-worker.js
├── icon.svg
├── tests/engine.test.mjs
└── .github/workflows/pages.yml
```

## License

MIT License
