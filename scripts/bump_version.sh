#!/usr/bin/env bash
# Set the cache-busting version tag in docs/index.html (style.css?v=, app.js?v=) to the
# current UTC time. Run before committing a deploy; app.js propagates it to every data
# file and the OpenPOM worker.
set -euo pipefail
cd "$(dirname "$0")/.."
V="$(date -u +%Y%m%d%H%M)"
sed -i '' -E "s/(style\.css|app\.js)\?v=[^\"]*/\1?v=$V/g" docs/index.html
grep -o -E '(style\.css|app\.js)\?v=[^"]*' docs/index.html
