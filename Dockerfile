# Build from this folder: docker build -t magic-royale .
# The match runs it with no network, a read-only root (writable /tmp), 1 CPU, 512 MB, and the map at /arena.
# Pinned by digest so the final rebuild gets exactly the image this was tested on.
FROM oven/bun:1.4.4-slim@sha256:1846adf22496bd711036aae8be147e57bdebbaea2e5264e3e0b5af1db187d9a4
WORKDIR /bot
COPY bot.ts mechfight.ts package.json ./
COPY src ./src
# Bundle to one file at build time: a broken import or syntax error fails the build, not the match.
RUN bun build /bot/bot.ts --target=bun --outfile=/bot/bot.js
ENTRYPOINT ["bun", "/bot/bot.js"]
