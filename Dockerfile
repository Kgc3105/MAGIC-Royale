# Build from this folder: docker build -t magic-royale .
# The match runs it with no network, a read-only root (writable /tmp), 1 CPU, 512 MB, and the map at /arena.
# Pinned by digest so the final rebuild gets exactly the image this was tested on.
FROM oven/bun:1.4.4-slim@sha256:1846adf22496bd711036aae8be147e57bdebbaea2e5264e3e0b5af1db187d9a4
WORKDIR /bot
COPY bot.ts mechfight.ts package.json ./
COPY src ./src
# Bundle to one file at build time: a broken import or syntax error fails the build, not the match.
RUN bun build /bot/bot.ts --target=bun --outfile=/bot/bot.js
# The match gives us one CPU. Left to itself the runtime compiles and collects garbage on extra
# threads, which can use up that allowance and freeze the bot for tens of milliseconds, long enough
# to miss a 50 ms answer. Compile on the main thread, skip the slowest compiler tier, one GC marker.
ENV BUN_JSC_useConcurrentJIT=0 BUN_JSC_useFTLJIT=0 BUN_JSC_numberOfGCMarkers=1
# Variant settings for this submission (see MECH_TUNE in src/brain.ts).
ENV MECH_TUNE='{"salt":3}'
ENTRYPOINT ["bun", "/bot/bot.js"]
