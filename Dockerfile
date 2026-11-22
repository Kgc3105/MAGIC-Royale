# Variant of our entry built on the open MIT-licensed giddygiddyboy controller (see giddy/LICENSE and README).
# Pinned by digest so the final rebuild gets exactly the toolchain this was tested with.
FROM golang:1.26.1-bookworm@sha256:ab3d6955bbc813a0f3fdf220c1d817dd89c0b3f283777db8ece4a32fe7858edd AS build
WORKDIR /src
COPY giddy/go.mod ./
COPY giddy/mech ./mech
COPY giddy/cmd/bot ./cmd/bot
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o /mechbot ./cmd/bot

FROM scratch
COPY --from=build /mechbot /mechbot
COPY giddy/LICENSE /LICENSE.giddygiddyboy
# Variant settings for this submission (see giddy/mech/tune.go).
ENV MECH_SALT=0 MECH_SQUAD=""
ENTRYPOINT ["/mechbot"]
