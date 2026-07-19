# finetune-gallery — multi-stage: Svelte build → Go build → minimal runtime.

FROM node:22-alpine AS web
WORKDIR /src/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build          # emits ../server/webdist
# outDir escapes /src/web — build into a sibling dir the next stage copies.

FROM golang:1.26-alpine AS build
WORKDIR /src/server
COPY server/go.mod server/go.sum ./
RUN go mod download
COPY server/ ./
COPY --from=web /src/server/webdist ./webdist
RUN CGO_ENABLED=0 go build -ldflags='-s -w' -o /out/finetune-gallery .

FROM alpine:3.21
RUN adduser -D -u 1000 app
COPY --from=build /out/finetune-gallery /usr/local/bin/finetune-gallery
USER app
EXPOSE 8092
ENTRYPOINT ["finetune-gallery"]
