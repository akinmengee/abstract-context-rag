# Multi-stage: build the static SPA with Node, serve it with nginx - no
# Node runtime in the final image, and the same image works on any host
# since nginx.conf proxies /api/ instead of a backend URL baked in at build time.
FROM node:20-slim AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
