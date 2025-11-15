# Watch HR Backend

Minimal API to ingest and serve heart rate and IBI data from a Galaxy Watch (standalone or via phone).

## Endpoints
- POST `/api/hr` → ingest a reading
- GET `/api/hr/latest` → latest reading (optionally filter by `deviceId`)
- GET `/api/hr/history` → recent readings (optionally filter by `deviceId`)
- GET `/api/hr/stream` → Server-Sent Events stream
- GET `/healthz` → health check

## Auth
Optional Bearer token via `AUTH_TOKEN`. If set, clients must send:

```
Authorization: Bearer <AUTH_TOKEN>
```

## Run locally
```
cp .env.example .env
npm install
npm run dev
```

## Docker
```
docker build -t watch-hr-backend .
docker run -p 3000:3000 --env-file .env watch-hr-backend
```

## ngrok (development)
Expose local server over HTTPS:
```
ngrok http 3000
```
Use the HTTPS URL in your watch app.

## Notes
- Storage is in-memory; it resets on restart.
- Add a DB (e.g., Postgres/SQLite) later if you need persistence.


