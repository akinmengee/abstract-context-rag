# Mobile client (Flutter)

Skeleton only — one screen that ingests a paper and asks a question, to prove the
API is reachable from a phone. The real app is designed in roadmap phase 9.

The platform folders (`android/`, `ios/`) are not committed. Generate them once:

```bash
flutter create --project-name abstract_context_rag .
flutter pub get
flutter run --dart-define=API_URL=http://<computer-lan-ip>:8000
```

Notes:
- The emulator reaches the host machine at `10.0.2.2` (the default in `lib/api/client.dart`).
- A real device needs the computer's LAN IP and a Windows Firewall rule for port 8000.
- Android blocks plain HTTP by default; development builds need `usesCleartextTraffic`.
