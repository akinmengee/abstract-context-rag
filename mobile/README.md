# Mobile client (Flutter)

Same account, same conversation history as `web/` - accounts and chat
history are backend-side (`backend/abstractrag/accounts/`), so a
conversation started on web continues here, and vice versa, through the
same JWT bearer API (see
[`backend/abstractrag/api/README.md`](../backend/abstractrag/api/README.md)
for the endpoints). Same design tokens as `web/src/index.css` too (see
`lib/theme.dart`): neutral surfaces, steel blue for brand chrome, warm gold
reserved for citations. `lib/screens/` mirrors `web/src/pages/` one for one
(login, register, chat); `lib/api/sse.dart` is the same hand-rolled
POST-based Server-Sent-Events reader as `web/src/api/sse.ts`, for the same
reason (Flutter's `http` package has no native `EventSource` either).

**Prerequisites:** Flutter 3.5+ / Dart 3.5+ (`environment.sdk` in
`pubspec.yaml`). Windows desktop and Android are both supported from this
setup; see "iOS" at the bottom for why iOS isn't.

The platform folders (`android/`, `windows/`, `ios/`) are not committed -
they're `flutter create` boilerplate, not source. Generate them once per
machine:

```bash
cd mobile
flutter create --platforms=windows,android --org com.abstractcontextrag .
flutter pub get
dart run flutter_launcher_icons
```

That last step replaces `flutter create`'s default Flutter logo with the brand badge
(`assets/app_icon.png`, same source as `web/public/favicon.png`) in the freshly
generated `android/` and `windows/` folders - config lives in `pubspec.yaml`'s
`flutter_launcher_icons:` section. Redo it if you ever regenerate those folders.

**If the repo path has a non-ASCII character** (this one does: `OneDrive\Masaüstü`),
the Android Gradle plugin refuses to build with a "project path contains
non-ASCII characters" error. Add this line to the freshly generated
`android/gradle.properties` (it will not survive regenerating the `android/`
folder, so re-add it if that ever happens):

```
android.overridePathCheck=true
```

`android/app/src/main/AndroidManifest.xml` also needs two things `flutter
create` doesn't add by default - already applied once, redo them if you
regenerate `android/`:
- `<uses-permission android:name="android.permission.INTERNET" />`
- `android:usesCleartextTraffic="true"` on `<application>` - the backend runs
  on plain HTTP on the LAN, and Android blocks cleartext traffic by default
  since API 28.

## Running it

**During development, run it as a Windows desktop app** - fastest loop, no
phone or emulator needed, and this is what "control it from the computer"
means day to day:

```bash
flutter run -d windows --dart-define=API_URL=http://localhost:8000
```

**On a real Android phone over the same Wi-Fi:** find this computer's LAN IP
(`ipconfig`, the Wi-Fi adapter's IPv4 address - e.g. `192.168.1.106`), make
sure `abstractrag serve` or the Docker `backend` service is reachable at
that IP and port 8000 (both already bind `0.0.0.0`, so nothing to change
there), and allow it through Windows Firewall if prompted. Then either:

```bash
# Connected over USB with USB debugging on:
flutter run --dart-define=API_URL=http://192.168.1.106:8000

# Or build an APK and install it without a cable:
flutter build apk --debug --dart-define=API_URL=http://192.168.1.106:8000
```

The debug APK lands at `build/app/outputs/flutter-apk/app-debug.apk` - send
it to the phone however's convenient (USB copy, a cloud drive, messaging
yourself the file) and open it there to install; Android will ask to allow
installs from that source the first time.

**Android emulator** reaches the host machine at `10.0.2.2` (the default in
`lib/api/client.dart` if `--dart-define=API_URL` is omitted).

**iOS**: not supported from this setup - building an `.ipa` needs Xcode and
a Mac, which Windows cannot provide. A `.apk` will never install on an
iPhone either; if iOS support is ever wanted, that's a separate effort on
different hardware, not a build flag here.

See the root [README](../README.md#mobile-flutter) for a screenshot and
where this fits in the rest of the project.
