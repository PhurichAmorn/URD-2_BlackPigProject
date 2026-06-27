# blackpig

An URD2 project Application.

---

## Getting Started

#### How to install Flutter
- https://www.fluttermapp.com/articles/install-flutter-mac
- https://www.youtube.com/watch?v=QG9bw4rWqrg&ab_channel=FlutterMapp
- https://www.youtube.com/watch?v=KdO9B_CZmzo&ab_channel=ProgrammingKnowledge

#### How to add packages
1. Find the package you want to add
2. Click the copy button (on the right of package name)
3. Go to pubspec.yaml
4. Find dependencies
5. Add under the comment
6. Run "flutter pub get" in terminal

*** Run "flutter pub get" everytime after making change in pubspec.yaml file ***

*** Need to import packages that will be used in that page before start ***

#### How to build the app

The project uses **Flutter flavors** to separate dev and production builds:

| Flavor | App Name     | App ID                    | Use Case                                    |
|--------|-------------|---------------------------|---------------------------------------------|
| `dev`  | DooMoo Dev  | `com.example.blackpig.dev` | Testing — debug features enabled by default |
| `prod` | DooMoo      | `com.example.blackpig`    | Production — clean release build            |

##### Run on device/emulator
```bash
# Dev flavor (debug features, testing)
flutter run --flavor dev --dart-define=APP_FLAVOR=dev

# Production flavor (clean)
flutter run --flavor prod --dart-define=APP_FLAVOR=prod
```

##### Build a release APK (single universal):
```bash
# Dev APK for internal testers
flutter clean
flutter build apk --release --flavor dev --dart-define=APP_FLAVOR=dev
# Output: build/app/outputs/flutter-apk/app-dev-release.apk

# Production APK for Play Store
flutter clean
flutter build apk --release --flavor prod --dart-define=APP_FLAVOR=prod
# Output: build/app/outputs/flutter-apk/app-prod-release.apk
```

##### Build an Android App Bundle (recommended for Play Store):
```bash
flutter clean
flutter build appbundle --release --flavor prod --dart-define=APP_FLAVOR=prod
# Output: build/app/outputs/bundle/prodRelease/app-prod-release.aab
```

##### Testing with debug features
Debug features (camera metadata, PCA overlay, auto height estimation, etc.) are:
- **Dev flavor**: Visible by default (even in release builds) — great for internal testing
- **Prod flavor**: Only visible in `flutter run` (debug build); hidden in release builds

You can also toggle them at runtime by **long-pressing the "รายละเอียด" title** on the DetailsPage.

*Focus development on Android Device*

---

Useful Resources
- [Material component widgets](https://docs.flutter.dev/ui/widgets/material)

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.