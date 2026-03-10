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
##### Build a release APK (single universal):
```bash
flutter clean
flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk
```

##### Build an Android App Bundle (recommended for Play Store):
```bash
flutter clean
flutter build appbundle --release
# Output: build/app/outputs/bundle/release/app-release.aab
```

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