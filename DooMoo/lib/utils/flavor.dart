class Flavor {
  static const String name = String.fromEnvironment(
    'APP_FLAVOR',
    defaultValue: 'prod',
  );
  static bool get isDev => name == 'dev';
  static bool get isProd => name == 'prod';
}
