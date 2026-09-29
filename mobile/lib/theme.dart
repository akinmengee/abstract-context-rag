// Same tokens as web/src/index.css: neutral surfaces, steel blue for brand
// chrome, and a warm gold reserved for citations - the product's actual
// differentiator (every answer traces back to a real passage) gets its own
// color instead of being folded into the one accent used for everything.
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

@immutable
class AppPalette extends ThemeExtension<AppPalette> {
  const AppPalette({
    required this.bgSidebar,
    required this.bgHover,
    required this.textMuted,
    required this.evidence,
    required this.evidenceBg,
  });

  final Color bgSidebar;
  final Color bgHover;
  final Color textMuted;
  final Color evidence;
  final Color evidenceBg;

  static const light = AppPalette(
    bgSidebar: Color(0xFFEEF1F0),
    bgHover: Color(0xFFE2E7E4),
    textMuted: Color(0xFF5C6B70),
    evidence: Color(0xFFA3762F),
    evidenceBg: Color(0xFFF3E6CF),
  );

  static const dark = AppPalette(
    bgSidebar: Color(0xFF0F1216),
    bgHover: Color(0xFF262B33),
    textMuted: Color(0xFF8B93A0),
    evidence: Color(0xFFD9A85C),
    evidenceBg: Color(0xFF3A2F1C),
  );

  @override
  AppPalette copyWith({
    Color? bgSidebar,
    Color? bgHover,
    Color? textMuted,
    Color? evidence,
    Color? evidenceBg,
  }) {
    return AppPalette(
      bgSidebar: bgSidebar ?? this.bgSidebar,
      bgHover: bgHover ?? this.bgHover,
      textMuted: textMuted ?? this.textMuted,
      evidence: evidence ?? this.evidence,
      evidenceBg: evidenceBg ?? this.evidenceBg,
    );
  }

  @override
  AppPalette lerp(ThemeExtension<AppPalette>? other, double t) {
    if (other is! AppPalette) return this;
    return AppPalette(
      bgSidebar: Color.lerp(bgSidebar, other.bgSidebar, t)!,
      bgHover: Color.lerp(bgHover, other.bgHover, t)!,
      textMuted: Color.lerp(textMuted, other.textMuted, t)!,
      evidence: Color.lerp(evidence, other.evidence, t)!,
      evidenceBg: Color.lerp(evidenceBg, other.evidenceBg, t)!,
    );
  }
}

extension AppPaletteContext on BuildContext {
  AppPalette get palette => Theme.of(this).extension<AppPalette>()!;
}

ThemeData buildLightTheme() {
  final base = ThemeData(
    useMaterial3: true,
    brightness: Brightness.light,
    colorScheme: const ColorScheme.light(
      primary: Color(0xFF3D6EA5),
      onPrimary: Colors.white,
      surface: Color(0xFFFFFFFF),
      onSurface: Color(0xFF14202E),
      outline: Color(0xFFDCE2DE),
      error: Color(0xFFB23A52),
    ),
    scaffoldBackgroundColor: const Color(0xFFF6F7F5),
  );
  return base.copyWith(
    textTheme: _textTheme(base.textTheme, const Color(0xFF14202E)),
    extensions: const [AppPalette.light],
  );
}

ThemeData buildDarkTheme() {
  final base = ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    colorScheme: const ColorScheme.dark(
      primary: Color(0xFF5B8FCB),
      onPrimary: Colors.white,
      surface: Color(0xFF1C2028),
      onSurface: Color(0xFFEDF1F6),
      outline: Color(0xFF272D36),
      error: Color(0xFFDD6A80),
    ),
    scaffoldBackgroundColor: const Color(0xFF14171C),
  );
  return base.copyWith(
    textTheme: _textTheme(base.textTheme, const Color(0xFFEDF1F6)),
    extensions: const [AppPalette.dark],
  );
}

TextTheme _textTheme(TextTheme base, Color color) {
  final body = GoogleFonts.ibmPlexSansTextTheme(base)
      .apply(bodyColor: color, displayColor: color);
  final display = GoogleFonts.newsreaderTextTheme(base);
  return body.copyWith(
    headlineSmall: display.headlineSmall?.copyWith(color: color),
    titleLarge: display.titleLarge?.copyWith(color: color),
    titleMedium: display.titleMedium?.copyWith(color: color),
  );
}
