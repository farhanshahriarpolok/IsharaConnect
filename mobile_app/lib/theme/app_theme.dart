import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppColors {
  static const Color bgDark = Color(0xFF0F172A);
  static const Color panelDark = Color(0xFF1E293B);
  static const Color surfaceDark = Color(0xFF334155);
  static const Color surfaceLight = Color(0xFF475569);

  static const Color cyanAccent = Color(0xFF06B6D4);
  static const Color cyanGlow = Color(0xFF22D3EE);
  static const Color emeraldSuccess = Color(0xFF10B981);
  static const Color coralError = Color(0xFFF43F5E);
  static const Color amberWarning = Color(0xFFF59E0B);
  static const Color indigoAccent = Color(0xFF6366F1);

  static const Color textPrimary = Color(0xFFF8FAFC);
  static const Color textSecondary = Color(0xFF94A3B8);
  static const Color textMuted = Color(0xFF64748B);

  static const Color glassBg = Color(0xB31E293B);
  static const Color glassBorder = Color(0x1AFFFFFF);
}

class AppTheme {
  static ThemeData get darkTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: AppColors.bgDark,
      primaryColor: AppColors.cyanAccent,
      colorScheme: const ColorScheme.dark(
        primary: AppColors.cyanAccent,
        secondary: AppColors.emeraldSuccess,
        surface: AppColors.panelDark,
        error: AppColors.coralError,
        onPrimary: AppColors.bgDark,
        onSurface: AppColors.textPrimary,
      ),
      textTheme: GoogleFonts.interTextTheme(
        ThemeData.dark().textTheme,
      ).apply(
        bodyColor: AppColors.textPrimary,
        displayColor: AppColors.textPrimary,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: AppColors.panelDark,
        elevation: 0,
        centerTitle: true,
        titleTextStyle: TextStyle(
          color: AppColors.cyanAccent,
          fontSize: 18,
          fontWeight: FontWeight.bold,
        ),
      ),
      cardTheme: CardTheme(
        color: AppColors.panelDark,
        elevation: 4,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: const BorderSide(color: AppColors.glassBorder, width: 1),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.cyanAccent,
          foregroundColor: AppColors.bgDark,
          textStyle: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          elevation: 2,
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: AppColors.bgDark,
        hintStyle: const TextStyle(color: AppColors.textMuted),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: AppColors.surfaceDark),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: AppColors.surfaceDark),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: AppColors.cyanAccent, width: 2),
        ),
      ),
    );
  }

  static BoxDecoration glassCardDecoration({Color? borderColor, double borderRadius = 16}) {
    return BoxDecoration(
      color: AppColors.glassBg,
      borderRadius: BorderRadius.circular(borderRadius),
      border: Border.all(color: borderColor ?? AppColors.glassBorder, width: 1),
      boxShadow: [
        BoxShadow(
          color: Colors.black.withOpacity(0.2),
          blurRadius: 10,
          offset: const Offset(0, 4),
        ),
      ],
    );
  }
}
