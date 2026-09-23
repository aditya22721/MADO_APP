import 'package:flutter/foundation.dart';

class AppConstants {
  static const String appName = 'MADO Assistant';

  // Auto-detects platform for backend URL
  static String get apiBaseUrl {
    if (kIsWeb) return 'http://localhost:8000';
    if (defaultTargetPlatform == TargetPlatform.android) {
      return 'http://10.0.2.2:8000'; // emulator
    }
    if (defaultTargetPlatform == TargetPlatform.iOS) {
      return 'http://localhost:8000';
    }
    // ⚠️ REPLACE with your PC's IP for physical phone
    return 'http://192.168.0.103:8000';
  }

  static const String chatEndpoint = '/api/v1/chat';
  static const String emergencyEndpoint = '/api/v1/emergency';
  static const String contactsEndpoint = '/api/v1/contacts';
  static const String locationEndpoint = '/api/v1/location';
  static const String weatherEndpoint = '/api/v1/weather';
  static const String healthEndpoint = '/health';
}