import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../models/contact_model.dart';
import '../utils/constants.dart';

class EmergencyService {
  /// Sends emergency alerts via the BACKEND (Twilio SMS + Gmail Email).
  /// No native app opening — everything happens server-side.
  Future<bool> alertContacts({
    required String userId,
    required List<EmergencyContact> contacts,
    required String message,
    required String address,
    required double lat,
    required double lng,
  }) async {
    if (contacts.isEmpty) {
      Fluttertoast.showToast(
        msg: '⚠️ No emergency contacts saved',
        backgroundColor: Colors.orange,
      );
      return false;
    }

    try {
      // Send alert via backend — backend will send SMS + Email + WhatsApp
      final res = await http.post(
        Uri.parse(
            '${AppConstants.apiBaseUrl}${AppConstants.emergencyEndpoint}'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'user_id': userId,
          'message': message,
          'location': {
            'lat': lat,
            'lng': lng,
            'address': address,
          },
        }),
      ).timeout(const Duration(seconds: 30));

      if (res.statusCode == 200) {
        Fluttertoast.showToast(
          msg: '✅ SMS + Email sent to ${contacts.length} contact(s)',
          backgroundColor: Colors.green,
          textColor: Colors.white,
          toastLength: Toast.LENGTH_LONG,
        );
        return true;
      } else {
        Fluttertoast.showToast(
          msg: '⚠️ Alert failed (${res.statusCode})',
          backgroundColor: Colors.orange,
          textColor: Colors.white,
        );
        return false;
      }
    } catch (e) {
      debugPrint('Emergency alert error: $e');
      Fluttertoast.showToast(
        msg: '⚠️ Could not reach server. Alert not sent.',
        backgroundColor: Colors.red,
        textColor: Colors.white,
        toastLength: Toast.LENGTH_LONG,
      );
      return false;
    }
  }
}