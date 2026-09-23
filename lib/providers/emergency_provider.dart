import 'package:flutter/material.dart';
import 'dart:async';
import 'package:fluttertoast/fluttertoast.dart';
import '../models/contact_model.dart';
import '../services/emergency_service.dart';

class EmergencyProvider extends ChangeNotifier {
  bool _active = false;
  String _type = '';
  String _message = '';
  Timer? _timer;
  bool _waiting = false;
  List<EmergencyContact> _contacts = [];
  String _userId = '1';

  bool get isEmergencyActive => _active;
  String get emergencyType => _type;
  bool get waitingForResponse => _waiting;

  final EmergencyService _service = EmergencyService();

  Future<void> triggerEmergency({
    required String userId,
    required String message,
    required Map<String, dynamic> location,
    List<EmergencyContact> contacts = const [],
  }) async {
    _active = true;
    _type = _detectType(message);
    _message = message;
    _contacts = contacts;
    _userId = userId;
    _waiting = true;
    notifyListeners();

    Fluttertoast.showToast(
      msg: '🚨 Emergency activated! Sending alerts...',
      backgroundColor: Colors.red,
      textColor: Colors.white,
      toastLength: Toast.LENGTH_LONG,
    );

    // Send alerts via backend (SMS + Email + WhatsApp)
    await _service.alertContacts(
      userId: userId,
      contacts: contacts,
      message: message,
      address: location['address'] ?? 'Unknown location',
      lat: (location['lat'] ?? 0).toDouble(),
      lng: (location['lng'] ?? 0).toDouble(),
    );

    // 30-second cooldown
    _timer?.cancel();
    _timer = Timer(const Duration(seconds: 30), () {
      _waiting = false;
      _active = false;
      notifyListeners();
    });
  }

  /// Re-send alerts manually (in case first attempt failed)
  Future<void> resendAlerts() async {
    if (_contacts.isEmpty) return;

    Fluttertoast.showToast(
      msg: '🔄 Re-sending alerts...',
      backgroundColor: Colors.orange,
      textColor: Colors.white,
    );

    await _service.alertContacts(
      userId: _userId,
      contacts: _contacts,
      message: _message,
      address: 'Location attached in original alert',
      lat: 0,
      lng: 0,
    );
  }

  void userResponded(String response) {
    _timer?.cancel();
    _waiting = false;
    if (response == 'ok') {
      _active = false;
      Fluttertoast.showToast(
        msg: '✅ Emergency cancelled',
        backgroundColor: Colors.green,
      );
    }
    notifyListeners();
  }

  void cancelEmergency() {
    _timer?.cancel();
    _active = false;
    _waiting = false;
    notifyListeners();
  }

  String _detectType(String m) {
    final l = m.toLowerCase();
    if (l.contains('heart') || l.contains('chest')) return 'Heart Attack';
    if (l.contains('bleed')) return 'Bleeding';
    if (l.contains('accident') || l.contains('crash')) return 'Accident';
    if (l.contains('fire')) return 'Fire';
    if (l.contains('attack')) return 'Attack';
    if (l.contains('stroke')) return 'Stroke';
    return 'General Emergency';
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }
}