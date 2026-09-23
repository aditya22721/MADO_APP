import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/contact_model.dart';
import '../utils/constants.dart';

class ContactProvider extends ChangeNotifier {
  List<EmergencyContact> _contacts = [];
  bool _loading = false;

  List<EmergencyContact> get emergencyContacts => _contacts;
  bool get isLoading => _loading;

  static const _key = 'cached_contacts';

  Future<void> loadCachedContacts() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString(_key);
      if (raw != null) {
        _contacts = (jsonDecode(raw) as List)
            .map((e) => EmergencyContact.fromJson(e))
            .toList();
        notifyListeners();
      }
    } catch (e) {
      debugPrint('loadCached: $e');
    }
  }

  Future<void> fetchContacts(String userId) async {
    _loading = true;
    notifyListeners();
    try {
      final res = await http
          .get(Uri.parse(
              '${AppConstants.apiBaseUrl}${AppConstants.contactsEndpoint}/$userId'))
          .timeout(const Duration(seconds: 10));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        _contacts = (data['contacts'] as List)
            .map((e) => EmergencyContact.fromJson(e))
            .toList();
        await _persist();
      }
    } catch (e) {
      debugPrint('fetchContacts: $e');
    }
    _loading = false;
    notifyListeners();
  }

  Future<void> _persist() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
        _key, jsonEncode(_contacts.map((c) => c.toJson()).toList()));
  }

  Future<bool> addContact({
    required String userId,
    required String name,
    required String phone,
    required String relationship,
    String? email,
    String? carrier,
    bool isPrimary = false,
  }) async {
    final newC = EmergencyContact(
      id: DateTime.now().millisecondsSinceEpoch,
      name: name,
      phone: phone,
      email: email,
      carrier: carrier,
      relationship: relationship,
      isPrimary: isPrimary,
    );
    _contacts.add(newC);
    await _persist();
    notifyListeners();

    try {
      await http.post(
        Uri.parse(
            '${AppConstants.apiBaseUrl}${AppConstants.contactsEndpoint}/add'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'user_id': userId,
          'name': name,
          'phone': phone,
          'email': email,
          'carrier': carrier,
          'relationship': relationship,
          'is_primary': isPrimary,
        }),
      ).timeout(const Duration(seconds: 10));
      await fetchContacts(userId);
    } catch (e) {
      debugPrint('addContact backend: $e');
    }
    return true;
  }

  Future<void> deleteContact(int id, String userId) async {
    try {
      await http.delete(Uri.parse(
          '${AppConstants.apiBaseUrl}${AppConstants.contactsEndpoint}/$id'));
    } catch (_) {}
    _contacts.removeWhere((c) => c.id == id);
    await _persist();
    notifyListeners();
  }

  Future<void> setPrimary(int id) async {
    _contacts =
        _contacts.map((c) => c.copyWith(isPrimary: c.id == id)).toList();
    await _persist();
    notifyListeners();
  }

  EmergencyContact? get primaryContact {
    if (_contacts.isEmpty) return null;
    return _contacts.firstWhere((c) => c.isPrimary,
        orElse: () => _contacts.first);
  }

  void clearCache() {
    _contacts.clear();
    notifyListeners();
  }
}