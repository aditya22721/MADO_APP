import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';
import '../models/user_model.dart';

class AuthProvider extends ChangeNotifier {
  User? _user;
  bool _loading = false;
  bool _auth = false;
  String _error = '';

  User? get user => _user;
  bool get isLoading => _loading;
  bool get isAuthenticated => _auth;
  String get error => _error;

  AuthProvider() {
    _loadSavedUser();
  }

  Future<void> _loadSavedUser() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString('user');
      if (raw != null) {
        _user = User.fromJson(jsonDecode(raw));
        _auth = true;
        notifyListeners();
      }
    } catch (e) {
      debugPrint('loadSavedUser: $e');
    }
  }

  Future<bool> signUp({
    required String name,
    required String email,
    required String phone,
    required String password,
  }) async {
    _loading = true;
    _error = '';
    notifyListeners();

    try {
      await Future.delayed(const Duration(milliseconds: 600));
      _user = User(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        name: name,
        email: email,
        phone: phone,
      );
      _auth = true;
      await _save();
      _loading = false;
      notifyListeners();
      return true;
    } catch (e) {
      _error = e.toString();
      _loading = false;
      notifyListeners();
      return false;
    }
  }

  Future<bool> login({
    required String email,
    required String password,
  }) async {
    _loading = true;
    _error = '';
    notifyListeners();

    try {
      await Future.delayed(const Duration(milliseconds: 600));
      _user = User(
        id: '1',
        name: 'MADO User',
        email: email,
        phone: '+919999999999',
      );
      _auth = true;
      await _save();
      _loading = false;
      notifyListeners();
      return true;
    } catch (e) {
      _error = e.toString();
      _loading = false;
      notifyListeners();
      return false;
    }
  }

  Future<void> logout() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('user');
    _user = null;
    _auth = false;
    notifyListeners();
  }

  Future<void> _save() async {
    if (_user == null) return;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('user', jsonEncode(_user!.toJson()));
  }
}