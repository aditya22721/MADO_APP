import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../models/chat_model.dart';
import '../utils/constants.dart';

class ChatProvider extends ChangeNotifier {
  final List<ChatMessage> _messages = [];
  bool _isLoading = false;
  bool _isEmergencyMode = false;

  List<ChatMessage> get messages => _messages;
  bool get isLoading => _isLoading;
  bool get isEmergencyMode => _isEmergencyMode;

  Future<void> sendMessage({
    required String userId,
    required String message,
    Map<String, dynamic>? location,
  }) async {
    _isLoading = true;
    notifyListeners();

    // Add user bubble
    _messages.add(ChatMessage(
      message: message,
      response: '',
      isUser: true,
    ));
    notifyListeners();

    try {
      final res = await http.post(
        Uri.parse('${AppConstants.apiBaseUrl}${AppConstants.chatEndpoint}'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'user_id': userId,
          'message': message,
          'location': location,
        }),
      ).timeout(const Duration(seconds: 30));

      _isLoading = false;

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        final emergency = data['is_emergency'] ?? false;
        if (emergency) _isEmergencyMode = true;

        _messages.add(ChatMessage(
          message: message,
          response: data['response'] ?? 'No response',
          isEmergency: emergency,
          isUser: false,
        ));
      } else {
        _messages.add(ChatMessage(
          message: message,
          response: '⚠️ Server returned ${res.statusCode}',
          isUser: false,
        ));
      }
    } catch (e) {
      _isLoading = false;
      debugPrint('sendMessage error: $e');
      _messages.add(ChatMessage(
        message: message,
        response: '⚠️ Cannot reach server.\nCheck backend is running.',
        isUser: false,
      ));
    }

    notifyListeners();
  }

  void clearMessages() {
    _messages.clear();
    _isEmergencyMode = false;
    notifyListeners();
  }

  void resetEmergencyMode() {
    _isEmergencyMode = false;
    notifyListeners();
  }
}