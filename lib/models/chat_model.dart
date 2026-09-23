class ChatMessage {
  final String message;
  final String response;
  final bool isEmergency;
  final bool isUser;
  final DateTime timestamp;

  ChatMessage({
    required this.message,
    required this.response,
    this.isEmergency = false,
    this.isUser = true,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  factory ChatMessage.fromJson(Map<String, dynamic> json) => ChatMessage(
        message: json['message'] ?? '',
        response: json['response'] ?? '',
        isEmergency: json['is_emergency'] ?? false,
        isUser: false,
        timestamp: json['timestamp'] != null
            ? DateTime.parse(json['timestamp'])
            : DateTime.now(),
      );

  Map<String, dynamic> toJson() => {
        'message': message,
        'response': response,
        'is_emergency': isEmergency,
        'timestamp': timestamp.toIso8601String(),
      };
}