class EmergencyContact {
  final int id;
  final String name;
  final String phone;
  final String? email;
  final String? carrier;
  final String relationship;
  final bool isPrimary;
  final DateTime createdAt;

  EmergencyContact({
    required this.id,
    required this.name,
    required this.phone,
    this.email,
    this.carrier,
    required this.relationship,
    this.isPrimary = false,
    DateTime? createdAt,
  }) : createdAt = createdAt ?? DateTime.now();

  factory EmergencyContact.fromJson(Map<String, dynamic> json) =>
      EmergencyContact(
        id: json['id'] ?? 0,
        name: json['name'] ?? 'Unknown',
        phone: json['phone'] ?? '',
        email: json['email'],
        carrier: json['carrier'],
        relationship: json['relationship'] ?? 'Friend',
        isPrimary: json['is_primary'] == 1 || json['is_primary'] == true,
        createdAt: json['created_at'] != null
            ? DateTime.parse(json['created_at'])
            : DateTime.now(),
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'phone': phone,
        'email': email,
        'carrier': carrier,
        'relationship': relationship,
        'is_primary': isPrimary ? 1 : 0,
        'created_at': createdAt.toIso8601String(),
      };

  EmergencyContact copyWith({bool? isPrimary}) => EmergencyContact(
        id: id,
        name: name,
        phone: phone,
        email: email,
        carrier: carrier,
        relationship: relationship,
        isPrimary: isPrimary ?? this.isPrimary,
        createdAt: createdAt,
      );
}