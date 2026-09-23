import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:fluttertoast/fluttertoast.dart';
import '../providers/emergency_provider.dart';
import '../providers/contact_provider.dart';
import '../providers/location_provider.dart';
import '../providers/auth_provider.dart';

class EmergencyScreen extends StatefulWidget {
  const EmergencyScreen({super.key});

  @override
  State<EmergencyScreen> createState() => _EmergencyScreenState();
}

class _EmergencyScreenState extends State<EmergencyScreen> {
  final _msg = TextEditingController();
  bool _sending = false;

  @override
  void dispose() {
    _msg.dispose();
    super.dispose();
  }

  Future<void> _trigger() async {
    final contacts = Provider.of<ContactProvider>(context, listen: false);
    final loc = Provider.of<LocationProvider>(context, listen: false);
    final emergency = Provider.of<EmergencyProvider>(context, listen: false);
    final auth = Provider.of<AuthProvider>(context, listen: false);

    if (contacts.emergencyContacts.isEmpty) {
      Fluttertoast.showToast(
        msg: '⚠️ Add emergency contacts first',
        backgroundColor: Colors.orange,
      );
      return;
    }

    setState(() => _sending = true);

    final location = loc.currentPosition != null
        ? {
            'lat': loc.currentPosition!.latitude,
            'lng': loc.currentPosition!.longitude,
            'address': loc.currentAddress,
          }
        : {'lat': 0, 'lng': 0, 'address': 'Unknown'};

    await emergency.triggerEmergency(
      userId: auth.user?.id ?? '1',
      message: _msg.text.isEmpty
          ? 'EMERGENCY! Need immediate help!'
          : _msg.text,
      location: location,
      contacts: contacts.emergencyContacts,
    );

    if (mounted) setState(() => _sending = false);
  }

  @override
  Widget build(BuildContext context) {
    final em = Provider.of<EmergencyProvider>(context);
    final contacts = Provider.of<ContactProvider>(context);
    final loc = Provider.of<LocationProvider>(context);

    if (em.isEmergencyActive) {
      return _buildActiveScreen(em);
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Emergency'),
        backgroundColor: Colors.red,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            // -------- SOS BUTTON --------
            GestureDetector(
              onTap: _sending ? null : _trigger,
              child: Container(
                width: double.infinity,
                height: 200,
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Color(0xFFFF4444), Color(0xFFFF6B6B)],
                  ),
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.red.withOpacity(0.3),
                      blurRadius: 20,
                      spreadRadius: 5,
                    ),
                  ],
                ),
                child: _sending
                    ? const Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            CircularProgressIndicator(color: Colors.white),
                            SizedBox(height: 16),
                            Text(
                              'Sending alerts...',
                              style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold),
                            ),
                          ],
                        ),
                      )
                    : const Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.sos, size: 80, color: Colors.white),
                          SizedBox(height: 8),
                          Text(
                            'TAP FOR EMERGENCY',
                            style: TextStyle(
                                fontSize: 22,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                                letterSpacing: 3),
                          ),
                          SizedBox(height: 4),
                          Text(
                            'SMS + Email will be sent automatically',
                            style: TextStyle(color: Colors.white70),
                          ),
                        ],
                      ),
              ),
            ),
            const SizedBox(height: 24),

            // -------- MESSAGE --------
            TextField(
              controller: _msg,
              maxLines: 3,
              decoration: InputDecoration(
                labelText: 'Describe emergency (optional)',
                hintText: 'e.g., I am having chest pain',
                border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12)),
              ),
            ),
            const SizedBox(height: 24),

            // -------- CONTACTS --------
            Container(
              padding: const EdgeInsets.all(16),
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.grey[100],
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    '📞 Emergency Contacts',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  const SizedBox(height: 8),
                  if (contacts.emergencyContacts.isEmpty)
                    const Text('None added yet',
                        style: TextStyle(color: Colors.grey))
                  else
                    ...contacts.emergencyContacts.map((c) => ListTile(
                          dense: true,
                          leading: CircleAvatar(
                            backgroundColor:
                                c.isPrimary ? Colors.green : Colors.grey,
                            child: Text(
                              c.name.isNotEmpty
                                  ? c.name[0].toUpperCase()
                                  : '?',
                              style: const TextStyle(color: Colors.white),
                            ),
                          ),
                          title: Text(c.name),
                          subtitle: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(c.phone),
                              if (c.email != null && c.email!.isNotEmpty)
                                Text(
                                  '📧 ${c.email}',
                                  style: const TextStyle(fontSize: 12),
                                ),
                            ],
                          ),
                        )),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // -------- LOCATION --------
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.blue[50],
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                children: [
                  const Icon(Icons.location_on, color: Colors.blue),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(loc.currentAddress.isEmpty
                        ? 'Location not available'
                        : loc.currentAddress),
                  ),
                  IconButton(
                    icon: const Icon(Icons.refresh),
                    onPressed: () => loc.getCurrentLocation(),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActiveScreen(EmergencyProvider em) {
    return Scaffold(
      backgroundColor: Colors.red.shade900,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            children: [
              const Icon(Icons.warning_amber, size: 100, color: Colors.white),
              const SizedBox(height: 16),
              const Text(
                '🚨 EMERGENCY MODE',
                style: TextStyle(
                    fontSize: 26,
                    fontWeight: FontWeight.bold,
                    color: Colors.white),
              ),
              const SizedBox(height: 8),
              Text(
                'Type: ${em.emergencyType}',
                style: const TextStyle(fontSize: 16, color: Colors.white70),
              ),
              const SizedBox(height: 24),
              if (em.waitingForResponse)
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white24,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Column(
                    children: [
                      const Text(
                        '✅ Alerts sent to all contacts',
                        style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 16),
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        'SMS + Email have been sent automatically',
                        style: TextStyle(color: Colors.white70, fontSize: 13),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 16),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          ElevatedButton(
                            onPressed: () => em.userResponded('help'),
                            style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.green),
                            child: const Text('I NEED HELP!'),
                          ),
                          const SizedBox(width: 12),
                          ElevatedButton(
                            onPressed: () => em.userResponded('ok'),
                            style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.white,
                                foregroundColor: Colors.black),
                            child: const Text("I'm OK"),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      TextButton.icon(
                        onPressed: () => em.resendAlerts(),
                        icon: const Icon(Icons.refresh, color: Colors.white),
                        label: const Text('Re-send alerts',
                            style: TextStyle(color: Colors.white)),
                      ),
                    ],
                  ),
                ),
              const Spacer(),
              ElevatedButton(
                onPressed: em.cancelEmergency,
                style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.white,
                    foregroundColor: Colors.red),
                child: const Text('Cancel Emergency'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}