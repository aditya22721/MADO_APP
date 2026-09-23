import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/chat_provider.dart';
import '../providers/auth_provider.dart';
import '../providers/location_provider.dart';
import '../providers/emergency_provider.dart';
import '../providers/contact_provider.dart';
import '../widgets/message_bubble.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _controller = TextEditingController();
  final _scroll = ScrollController();

  @override
  void dispose() {
    _controller.dispose();
    _scroll.dispose();
    super.dispose();
  }

  void _scrollDown() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.animateTo(_scroll.position.maxScrollExtent,
            duration: const Duration(milliseconds: 300), curve: Curves.easeOut);
      }
    });
  }

  Future<void> _send() async {
    final text = _controller.text.trim();
    if (text.isEmpty) return;
    _controller.clear();

    final auth = Provider.of<AuthProvider>(context, listen: false);
    final chat = Provider.of<ChatProvider>(context, listen: false);
    final loc = Provider.of<LocationProvider>(context, listen: false);
    final emergency = Provider.of<EmergencyProvider>(context, listen: false);
    final contacts = Provider.of<ContactProvider>(context, listen: false);

    Map<String, dynamic>? location;
    if (loc.currentPosition != null) {
      location = {
        'lat': loc.currentPosition!.latitude,
        'lng': loc.currentPosition!.longitude,
        'address': loc.currentAddress,
      };
    }

    await chat.sendMessage(
      userId: auth.user?.id ?? '1',
      message: text,
      location: location,
    );
    _scrollDown();

    if (chat.isEmergencyMode) {
      await emergency.triggerEmergency(
        userId: auth.user?.id ?? '1',
        message: text,
        location: location ?? {'lat': 0, 'lng': 0, 'address': 'Unknown'},
        contacts: contacts.emergencyContacts,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final chat = Provider.of<ChatProvider>(context);
    return Scaffold(
      appBar: AppBar(
        title: Text('MADO Assistant',
            style: GoogleFonts.poppins(fontWeight: FontWeight.w600)),
        backgroundColor:
            chat.isEmergencyMode ? Colors.red : Theme.of(context).colorScheme.primary,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.clear_all),
            onPressed: chat.clearMessages,
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: chat.messages.isEmpty
                ? _empty()
                : ListView.builder(
                    controller: _scroll,
                    padding: const EdgeInsets.all(16),
                    itemCount: chat.messages.length,
                    itemBuilder: (_, i) {
                      final m = chat.messages[i];
                      return MessageBubble(
                        message: m.message,
                        response: m.response,
                        isEmergency: m.isEmergency,
                        isUser: m.isUser,
                        timestamp: m.timestamp,
                      );
                    },
                  ),
          ),
          if (chat.isLoading)
            const Padding(
              padding: EdgeInsets.all(8),
              child: LinearProgressIndicator(),
            ),
          Container(
            padding: const EdgeInsets.all(8),
            color: Theme.of(context).scaffoldBackgroundColor,
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: InputDecoration(
                      hintText: 'Ask me anything...',
                      border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(24),
                          borderSide: BorderSide.none),
                      filled: true,
                      fillColor: Colors.grey[100],
                      contentPadding:
                          const EdgeInsets.symmetric(horizontal: 16),
                    ),
                    onSubmitted: (_) => _send(),
                  ),
                ),
                const SizedBox(width: 8),
                CircleAvatar(
                  backgroundColor: Theme.of(context).colorScheme.primary,
                  child: IconButton(
                    icon: const Icon(Icons.send, color: Colors.white),
                    onPressed: _send,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _empty() => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.chat_bubble_outline, size: 64, color: Colors.grey[400]),
            const SizedBox(height: 12),
            Text("Hello! I'm MADO",
                style: GoogleFonts.poppins(
                    fontSize: 20, fontWeight: FontWeight.w600)),
            const SizedBox(height: 8),
            Text('Try asking me something',
                style: GoogleFonts.poppins(color: Colors.grey[600])),
            const SizedBox(height: 24),
            Wrap(
              spacing: 8,
              children: [
                _chip('Tell me a joke'),
                _chip('What is the weather?'),
                _chip('Tell me about moon'),
                _chip('What is quantum computing?'),
              ],
            ),
          ],
        ),
      );

  Widget _chip(String s) => ActionChip(
        label: Text(s),
        onPressed: () {
          _controller.text = s;
          _send();
        },
      );
}