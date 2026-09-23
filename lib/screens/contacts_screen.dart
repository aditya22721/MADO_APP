import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:fluttertoast/fluttertoast.dart';
import '../providers/contact_provider.dart';
import '../providers/auth_provider.dart';

class ContactsScreen extends StatelessWidget {
  const ContactsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final contacts = Provider.of<ContactProvider>(context);
    final auth = Provider.of<AuthProvider>(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Emergency Contacts'),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: () => _addDialog(context, auth.user?.id ?? '1'),
          ),
        ],
      ),
      body: contacts.isLoading
          ? const Center(child: CircularProgressIndicator())
          : contacts.emergencyContacts.isEmpty
              ? const Center(
                  child: Padding(
                    padding: EdgeInsets.all(24),
                    child: Text(
                      'No emergency contacts yet.\nTap + to add.',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 16),
                    ),
                  ),
                )
              : ListView.builder(
                  padding: const EdgeInsets.all(12),
                  itemCount: contacts.emergencyContacts.length,
                  itemBuilder: (_, i) {
                    final c = contacts.emergencyContacts[i];
                    return Card(
                      margin: const EdgeInsets.only(bottom: 8),
                      child: ListTile(
                        leading: CircleAvatar(
                          backgroundColor:
                              c.isPrimary ? Colors.green : Colors.grey,
                          child: Text(
                            c.name.isNotEmpty ? c.name[0].toUpperCase() : '?',
                            style: const TextStyle(color: Colors.white),
                          ),
                        ),
                        title: Text(c.name),
                        subtitle: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(c.phone),
                            if (c.email != null && c.email!.isNotEmpty)
                              Text('📧 ${c.email}',
                                  style: const TextStyle(fontSize: 12)),
                            Text(
                              '📡 ${c.carrier ?? "Auto"} • ${c.relationship}',
                              style: const TextStyle(
                                  fontSize: 12, color: Colors.grey),
                            ),
                          ],
                        ),
                        trailing: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            if (!c.isPrimary)
                              IconButton(
                                icon: const Icon(Icons.star_border,
                                    color: Colors.amber),
                                onPressed: () => contacts.setPrimary(c.id),
                              ),
                            IconButton(
                              icon: const Icon(Icons.delete,
                                  color: Colors.red),
                              onPressed: () {
                                contacts.deleteContact(
                                    c.id, auth.user?.id ?? '1');
                              },
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _addDialog(context, auth.user?.id ?? '1'),
        icon: const Icon(Icons.add),
        label: const Text('Add'),
      ),
    );
  }

  void _addDialog(BuildContext context, String userId) {
    final name = TextEditingController();
    final phone = TextEditingController();
    final email = TextEditingController();
    final rel = TextEditingController();

    bool primary = false;
    String? carrier = 'Airtel';

    final carriers = ['Airtel', 'Jio', 'Vi', 'BSNL', 'Auto-detect'];

    showDialog(
      context: context,
      builder: (_) => StatefulBuilder(
        builder: (ctx, setSt) => AlertDialog(
          title: const Text('Add Emergency Contact'),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: name,
                  decoration: const InputDecoration(labelText: 'Name'),
                ),
                TextField(
                  controller: phone,
                  keyboardType: TextInputType.phone,
                  decoration: const InputDecoration(
                      labelText: 'Phone (10 digits)'),
                ),
                TextField(
                  controller: email,
                  keyboardType: TextInputType.emailAddress,
                  decoration: const InputDecoration(
                      labelText: 'Email (for alerts)'),
                ),
                const SizedBox(height: 12),
                DropdownButtonFormField<String>(
                  value: carrier,
                  decoration: const InputDecoration(
                    labelText: 'Mobile Carrier',
                    helperText: 'Needed to send SMS alerts',
                  ),
                  items: carriers
                      .map((c) => DropdownMenuItem(value: c, child: Text(c)))
                      .toList(),
                  onChanged: (v) => setSt(() => carrier = v),
                ),
                TextField(
                  controller: rel,
                  decoration:
                      const InputDecoration(labelText: 'Relationship'),
                ),
                Row(
                  children: [
                    Checkbox(
                        value: primary,
                        onChanged: (v) =>
                            setSt(() => primary = v ?? false)),
                    const Text('Primary contact'),
                  ],
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () async {
                if (name.text.isEmpty || phone.text.isEmpty) {
                  Fluttertoast.showToast(msg: 'Name & phone required');
                  return;
                }
                final finalCarrier =
                    (carrier == 'Auto-detect') ? null : carrier;

                await Provider.of<ContactProvider>(ctx, listen: false)
                    .addContact(
                  userId: userId,
                  name: name.text,
                  phone: phone.text,
                  email: email.text.isNotEmpty ? email.text : null,
                  carrier: finalCarrier,
                  relationship:
                      rel.text.isNotEmpty ? rel.text : 'Friend',
                  isPrimary: primary,
                );
                if (ctx.mounted) Navigator.pop(ctx);
                Fluttertoast.showToast(
                  msg: '✅ Contact added',
                  backgroundColor: Colors.green,
                );
              },
              child: const Text('Add'),
            ),
          ],
        ),
      ),
    );
  }
}