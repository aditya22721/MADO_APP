import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:flutter/foundation.dart';

class LocationProvider extends ChangeNotifier {
  Position? _position;
  String _address = '';
  bool _loading = false;

  Position? get currentPosition => _position;
  String get currentAddress => _address;
  bool get isLoading => _loading;

  void _setMock() {
    _position = Position(
      latitude: 28.6139,
      longitude: 77.2090,
      timestamp: DateTime.now(),
      accuracy: 100,
      altitude: 0,
      altitudeAccuracy: 0,
      heading: 0,
      headingAccuracy: 0,
      speed: 0,
      speedAccuracy: 0,
    );
    _address = 'New Delhi, India';
  }

  Future<Map<String, dynamic>?> getCurrentLocation() async {
    _loading = true;
    notifyListeners();

    try {
      if (kIsWeb) {
        _setMock();
        _loading = false;
        notifyListeners();
        return _payload();
      }

      final perm = await Geolocator.checkPermission();
      if (perm == LocationPermission.denied) {
        await Geolocator.requestPermission();
      }

      _position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );
      _address =
          '${_position!.latitude.toStringAsFixed(4)}, ${_position!.longitude.toStringAsFixed(4)}';
    } catch (e) {
      debugPrint('Location error: $e');
      _setMock();
    }

    _loading = false;
    notifyListeners();
    return _payload();
  }

  Map<String, dynamic> _payload() => {
        'lat': _position?.latitude ?? 0,
        'lng': _position?.longitude ?? 0,
        'address': _address,
      };
}