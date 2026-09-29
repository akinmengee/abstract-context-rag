import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'api/client.dart';

/// Plain ChangeNotifier + provider - this app's equivalent of the web
/// client's React Context, same "don't over-build this" preference.
class AuthState extends ChangeNotifier {
  AuthState(this._api);

  final RagApi _api;
  String? token;
  String? email;
  bool loading = true;

  Future<void> restore() async {
    token = await _api.storedToken;
    final prefs = await SharedPreferences.getInstance();
    email = prefs.getString('acr_email');
    loading = false;
    notifyListeners();
  }

  Future<void> login(String emailInput, String password) async {
    final result = await _api.login(emailInput, password);
    await _persist(result);
  }

  Future<void> register(String emailInput, String password) async {
    final result = await _api.register(emailInput, password);
    await _persist(result);
  }

  Future<void> _persist(TokenResponse result) async {
    token = result.accessToken;
    email = result.email;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('acr_email', result.email);
    notifyListeners();
  }

  Future<void> logout() async {
    await _api.clearToken();
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('acr_email');
    token = null;
    email = null;
    notifyListeners();
  }
}
