// Thin client for the backend API - same REST/JWT contract web/src/api/client.ts
// uses, so a conversation started on web continues here unchanged.
// The host defaults to the Android emulator's alias for the development
// machine; on a real phone pass the computer's LAN IP with
// --dart-define=API_URL=http://<lan-ip>:8000 (same network, see mobile/README.md).

import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

const String apiUrl = String.fromEnvironment(
  'API_URL',
  defaultValue: 'http://10.0.2.2:8000',
);

class ApiException implements Exception {
  ApiException(this.message);
  final String message;
  @override
  String toString() => message;
}

class Citation {
  Citation({
    required this.marker,
    required this.chunkId,
    required this.title,
    this.section,
    this.page,
  });

  final int marker;
  final String chunkId;
  final String title;
  final String? section;
  final int? page;

  factory Citation.fromJson(Map<String, dynamic> json) => Citation(
        marker: json['marker'] as int,
        chunkId: json['chunk_id'] as String,
        title: json['title'] as String,
        section: json['section'] as String?,
        page: json['page'] as int?,
      );
}

class Answer {
  Answer(
      {required this.text, required this.citations, required this.abstained});

  final String text;
  final List<Citation> citations;
  final bool abstained;

  factory Answer.fromJson(Map<String, dynamic> json) => Answer(
        text: json['text'] as String,
        citations: (json['citations'] as List<dynamic>? ?? [])
            .map((citation) =>
                Citation.fromJson(citation as Map<String, dynamic>))
            .toList(),
        abstained: json['abstained'] as bool? ?? false,
      );
}

class DocumentSummary {
  DocumentSummary({required this.documentId, required this.title});
  final String documentId;
  final String title;

  factory DocumentSummary.fromJson(Map<String, dynamic> json) =>
      DocumentSummary(
        documentId: json['document_id'] as String,
        title: json['title'] as String,
      );
}

class ConversationSummary {
  ConversationSummary({
    required this.id,
    required this.title,
    required this.documentId,
    required this.updatedAt,
  });

  final String id;
  final String title;
  final String? documentId;
  final DateTime updatedAt;

  factory ConversationSummary.fromJson(Map<String, dynamic> json) =>
      ConversationSummary(
        id: json['id'] as String,
        title: json['title'] as String,
        documentId: json['document_id'] as String?,
        updatedAt: DateTime.parse(json['updated_at'] as String),
      );
}

class MessageOut {
  MessageOut(
      {required this.role, required this.content, required this.citations});

  final String role;
  final String content;
  final List<Citation>? citations;

  factory MessageOut.fromJson(Map<String, dynamic> json) => MessageOut(
        role: json['role'] as String,
        content: json['content'] as String,
        citations: (json['citations'] as List<dynamic>?)
            ?.map((citation) =>
                Citation.fromJson(citation as Map<String, dynamic>))
            .toList(),
      );
}

class ConversationDetail extends ConversationSummary {
  ConversationDetail({
    required super.id,
    required super.title,
    required super.documentId,
    required super.updatedAt,
    required this.messages,
  });

  final List<MessageOut> messages;

  factory ConversationDetail.fromJson(Map<String, dynamic> json) =>
      ConversationDetail(
        id: json['id'] as String,
        title: json['title'] as String,
        documentId: json['document_id'] as String?,
        updatedAt: DateTime.parse(json['updated_at'] as String),
        messages: (json['messages'] as List<dynamic>)
            .map((message) =>
                MessageOut.fromJson(message as Map<String, dynamic>))
            .toList(),
      );
}

class TokenResponse {
  TokenResponse({required this.accessToken, required this.email});
  final String accessToken;
  final String email;

  factory TokenResponse.fromJson(Map<String, dynamic> json) => TokenResponse(
        accessToken: json['access_token'] as String,
        email: json['email'] as String,
      );
}

/// Reads/writes the bearer token via SharedPreferences (this app's
/// equivalent of the web client's localStorage) and attaches it to every
/// request below.
class RagApi {
  static const _tokenKey = 'acr_token';

  Future<String?> get storedToken async =>
      (await SharedPreferences.getInstance()).getString(_tokenKey);

  Future<void> _saveToken(String token) async {
    await (await SharedPreferences.getInstance()).setString(_tokenKey, token);
  }

  Future<void> clearToken() async {
    await (await SharedPreferences.getInstance()).remove(_tokenKey);
  }

  Future<Map<String, String>> _headers() async {
    final token = await storedToken;
    return {
      'Content-Type': 'application/json',
      if (token != null) 'Authorization': 'Bearer $token',
    };
  }

  dynamic _unwrap(http.Response response) {
    if (response.statusCode >= 400) {
      final body = jsonDecode(response.body);
      throw ApiException(body['detail']?.toString() ??
          'request failed (${response.statusCode})');
    }
    if (response.body.isEmpty) return null;
    return jsonDecode(response.body);
  }

  Future<TokenResponse> register(String email, String password) async {
    final response = await http.post(
      Uri.parse('$apiUrl/api/v1/auth/register'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'email': email, 'password': password}),
    );
    final token =
        TokenResponse.fromJson(_unwrap(response) as Map<String, dynamic>);
    await _saveToken(token.accessToken);
    return token;
  }

  Future<TokenResponse> login(String email, String password) async {
    final response = await http.post(
      Uri.parse('$apiUrl/api/v1/auth/login'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'email': email, 'password': password}),
    );
    final token =
        TokenResponse.fromJson(_unwrap(response) as Map<String, dynamic>);
    await _saveToken(token.accessToken);
    return token;
  }

  Future<List<DocumentSummary>> listDocuments() async {
    final response = await http.get(Uri.parse('$apiUrl/api/v1/documents'),
        headers: await _headers());
    return (_unwrap(response) as List<dynamic>)
        .map((doc) => DocumentSummary.fromJson(doc as Map<String, dynamic>))
        .toList();
  }

  Future<DocumentSummary> ingestArxiv(String arxivId) async {
    final response = await http.post(
      Uri.parse('$apiUrl/api/v1/ingest'),
      headers: await _headers(),
      body: jsonEncode({'arxiv_id': arxivId}),
    );
    return DocumentSummary.fromJson(_unwrap(response) as Map<String, dynamic>);
  }

  Future<DocumentSummary> ingestWikipedia(String article) async {
    final response = await http.post(
      Uri.parse('$apiUrl/api/v1/ingest'),
      headers: await _headers(),
      body: jsonEncode({'wikipedia': article}),
    );
    return DocumentSummary.fromJson(_unwrap(response) as Map<String, dynamic>);
  }

  Future<List<ConversationSummary>> listConversations() async {
    final response = await http.get(Uri.parse('$apiUrl/api/v1/conversations'),
        headers: await _headers());
    return (_unwrap(response) as List<dynamic>)
        .map((c) => ConversationSummary.fromJson(c as Map<String, dynamic>))
        .toList();
  }

  Future<ConversationSummary> createConversation(
      {String? documentId, String? title}) async {
    final response = await http.post(
      Uri.parse('$apiUrl/api/v1/conversations'),
      headers: await _headers(),
      body: jsonEncode({'document_id': documentId, 'title': title}),
    );
    return ConversationSummary.fromJson(
        _unwrap(response) as Map<String, dynamic>);
  }

  Future<ConversationDetail> getConversation(String id) async {
    final response = await http.get(
        Uri.parse('$apiUrl/api/v1/conversations/$id'),
        headers: await _headers());
    return ConversationDetail.fromJson(
        _unwrap(response) as Map<String, dynamic>);
  }

  Future<void> renameConversation(String id, String title) async {
    final response = await http.patch(
      Uri.parse('$apiUrl/api/v1/conversations/$id'),
      headers: await _headers(),
      body: jsonEncode({'title': title}),
    );
    _unwrap(response);
  }

  Future<void> deleteConversation(String id) async {
    final response = await http.delete(
        Uri.parse('$apiUrl/api/v1/conversations/$id'),
        headers: await _headers());
    _unwrap(response);
  }
}
