// Thin client for the backend API.
// The host defaults to the Android emulator's alias for the development machine;
// on a real phone pass the computer's LAN IP with --dart-define=API_URL=...

import 'dart:convert';

import 'package:http/http.dart' as http;

const String apiUrl = String.fromEnvironment(
  'API_URL',
  defaultValue: 'http://10.0.2.2:8000',
);

class Citation {
  Citation({required this.marker, required this.title, this.section, this.page});

  final int marker;
  final String title;
  final String? section;
  final int? page;

  factory Citation.fromJson(Map<String, dynamic> json) => Citation(
        marker: json['marker'] as int,
        title: json['title'] as String,
        section: json['section'] as String?,
        page: json['page'] as int?,
      );
}

class Answer {
  Answer({required this.text, required this.citations, required this.abstained});

  final String text;
  final List<Citation> citations;
  final bool abstained;

  factory Answer.fromJson(Map<String, dynamic> json) => Answer(
        text: json['text'] as String,
        citations: (json['citations'] as List<dynamic>)
            .map((citation) => Citation.fromJson(citation as Map<String, dynamic>))
            .toList(),
        abstained: json['abstained'] as bool,
      );
}

class RagApi {
  Future<Answer> ask(String question, {String? documentId}) async {
    final response = await http.post(
      Uri.parse('$apiUrl/api/v1/chat'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'question': question, 'document_id': documentId}),
    );
    if (response.statusCode >= 400) {
      throw Exception(jsonDecode(response.body)['detail'] ?? 'request failed');
    }
    return Answer.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  Future<String> ingestArxiv(String arxivId) async {
    final response = await http.post(
      Uri.parse('$apiUrl/api/v1/ingest'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'arxiv_id': arxivId}),
    );
    if (response.statusCode >= 400) {
      throw Exception(jsonDecode(response.body)['detail'] ?? 'request failed');
    }
    final result = jsonDecode(response.body) as Map<String, dynamic>;
    return '${result['title']} — ${result['chunk_count']} chunks';
  }
}
