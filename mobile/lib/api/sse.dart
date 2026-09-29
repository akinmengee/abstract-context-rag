// Hand-rolled SSE-over-http client, same reasoning as web/src/api/sse.ts:
// /chat/stream is a POST with a JSON body and a bearer header, which no
// built-in SSE client (they're all GET-only, header-less) supports.

import 'dart:convert';

import 'package:http/http.dart' as http;

import 'client.dart';

class StreamEvent {
  StreamEvent({required this.event, this.token, this.answer});

  final String event; // "citations" | "token" | "done"
  final String? token;
  final Answer? answer;
}

Stream<StreamEvent> streamChat(String conversationId, String question) async* {
  final api = RagApi();
  final token = await api.storedToken;

  final request = http.Request('POST', Uri.parse('$apiUrl/api/v1/chat/stream'))
    ..headers.addAll({
      'Content-Type': 'application/json',
      if (token != null) 'Authorization': 'Bearer $token',
    })
    ..body =
        jsonEncode({'conversation_id': conversationId, 'question': question});

  final streamed = await http.Client().send(request);
  if (streamed.statusCode >= 400) {
    final body = await streamed.stream.bytesToString();
    final detail =
        body.isEmpty ? null : (jsonDecode(body)['detail'] as String?);
    throw ApiException(detail ?? 'request failed (${streamed.statusCode})');
  }

  // Buffered across chunks - a chunk boundary can land mid-line or
  // mid-frame, so only complete "\n\n"-terminated frames are parsed.
  var buffer = '';

  await for (final chunk in streamed.stream.transform(utf8.decoder)) {
    buffer += chunk;
    var boundary = buffer.indexOf('\n\n');
    while (boundary != -1) {
      final frame = buffer.substring(0, boundary);
      buffer = buffer.substring(boundary + 2);
      final parsed = _parseFrame(frame);
      if (parsed != null) yield parsed;
      boundary = buffer.indexOf('\n\n');
    }
  }
}

StreamEvent? _parseFrame(String frame) {
  var eventName = 'message';
  String? dataLine;
  for (final line in const LineSplitter().convert(frame)) {
    if (line.startsWith('event: ')) {
      eventName = line.substring(7);
    } else if (line.startsWith('data: ')) {
      dataLine = line.substring(6);
    }
  }
  if (dataLine == null) return null;
  final data = jsonDecode(dataLine) as Map<String, dynamic>;
  return StreamEvent(
    event: eventName,
    token: data['token'] as String?,
    answer: data['answer'] != null
        ? Answer.fromJson(data['answer'] as Map<String, dynamic>)
        : null,
  );
}
