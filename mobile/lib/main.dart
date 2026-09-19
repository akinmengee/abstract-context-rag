// Skeleton: one screen that ingests a paper and answers a question against it,
// enough to prove the API works from a phone. The real app is designed later
// (roadmap phase 9).

import 'package:flutter/material.dart';

import 'api/client.dart';

void main() => runApp(const RagApp());

class RagApp extends StatelessWidget {
  const RagApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'abstract-context-rag',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: const AskPage(),
    );
  }
}

class AskPage extends StatefulWidget {
  const AskPage({super.key});

  @override
  State<AskPage> createState() => _AskPageState();
}

class _AskPageState extends State<AskPage> {
  final _api = RagApi();
  final _arxivController = TextEditingController(text: '2005.11401');
  final _questionController = TextEditingController();

  String _status = '';
  Answer? _answer;

  Future<void> _run(Future<void> Function() action) async {
    setState(() => _status = 'working…');
    try {
      await action();
      setState(() => _status = '');
    } catch (error) {
      setState(() => _status = error.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('abstract-context-rag')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: ListView(
          children: [
            TextField(
              controller: _arxivController,
              decoration: const InputDecoration(labelText: 'arXiv ID'),
            ),
            FilledButton(
              onPressed: () => _run(() async {
                final result = await _api.ingestArxiv(_arxivController.text);
                setState(() => _status = result);
              }),
              child: const Text('Add paper'),
            ),
            const SizedBox(height: 24),
            TextField(
              controller: _questionController,
              decoration: const InputDecoration(labelText: 'Question'),
            ),
            FilledButton(
              onPressed: () => _run(() async {
                final answer = await _api.ask(_questionController.text);
                setState(() => _answer = answer);
              }),
              child: const Text('Ask'),
            ),
            if (_status.isNotEmpty) Padding(
              padding: const EdgeInsets.symmetric(vertical: 12),
              child: Text(_status),
            ),
            if (_answer != null) ...[
              const SizedBox(height: 16),
              Text(_answer!.text),
              const SizedBox(height: 8),
              for (final citation in _answer!.citations)
                Text(
                  '[${citation.marker}] ${citation.title} ${citation.section ?? ''}',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
            ],
          ],
        ),
      ),
    );
  }
}
