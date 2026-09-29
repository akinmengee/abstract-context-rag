import 'package:flutter/material.dart';

import '../api/client.dart';
import '../theme.dart';

/// Shows the New Chat picker and returns the created conversation, or null
/// if the user cancelled. Mirrors web's NewChatDialog: pick an existing
/// document, ingest a new one, or "all documents" - fixed for the whole
/// conversation's lifetime once chosen.
Future<ConversationSummary?> showNewChatDialog(
    BuildContext context, RagApi api) {
  return showModalBottomSheet<ConversationSummary>(
    context: context,
    isScrollControlled: true,
    backgroundColor: Theme.of(context).colorScheme.surface,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
    ),
    builder: (context) => _NewChatSheet(api: api),
  );
}

class _NewChatSheet extends StatefulWidget {
  const _NewChatSheet({required this.api});
  final RagApi api;

  @override
  State<_NewChatSheet> createState() => _NewChatSheetState();
}

class _NewChatSheetState extends State<_NewChatSheet> {
  List<DocumentSummary>? _documents;
  final _arxivController = TextEditingController();
  final _wikiController = TextEditingController();
  bool _busy = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    widget.api.listDocuments().then((docs) {
      if (mounted) setState(() => _documents = docs);
    }).catchError((_) {
      if (mounted) setState(() => _documents = []);
    });
  }

  Future<void> _createFor(Future<DocumentSummary> Function() ingest) async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final doc = await ingest();
      final conversation = await widget.api
          .createConversation(documentId: doc.documentId, title: doc.title);
      if (mounted) Navigator.of(context).pop(conversation);
    } catch (error) {
      setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _pickExisting(DocumentSummary doc) =>
      _createFor(() async => doc);

  Future<void> _allDocuments() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final conversation = await widget.api
          .createConversation(documentId: null, title: 'All documents');
      if (mounted) Navigator.of(context).pop(conversation);
    } catch (error) {
      setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final palette = context.palette;
    return Padding(
      padding: EdgeInsets.fromLTRB(
          20, 16, 20, 16 + MediaQuery.of(context).viewInsets.bottom),
      child: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('New Chat', style: theme.textTheme.titleLarge),
            const SizedBox(height: 16),
            Text('EXISTING DOCUMENT',
                style: TextStyle(
                    fontSize: 11, letterSpacing: 1, color: palette.textMuted)),
            const SizedBox(height: 6),
            if (_documents == null)
              const Padding(
                  padding: EdgeInsets.symmetric(vertical: 8),
                  child: LinearProgressIndicator())
            else if (_documents!.isEmpty)
              Text('No documents ingested yet.',
                  style: TextStyle(color: palette.textMuted))
            else
              ConstrainedBox(
                constraints: const BoxConstraints(maxHeight: 160),
                child: ListView.builder(
                  shrinkWrap: true,
                  itemCount: _documents!.length,
                  itemBuilder: (context, index) {
                    final doc = _documents![index];
                    return ListTile(
                      dense: true,
                      title: Text(doc.title,
                          maxLines: 1, overflow: TextOverflow.ellipsis),
                      onTap: _busy ? null : () => _pickExisting(doc),
                    );
                  },
                ),
              ),
            const Divider(height: 28),
            Text('INGEST A NEW ARXIV PAPER',
                style: TextStyle(
                    fontSize: 11, letterSpacing: 1, color: palette.textMuted)),
            const SizedBox(height: 6),
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _arxivController,
                    decoration: const InputDecoration(
                        hintText: 'e.g. 2005.11401', isDense: true),
                  ),
                ),
                const SizedBox(width: 8),
                FilledButton(
                  onPressed: _busy || _arxivController.text.trim().isEmpty
                      ? null
                      : () => _createFor(() =>
                          widget.api.ingestArxiv(_arxivController.text.trim())),
                  child: const Text('Ingest'),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Text(
              'INGEST A NEW WIKIPEDIA ARTICLE',
              style: TextStyle(
                  fontSize: 11, letterSpacing: 1, color: palette.textMuted),
            ),
            const SizedBox(height: 6),
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _wikiController,
                    decoration: const InputDecoration(
                        hintText: 'Paste a wikipedia.org link, or exact title',
                        isDense: true),
                  ),
                ),
                const SizedBox(width: 8),
                FilledButton(
                  onPressed: _busy || _wikiController.text.trim().isEmpty
                      ? null
                      : () => _createFor(() => widget.api
                          .ingestWikipedia(_wikiController.text.trim())),
                  child: const Text('Ingest'),
                ),
              ],
            ),
            const Divider(height: 28),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton(
                onPressed: _busy ? null : _allDocuments,
                child: const Text('All documents (cross-paper)'),
              ),
            ),
            if (_error != null) ...[
              const SizedBox(height: 12),
              Text(_error!,
                  style:
                      TextStyle(color: theme.colorScheme.error, fontSize: 13)),
            ],
          ],
        ),
      ),
    );
  }
}
