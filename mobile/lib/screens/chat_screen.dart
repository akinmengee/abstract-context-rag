import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../api/client.dart';
import '../api/sse.dart';
import '../auth_state.dart';
import '../theme.dart';
import '../widgets/brand_mark.dart';
import '../widgets/message_bubble.dart';
import '../widgets/new_chat_dialog.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _api = RagApi();
  final _questionController = TextEditingController();
  final _scrollController = ScrollController();

  List<ConversationSummary> _conversations = [];
  String? _activeConversationId;
  List<DisplayMessage> _messages = [];
  bool _loadingMessages = false;
  bool _sending = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _refreshConversations();
  }

  Future<void> _refreshConversations() async {
    final list = await _api.listConversations();
    if (mounted) setState(() => _conversations = list);
  }

  Future<void> _openConversation(String id) async {
    Navigator.of(context).maybePop(); // close the drawer if open
    setState(() {
      _activeConversationId = id;
      _loadingMessages = true;
      _messages = [];
    });
    final detail = await _api.getConversation(id);
    if (!mounted) return;
    setState(() {
      _messages = detail.messages
          .map((m) => DisplayMessage(
              role: m.role, content: m.content, citations: m.citations))
          .toList();
      _loadingMessages = false;
    });
  }

  Future<void> _startNewChat() async {
    final conversation = await showNewChatDialog(context, _api);
    if (conversation == null) return;
    await _refreshConversations();
    await _openConversation(conversation.id);
  }

  Future<void> _renameConversation(ConversationSummary conversation) async {
    final controller = TextEditingController(text: conversation.title);
    final title = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Rename conversation'),
        content: TextField(controller: controller, autofocus: true),
        actions: [
          TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Cancel')),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(controller.text.trim()),
            child: const Text('Save'),
          ),
        ],
      ),
    );
    if (title == null || title.isEmpty || title == conversation.title) return;
    await _api.renameConversation(conversation.id, title);
    await _refreshConversations();
  }

  Future<void> _deleteConversation(ConversationSummary conversation) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete this conversation?'),
        actions: [
          TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.of(context).pop(true),
              child: const Text('Delete')),
        ],
      ),
    );
    if (confirmed != true) return;
    await _api.deleteConversation(conversation.id);
    if (conversation.id == _activeConversationId) {
      setState(() {
        _activeConversationId = null;
        _messages = [];
      });
    }
    await _refreshConversations();
  }

  Future<void> _send() async {
    final text = _questionController.text.trim();
    final conversationId = _activeConversationId;
    if (text.isEmpty || _sending || conversationId == null) return;

    setState(() {
      _sending = true;
      _error = null;
      _questionController.clear();
      _messages = [
        ..._messages,
        DisplayMessage(role: 'user', content: text),
        DisplayMessage(role: 'assistant', content: ''),
      ];
    });
    _scrollToEnd();

    try {
      await for (final event in streamChat(conversationId, text)) {
        if (event.event == 'token' && event.token != null) {
          setState(() => _messages.last.content += event.token!);
        } else if (event.event == 'done' && event.answer != null) {
          setState(() {
            _messages[_messages.length - 1] = DisplayMessage(
              role: 'assistant',
              content: event.answer!.text,
              citations: event.answer!.citations,
            );
          });
        }
        _scrollToEnd();
      }
    } catch (error) {
      setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  void _scrollToEnd() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final palette = context.palette;

    return Scaffold(
      appBar: AppBar(
        title: Text(
          _conversations
              .firstWhere(
                (c) => c.id == _activeConversationId,
                orElse: () => ConversationSummary(
                    id: '',
                    title: 'Abstract Context RAG',
                    documentId: null,
                    updatedAt: DateTime.now()),
              )
              .title,
          style: theme.textTheme.titleMedium,
        ),
      ),
      drawer: Drawer(
        backgroundColor: palette.bgSidebar,
        child: SafeArea(
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
                child: Row(
                  children: [
                    BrandMark(color: theme.colorScheme.onSurface, size: 26),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text.rich(
                        TextSpan(
                          style: theme.textTheme.titleMedium,
                          children: [
                            const TextSpan(text: 'Abstract Context '),
                            TextSpan(
                                text: 'RAG',
                                style: TextStyle(
                                    color: theme.colorScheme.primary)),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: SizedBox(
                  width: double.infinity,
                  child: OutlinedButton.icon(
                    onPressed: _startNewChat,
                    icon: const Icon(Icons.add),
                    label: const Text('New Chat'),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Align(
                  alignment: Alignment.centerLeft,
                  child: Text('RECENTS',
                      style: TextStyle(
                          fontSize: 11,
                          letterSpacing: 1,
                          color: palette.textMuted)),
                ),
              ),
              Expanded(
                child: ListView.builder(
                  itemCount: _conversations.length,
                  itemBuilder: (context, index) {
                    final conversation = _conversations[index];
                    final active = conversation.id == _activeConversationId;
                    return ListTile(
                      dense: true,
                      selected: active,
                      selectedTileColor: palette.bgHover,
                      title: Text(conversation.title,
                          maxLines: 1, overflow: TextOverflow.ellipsis),
                      onTap: () => _openConversation(conversation.id),
                      trailing: PopupMenuButton<String>(
                        icon: const Icon(Icons.more_horiz, size: 18),
                        onSelected: (value) {
                          if (value == 'rename') {
                            _renameConversation(conversation);
                          }
                          if (value == 'delete') {
                            _deleteConversation(conversation);
                          }
                        },
                        itemBuilder: (context) => const [
                          PopupMenuItem(value: 'rename', child: Text('Rename')),
                          PopupMenuItem(value: 'delete', child: Text('Delete')),
                        ],
                      ),
                    );
                  },
                ),
              ),
              const Divider(height: 1),
              Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    Expanded(
                      child: Text(
                        context.watch<AuthState>().email ?? '',
                        style:
                            TextStyle(fontSize: 12, color: palette.textMuted),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    TextButton(
                      onPressed: () => context.read<AuthState>().logout(),
                      child: const Text('Log out'),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
      body: _activeConversationId == null
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text('Start a New Chat to begin.',
                        style: theme.textTheme.titleMedium),
                    const SizedBox(height: 8),
                    Text(
                      'Every answer cites the passage it came from.',
                      style: TextStyle(color: palette.textMuted),
                    ),
                  ],
                ),
              ),
            )
          : Column(
              children: [
                Expanded(
                  child: _loadingMessages
                      ? const Center(child: CircularProgressIndicator())
                      : ListView.builder(
                          controller: _scrollController,
                          padding: const EdgeInsets.all(16),
                          itemCount: _messages.length,
                          itemBuilder: (context, index) =>
                              MessageBubble(message: _messages[index]),
                        ),
                ),
                if (_error != null)
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: Text(_error!,
                        style: TextStyle(
                            color: theme.colorScheme.error, fontSize: 13)),
                  ),
                SafeArea(
                  top: false,
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Row(
                      children: [
                        Expanded(
                          child: TextField(
                            controller: _questionController,
                            decoration: const InputDecoration(
                                hintText: 'Ask a question…'),
                            minLines: 1,
                            maxLines: 4,
                            enabled: !_sending,
                            onSubmitted: (_) => _send(),
                          ),
                        ),
                        const SizedBox(width: 8),
                        IconButton.filled(
                          onPressed: _sending ? null : _send,
                          icon: const Icon(Icons.arrow_upward),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
    );
  }
}
