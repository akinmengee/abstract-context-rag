import 'package:flutter/material.dart';

import '../api/client.dart';
import '../theme.dart';

class DisplayMessage {
  DisplayMessage({required this.role, required this.content, this.citations});
  final String role; // "user" | "assistant"
  String content;
  List<Citation>? citations;
}

// Mirrors prompts.ABSTAIN_MESSAGE on the backend.
const _abstainMessage = 'This source does not contain that information.';

class MessageBubble extends StatelessWidget {
  const MessageBubble({super.key, required this.message});

  final DisplayMessage message;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final palette = context.palette;
    final isUser = message.role == 'user';
    final isAbstain = !isUser && message.content == _abstainMessage;

    final bubbleColor = isUser
        ? theme.colorScheme.primary
        : isAbstain
            ? Colors.transparent
            : theme.colorScheme.surface;
    final textColor =
        isUser ? theme.colorScheme.onPrimary : theme.colorScheme.onSurface;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 520),
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 4),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          decoration: BoxDecoration(
            color: bubbleColor,
            borderRadius: BorderRadius.only(
              topLeft: const Radius.circular(14),
              topRight: const Radius.circular(14),
              bottomLeft: Radius.circular(isUser ? 14 : 4),
              bottomRight: Radius.circular(isUser ? 4 : 14),
            ),
            border: !isUser
                ? Border.all(
                    color: theme.colorScheme.outline,
                    style: isAbstain ? BorderStyle.solid : BorderStyle.solid,
                  )
                : null,
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                message.content,
                style: isAbstain
                    ? theme.textTheme.bodyMedium?.copyWith(
                        color: palette.textMuted,
                        fontStyle: FontStyle.italic,
                      )
                    : (isUser
                            ? theme.textTheme.bodyMedium
                            : theme.textTheme.titleMedium)
                        ?.copyWith(
                            color: textColor,
                            height: 1.5,
                            fontWeight: FontWeight.normal),
              ),
              if (message.citations != null &&
                  message.citations!.isNotEmpty) ...[
                const SizedBox(height: 10),
                ...message.citations!.map(
                  (citation) => Padding(
                    padding: const EdgeInsets.only(bottom: 4),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 6, vertical: 1),
                          decoration: BoxDecoration(
                            color: palette.evidenceBg,
                            borderRadius: BorderRadius.circular(999),
                          ),
                          child: Text(
                            '${citation.marker}',
                            style: TextStyle(
                              color: palette.evidence,
                              fontSize: 11,
                              fontFamily: 'monospace',
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                        const SizedBox(width: 6),
                        Expanded(
                          child: Text(
                            [
                              citation.title,
                              if (citation.section != null) citation.section,
                              if (citation.page != null) 'p.${citation.page}',
                            ].join(' — '),
                            style: TextStyle(
                                color: palette.textMuted, fontSize: 12),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
