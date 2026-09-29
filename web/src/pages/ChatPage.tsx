import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import { getConversation } from "../api/client";
import { ChatPanel } from "../components/ChatPanel";
import type { DisplayMessage } from "../components/MessageBubble";
import { Sidebar } from "../components/Sidebar";
import styles from "./ChatPage.module.css";

export function ChatPage() {
  const { conversationId } = useParams<{ conversationId: string }>();
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (!conversationId) {
      setMessages([]);
      setLoaded(true);
      return;
    }
    setLoaded(false);
    getConversation(conversationId).then((detail) => {
      setMessages(
        detail.messages.map((message) => ({
          role: message.role,
          content: message.content,
          citations: message.citations,
        })),
      );
      setLoaded(true);
    });
  }, [conversationId]);

  return (
    <div className={styles.layout}>
      <Sidebar activeConversationId={conversationId} />
      {!conversationId ? (
        <div className={styles.emptyState}>Start a New Chat to begin.</div>
      ) : (
        loaded && <ChatPanel key={conversationId} conversationId={conversationId} initialMessages={messages} />
      )}
    </div>
  );
}
