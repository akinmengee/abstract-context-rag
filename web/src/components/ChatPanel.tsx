import { useEffect, useRef, useState, type KeyboardEvent } from "react";

import { streamChat } from "../api/sse";
import { MessageBubble, type DisplayMessage } from "./MessageBubble";
import styles from "./ChatPanel.module.css";

export function ChatPanel({
  conversationId,
  initialMessages,
}: {
  conversationId: string;
  initialMessages: DisplayMessage[];
}) {
  const [messages, setMessages] = useState<DisplayMessage[]>(initialMessages);
  const [question, setQuestion] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMessages(initialMessages);
  }, [conversationId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    const text = question.trim();
    if (!text || sending) return;
    setError("");
    setQuestion("");
    setSending(true);
    setMessages((current) => [...current, { role: "user", content: text }, { role: "assistant", content: "" }]);

    try {
      for await (const event of streamChat(conversationId, text)) {
        if (event.event === "token" && event.token) {
          setMessages((current) => {
            const next = [...current];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, content: last.content + event.token };
            return next;
          });
        } else if (event.event === "done" && event.answer) {
          const finalAnswer = event.answer;
          setMessages((current) => {
            const next = [...current];
            next[next.length - 1] = {
              role: "assistant",
              content: finalAnswer.text,
              citations: finalAnswer.citations,
            };
            return next;
          });
        }
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSending(false);
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  }

  return (
    <div className={styles.panel}>
      {messages.length === 0 ? (
        <div className={styles.empty}>
          <p className={styles.emptyTitle}>Ask about this source.</p>
          <p className={styles.emptySubtitle}>Every answer cites the passage it came from.</p>
        </div>
      ) : (
        <div className={styles.messages}>
          {messages.map((message, index) => (
            <MessageBubble key={index} message={message} />
          ))}
          <div ref={endRef} />
        </div>
      )}

      {error && <p className={styles.error}>{error}</p>}

      <div className={styles.composer}>
        <div className={styles.composerInner}>
          <textarea
            rows={1}
            placeholder="Ask a question…"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleKeyDown}
            disabled={sending}
          />
          <button onClick={send} disabled={sending || !question.trim()}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
