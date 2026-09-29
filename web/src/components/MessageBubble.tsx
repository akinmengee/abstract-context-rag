import type { Citation } from "../api/client";
import styles from "./MessageBubble.module.css";

export interface DisplayMessage {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[] | null;
}

// Mirrors prompts.ABSTAIN_MESSAGE on the backend - the one answer that
// isn't grounded in anything, so it gets a quieter, honest treatment
// instead of the citation styling every other answer earns.
const ABSTAIN_MESSAGE = "This source does not contain that information.";

export function MessageBubble({ message }: { message: DisplayMessage }) {
  const isAbstain = message.role === "assistant" && message.content === ABSTAIN_MESSAGE;
  const variant = message.role === "user" ? styles.user : styles.assistant;

  return (
    <div className={`${styles.bubble} ${variant} ${isAbstain ? styles.abstain : ""}`}>
      <div>{message.content}</div>
      {message.citations && message.citations.length > 0 && (
        <ul className={styles.citations}>
          {message.citations.map((citation) => (
            <li key={citation.chunk_id}>
              <span className="citationMark">{citation.marker}</span>
              <span className={styles.citationSource}>
                {citation.title}
                {citation.section ? ` — ${citation.section}` : ""}
                {citation.page ? `, p.${citation.page}` : ""}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
