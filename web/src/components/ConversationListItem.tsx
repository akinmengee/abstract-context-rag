import { useState } from "react";
import { Link } from "react-router-dom";

import styles from "./ConversationListItem.module.css";

export function ConversationListItem({
  id,
  title,
  active,
  onRename,
  onDelete,
}: {
  id: string;
  title: string;
  active: boolean;
  onRename(id: string, title: string): void;
  onDelete(id: string): void;
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draftTitle, setDraftTitle] = useState(title);

  function commitRename() {
    setEditing(false);
    const trimmed = draftTitle.trim();
    if (trimmed && trimmed !== title) onRename(id, trimmed);
    else setDraftTitle(title);
  }

  return (
    <div className={`${styles.item} ${active ? styles.active : ""}`}>
      {editing ? (
        <input
          className={styles.titleInput}
          autoFocus
          value={draftTitle}
          onChange={(event) => setDraftTitle(event.target.value)}
          onBlur={commitRename}
          onKeyDown={(event) => {
            if (event.key === "Enter") commitRename();
            if (event.key === "Escape") {
              setDraftTitle(title);
              setEditing(false);
            }
          }}
        />
      ) : (
        <Link to={`/chat/${id}`} className={styles.title}>
          {title}
        </Link>
      )}

      <button
        className={styles.menuButton}
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          setMenuOpen((open) => !open);
        }}
        aria-label="Conversation options"
      >
        ⋯
      </button>

      {menuOpen && (
        <div className={styles.menu} onClick={(event) => event.stopPropagation()}>
          <button
            onClick={() => {
              setMenuOpen(false);
              setEditing(true);
            }}
          >
            Rename
          </button>
          <button
            className={styles.danger}
            onClick={() => {
              setMenuOpen(false);
              onDelete(id);
            }}
          >
            Delete
          </button>
        </div>
      )}
    </div>
  );
}
