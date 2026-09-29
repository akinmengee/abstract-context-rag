import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  deleteConversation,
  listConversations,
  renameConversation,
  type ConversationSummary,
} from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { ConversationListItem } from "./ConversationListItem";
import { NewChatDialog } from "./NewChatDialog";
import styles from "./Sidebar.module.css";

export function Sidebar({ activeConversationId }: { activeConversationId?: string }) {
  const navigate = useNavigate();
  const { userEmail, logout } = useAuth();
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);
  const [search, setSearch] = useState("");

  useEffect(() => {
    refresh();
  }, [activeConversationId]);

  async function refresh() {
    setConversations(await listConversations());
  }

  async function handleRename(id: string, title: string) {
    await renameConversation(id, title);
    refresh();
  }

  async function handleDelete(id: string) {
    if (!confirm("Delete this conversation?")) return;
    await deleteConversation(id);
    if (id === activeConversationId) navigate("/");
    refresh();
  }

  const visibleConversations = search.trim()
    ? conversations.filter((conversation) =>
        conversation.title.toLowerCase().includes(search.trim().toLowerCase()),
      )
    : conversations;

  return (
    <>
      <button
        className={styles.toggle}
        onClick={() => setPanelOpen((open) => !open)}
        aria-label="Toggle menu"
        aria-expanded={panelOpen}
      >
        ⋮
      </button>

      {panelOpen && <div className={styles.backdrop} onClick={() => setPanelOpen(false)} />}

      <aside className={`${styles.sidebar} ${panelOpen ? styles.open : ""}`}>
        <div className={styles.brand}>
          <span className={styles.brandMark} role="img" aria-label="Abstract Context RAG" />
          <div className={styles.brandText}>
            <span className={styles.brandName}>Abstract Context</span>
            <span className={styles.brandAccent}>RAG</span>
          </div>
        </div>

        <button
          className={styles.newChat}
          onClick={() => {
            setDialogOpen(true);
            setPanelOpen(false);
          }}
        >
          + New Chat
        </button>

        <input
          className={styles.search}
          type="text"
          placeholder="Search conversations"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          aria-label="Search conversations"
        />

        <div className={styles.label}>Recents</div>
        <nav className={styles.list}>
          {visibleConversations.map((conversation) => (
            <ConversationListItem
              key={conversation.id}
              id={conversation.id}
              title={conversation.title}
              active={conversation.id === activeConversationId}
              onRename={handleRename}
              onDelete={handleDelete}
            />
          ))}
          {visibleConversations.length === 0 && (
            <div className={styles.empty}>
              {search.trim() ? "No conversations match your search." : "No conversations yet."}
            </div>
          )}
        </nav>

        <div className={styles.footer}>
          <span>{userEmail}</span>
          <button className={styles.logout} onClick={logout}>
            Log out
          </button>
        </div>
      </aside>

      {dialogOpen && (
        <NewChatDialog
          onClose={() => setDialogOpen(false)}
          onCreated={(conversation) => {
            setDialogOpen(false);
            refresh();
            navigate(`/chat/${conversation.id}`);
          }}
        />
      )}
    </>
  );
}
