from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText

from .core.db import Database
from .core.engine import PanaceaEngine
from .core.defaults import DEFAULTS


class SettingsDialog(tk.Toplevel):
    def __init__(self, master: tk.Misc, db: Database):
        super().__init__(master)
        self.title("Settings")
        self.db = db
        self.resizable(False, False)

        def row(label: str, key: str, show: str | None = None):
            f = tk.Frame(self)
            f.pack(fill="x", padx=10, pady=4)
            tk.Label(f, text=label, width=22, anchor="w").pack(side="left")
            v = tk.StringVar(value=self.db.get_setting(key, DEFAULTS.get(key, "")) or "")
            e = tk.Entry(f, textvariable=v, width=50, show=show if show else "")
            e.pack(side="left")
            return v

        # Provider
        fprov = tk.Frame(self)
        fprov.pack(fill="x", padx=10, pady=4)
        tk.Label(fprov, text="Provider", width=22, anchor="w").pack(side="left")
        self.provider_var = tk.StringVar(value=self.db.get_setting("provider_mode", "auto") or "auto")
        tk.OptionMenu(fprov, self.provider_var, "auto", "novita", "openrouter", "custom").pack(side="left")

        self.novita_base = row("Novita base_url", "novita_base_url")
        self.novita_key = row("Novita api_key", "novita_api_key", show="*")
        self.openrouter_base = row("OpenRouter base_url", "openrouter_base_url")
        self.openrouter_key = row("OpenRouter api_key", "openrouter_api_key", show="*")
        self.custom_base = row("Custom base_url", "custom_base_url")
        self.custom_key = row("Custom api_key", "custom_api_key", show="*")

        # Models
        tk.Label(self, text="Models", font=(None, 10, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.model_light = row("Model light", "model_light")
        self.model_medium = row("Model medium", "model_medium")
        self.model_heavy = row("Model heavy", "model_heavy")
        self.model_arbiter = row("Model arbiter", "model_arbiter")

        # Params
        tk.Label(self, text="Generation", font=(None, 10, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.temperature = row("Temperature", "temperature")
        self.max_tokens = row("Max tokens", "max_tokens")
        self.history_max = row("History max msgs", "history_max_messages")

        # RAG
        tk.Label(self, text="RAG", font=(None, 10, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
        frag = tk.Frame(self)
        frag.pack(fill="x", padx=10, pady=4)
        tk.Label(frag, text="RAG enabled", width=22, anchor="w").pack(side="left")
        self.rag_enabled = tk.IntVar(value=1 if (self.db.get_setting("rag_enabled", "1") == "1") else 0)
        tk.Checkbutton(frag, variable=self.rag_enabled).pack(side="left")
        self.rag_top_k = row("RAG top_k", "rag_top_k")
        self.rag_chunk_chars = row("Chunk chars", "rag_chunk_chars")
        self.rag_overlap_chars = row("Overlap chars", "rag_chunk_overlap_chars")

        fmode = tk.Frame(self)
        fmode.pack(fill="x", padx=10, pady=4)
        tk.Label(fmode, text="Retrieval mode", width=22, anchor="w").pack(side="left")
        self.rag_mode = tk.StringVar(value=self.db.get_setting("rag_retrieval_mode", "bm25") or "bm25")
        tk.OptionMenu(fmode, self.rag_mode, "lexical", "bm25", "faiss").pack(side="left")

        self.rag_emb_model = row("Embedding model", "rag_embedding_model")

        frb = tk.Frame(self)
        frb.pack(fill="x", padx=10, pady=4)
        tk.Button(frb, text="Rebuild FAISS index", command=self._rebuild_index).pack(side="left")

        # Buttons
        fb = tk.Frame(self)
        fb.pack(fill="x", padx=10, pady=12)
        tk.Button(fb, text="Save", command=self._save).pack(side="left")
        tk.Button(fb, text="Close", command=self.destroy).pack(side="right")

        self.grab_set()

    def _save(self):
        self.db.set_setting("provider_mode", self.provider_var.get())
        for k, var in [
            ("novita_base_url", self.novita_base),
            ("novita_api_key", self.novita_key),
            ("openrouter_base_url", self.openrouter_base),
            ("openrouter_api_key", self.openrouter_key),
            ("custom_base_url", self.custom_base),
            ("custom_api_key", self.custom_key),
            ("model_light", self.model_light),
            ("model_medium", self.model_medium),
            ("model_heavy", self.model_heavy),
            ("model_arbiter", self.model_arbiter),
            ("temperature", self.temperature),
            ("max_tokens", self.max_tokens),
            ("history_max_messages", self.history_max),
            ("rag_top_k", self.rag_top_k),
            ("rag_chunk_chars", self.rag_chunk_chars),
            ("rag_chunk_overlap_chars", self.rag_overlap_chars),
            ("rag_embedding_model", self.rag_emb_model),
        ]:
            self.db.set_setting(k, var.get())
        self.db.set_setting("rag_enabled", "1" if self.rag_enabled.get() else "0")
        self.db.set_setting("rag_retrieval_mode", self.rag_mode.get())
        messagebox.showinfo("Saved", "Settings saved.")

    def _rebuild_index(self):
        self.db.set_setting("rag_index_dirty", "1")
        messagebox.showinfo("RAG", "Index marked for rebuild. It will rebuild on next retrieval.")


class RagManagerDialog(tk.Toplevel):
    def __init__(self, master: tk.Misc, db: Database):
        super().__init__(master)
        self.title("RAG Documents")
        self.db = db
        self.geometry("700x300")

        self.listbox = tk.Listbox(self)
        self.listbox.pack(fill="both", expand=True, padx=10, pady=10)

        fb = tk.Frame(self)
        fb.pack(fill="x", padx=10, pady=(0, 10))
        tk.Button(fb, text="Toggle enabled", command=self.toggle).pack(side="left")
        tk.Button(fb, text="Delete", command=self.delete).pack(side="left", padx=6)
        tk.Button(fb, text="Refresh", command=self.refresh).pack(side="left", padx=6)
        tk.Button(fb, text="Close", command=self.destroy).pack(side="right")

        self.refresh()
        self.grab_set()

    def refresh(self):
        self.listbox.delete(0, tk.END)
        self.docs = self.db.rag_list_docs()
        for d in self.docs:
            flag = "ON" if d.get("enabled") else "OFF"
            self.listbox.insert(tk.END, f"[{flag}] #{d['id']}  {d['title']}  ({d['path']})")

    def _selected_doc(self):
        sel = self.listbox.curselection()
        if not sel:
            return None
        idx = sel[0]
        return self.docs[idx]

    def toggle(self):
        d = self._selected_doc()
        if not d:
            return
        self.db.rag_set_doc_enabled(int(d['id']), not bool(d.get('enabled')))
        self.refresh()

    def delete(self):
        d = self._selected_doc()
        if not d:
            return
        if not messagebox.askyesno("Confirm", f"Delete document #{d['id']}?"):
            return
        self.db.rag_delete_doc(int(d['id']))
        self.refresh()


class HistoDialog(tk.Toplevel):
    def __init__(self, master: tk.Misc, db: Database, dialog_id: int, on_saved):
        super().__init__(master)
        self.title("Histo metadata")
        self.db = db
        self.dialog_id = dialog_id
        self.on_saved = on_saved
        self.resizable(False, False)

        st = db.histo_get(dialog_id)

        def row(label: str, var: tk.StringVar):
            f = tk.Frame(self)
            f.pack(fill="x", padx=10, pady=4)
            tk.Label(f, text=label, width=18, anchor="w").pack(side="left")
            tk.Entry(f, textvariable=var, width=50).pack(side="left")

        self.stain = tk.StringVar(value=st.get("stain", ""))
        self.mag = tk.StringVar(value=st.get("magnification", ""))
        self.quality = tk.StringVar(value=st.get("quality", ""))
        self.note = tk.StringVar(value=st.get("note", ""))

        row("Stain", self.stain)
        row("Magnification", self.mag)
        row("Quality", self.quality)
        row("Note", self.note)

        fb = tk.Frame(self)
        fb.pack(fill="x", padx=10, pady=10)
        tk.Button(fb, text="Save", command=self._save).pack(side="left")
        tk.Button(fb, text="Close", command=self.destroy).pack(side="right")

        self.grab_set()

    def _save(self):
        self.db.histo_set(self.dialog_id, self.stain.get(), self.mag.get(), self.quality.get(), self.note.get())
        try:
            self.on_saved()
        except Exception:
            pass
        messagebox.showinfo("Saved", "Histo metadata saved")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Panacea Desktop")
        self.geometry("1100x700")

        self.db = Database()
        self.engine = PanaceaEngine(self.db)

        self.pending_images: list[str] = []

        self._build_ui()
        self._load_dialogs()
        self._load_active_dialog()

        self.worker_thread: threading.Thread | None = None
        self.result_queue: queue.Queue = queue.Queue()
        # Streaming state (updated via queue)
        self._streaming_assistant_msg_id: int | None = None
        self._streaming_text: str = ""
        self.after(100, self._poll_results)

    # ---------- UI ----------
    def _build_ui(self):
        # Top bar
        top = tk.Frame(self)
        top.pack(fill="x")

        tk.Label(top, text="Pack:").pack(side="left", padx=(8, 2))
        self.pack_var = tk.StringVar(value=self.db.get_setting("prompt_pack", "gp") or "gp")
        tk.OptionMenu(top, self.pack_var, "gp", "derm", "histo", command=lambda _=None: self._save_mode_settings()).pack(side="left")

        tk.Label(top, text="Mode:").pack(side="left", padx=(10, 2))
        self.mode_var = tk.StringVar(value=self.db.get_setting("reasoning_mode", "medium") or "medium")
        tk.OptionMenu(top, self.mode_var, "light", "medium", "heavy", "consensus", command=lambda _=None: self._save_mode_settings()).pack(side="left")

        tk.Button(top, text="Settings", command=self._open_settings).pack(side="left", padx=10)
        tk.Button(top, text="Export chat", command=self._export_chat).pack(side="left")
        tk.Button(top, text="RAG docs", command=self._open_rag_manager).pack(side="left")
        tk.Button(top, text="Add doc", command=self._add_doc).pack(side="left", padx=6)
        tk.Button(top, text="Attach image", command=self._attach_images).pack(side="left", padx=6)
        tk.Button(top, text="Histo meta", command=self._open_histo).pack(side="left", padx=6)

        self.attach_label = tk.Label(top, text="")
        self.attach_label.pack(side="right", padx=8)

        # Main split
        main = tk.Frame(self)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, width=280)
        left.pack(side="left", fill="y")

        self.dialogs_list = tk.Listbox(left)
        self.dialogs_list.pack(fill="both", expand=True, padx=8, pady=8)
        self.dialogs_list.bind("<<ListboxSelect>>", self._on_dialog_select)

        lb = tk.Frame(left)
        lb.pack(fill="x", padx=8, pady=(0, 8))
        tk.Button(lb, text="New", command=self._new_dialog).pack(side="left")
        tk.Button(lb, text="Rename", command=self._rename_dialog).pack(side="left", padx=6)
        tk.Button(lb, text="Delete", command=self._delete_dialog).pack(side="left", padx=6)
        tk.Button(lb, text="Clear", command=self._clear_dialog).pack(side="left")

        right = tk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        self.chat = ScrolledText(right, state="disabled", wrap="word")
        self.chat.pack(fill="both", expand=True, padx=8, pady=8)

        bottom = tk.Frame(right)
        bottom.pack(fill="x", padx=8, pady=(0, 8))

        self.input = ScrolledText(bottom, height=4, wrap="word")
        self.input.pack(side="left", fill="both", expand=True)

        tk.Button(bottom, text="Send", command=self._send).pack(side="left", padx=8)

    # ---------- Dialog management ----------
    def _load_dialogs(self):
        self.dialogs = self.db.list_dialogs()
        self.dialogs_list.delete(0, tk.END)
        for d in self.dialogs:
            self.dialogs_list.insert(tk.END, f"#{d['id']} {d['title']}")

    def _select_dialog_in_list(self, dialog_id: int):
        for i, d in enumerate(self.dialogs):
            if int(d['id']) == int(dialog_id):
                self.dialogs_list.selection_clear(0, tk.END)
                self.dialogs_list.selection_set(i)
                self.dialogs_list.activate(i)
                self.dialogs_list.see(i)
                return

    def _load_active_dialog(self):
        dialog_id = self.db.get_active_dialog()
        if dialog_id is None:
            dialog_id = self.db.create_dialog("New dialog")
        self.active_dialog_id = dialog_id
        self._load_dialogs()
        self._select_dialog_in_list(dialog_id)
        self._render_chat()

    def _on_dialog_select(self, event=None):
        sel = self.dialogs_list.curselection()
        if not sel:
            return
        idx = sel[0]
        d = self.dialogs[idx]
        self.active_dialog_id = int(d['id'])
        self.db.set_active_dialog(self.active_dialog_id)
        self._render_chat()

    def _new_dialog(self):
        title = simpledialog.askstring("New dialog", "Title:")
        if not title:
            return
        did = self.db.create_dialog(title)
        self.db.set_active_dialog(did)
        self._load_active_dialog()

    def _rename_dialog(self):
        did = self.active_dialog_id
        title = simpledialog.askstring("Rename", "New title:")
        if not title:
            return
        self.db.rename_dialog(did, title)
        self._load_dialogs()
        self._select_dialog_in_list(did)

    def _delete_dialog(self):
        did = self.active_dialog_id
        if not messagebox.askyesno("Confirm", f"Delete dialog #{did}?"):
            return
        self.db.delete_dialog(did)
        self._load_active_dialog()

    def _clear_dialog(self):
        did = self.active_dialog_id
        if not messagebox.askyesno("Confirm", f"Clear dialog #{did} messages?"):
            return
        self.db.clear_dialog(did)
        self._render_chat()

    # ---------- Chat rendering ----------
    def _render_chat(self):
        msgs = self.db.get_messages(self.active_dialog_id)
        self.chat.configure(state="normal")
        self.chat.delete("1.0", tk.END)
        for m in msgs:
            role = m.get("role")
            content = m.get("content")
            if role == "user":
                self.chat.insert(tk.END, f"You: {content}\n\n")
            elif role == "assistant":
                self.chat.insert(tk.END, f"Panacea: {content}\n\n")
        self.chat.configure(state="disabled")
        self.chat.see(tk.END)

    # ---------- Actions ----------
    def _save_mode_settings(self):
        self.db.set_setting("prompt_pack", self.pack_var.get())
        self.db.set_setting("reasoning_mode", self.mode_var.get())

    def _open_settings(self):
        SettingsDialog(self, self.db)

    def _open_rag_manager(self):
        RagManagerDialog(self, self.db)

    def _add_doc(self):
        path = filedialog.askopenfilename(title="Add document for RAG")
        if not path:
            return
        try:
            chunk_chars = int(self.db.get_setting("rag_chunk_chars", "1200") or "1200")
            overlap = int(self.db.get_setting("rag_chunk_overlap_chars", "200") or "200")
            from .core.rag import add_document_to_rag

            add_document_to_rag(self.db, path, title=None, chunk_chars=chunk_chars, overlap_chars=overlap)
            messagebox.showinfo("RAG", "Document added.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _attach_images(self):
        paths = filedialog.askopenfilenames(title="Attach images", filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")])
        if not paths:
            return
        self.pending_images.extend(list(paths))
        self._update_attach_label()

    def _update_attach_label(self):
        if not self.pending_images:
            self.attach_label.config(text="")
        else:
            self.attach_label.config(text=f"Attached images: {len(self.pending_images)}")

    def _open_histo(self):
        if self.pack_var.get() != "histo":
            if not messagebox.askyesno("Histo", "Switch prompt pack to 'histo'?"):
                return
            self.pack_var.set("histo")
            self._save_mode_settings()
        HistoDialog(self, self.db, self.active_dialog_id, on_saved=lambda: None)

    def _send(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Busy", "Generation is already running.")
            return
        text = self.input.get("1.0", tk.END).strip()
        if not text:
            return
        self._save_mode_settings()

        # Clear input immediately
        self.input.delete("1.0", tk.END)

        # Add user message (will be persisted inside engine handle)
        did = self.active_dialog_id
        images = list(self.pending_images)
        self.pending_images.clear()
        self._update_attach_label()

        self.chat.configure(state="normal")
        self.chat.insert(tk.END, f"You: {text}\n\n")
        self.chat.insert(tk.END, "Panacea: ")
        self._assistant_start_index = self.chat.index(tk.END)
        self.chat.insert(tk.END, "\n\n")
        self.chat.configure(state="disabled")
        self.chat.see(tk.END)

        def worker():
            try:
                msg_id, provider, model, it, _candidates = self.engine.handle_user_turn_stream(did, text, images)
                acc = ""
                for piece in it:
                    acc += piece
                    self.result_queue.put(("delta", piece))
                # persist final content
                self.db.update_message_content(msg_id, acc)
                self.result_queue.put(("done", {"provider": provider, "model": model}))
            except Exception as e:
                self.result_queue.put(("err", str(e)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _poll_results(self):
        try:
            while True:
                kind, payload = self.result_queue.get_nowait()
                if kind == "delta":
                    self.chat.configure(state="normal")
                    # insert before the trailing newlines we added
                    self.chat.insert("end-2c", payload)
                    self.chat.configure(state="disabled")
                    self.chat.see(tk.END)
                elif kind == "done":
                    # re-render to ensure DB is source of truth
                    self._render_chat()
                else:
                    messagebox.showerror("Error", payload)
                    self._render_chat()
        except queue.Empty:
            pass
        self.after(100, self._poll_results)

    def _export_chat(self):
        did = getattr(self, "active_dialog_id", None)
        if did is None:
            return
        path = filedialog.asksaveasfilename(
            title="Export chat",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("Markdown", "*.md"), ("JSON", "*.json")],
        )
        if not path:
            return
        try:
            msgs = self.db.get_messages(did)
            if path.lower().endswith(".json"):
                import json

                with open(path, "w", encoding="utf-8") as f:
                    json.dump(msgs, f, ensure_ascii=False, indent=2)
            else:
                is_md = path.lower().endswith(".md")
                with open(path, "w", encoding="utf-8") as f:
                    for m in msgs:
                        role = m.get("role")
                        content = (m.get("content") or "").strip()
                        atts = m.get("attachments") or {}
                        if is_md:
                            f.write(f"## {role}\n\n{content}\n\n")
                        else:
                            prefix = "You" if role == "user" else "Panacea" if role == "assistant" else role
                            f.write(f"{prefix}: {content}\n\n")
                        if atts.get("images"):
                            f.write(f"[attachments: {len(atts['images'])} image(s)]\n\n")
            messagebox.showinfo("Export", "Chat exported.")
        except Exception as e:
            messagebox.showerror("Error", str(e))


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
