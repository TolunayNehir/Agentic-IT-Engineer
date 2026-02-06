import os
import glob
import re
import json
from datetime import datetime
from typing import Any, List, Optional, Tuple
import warnings

import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore
try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    NearestNeighbors = None  # type: ignore

try:
    from groq import Groq
except ImportError:
    raise ImportError(
        "The groq library is not installed. Run `pip install groq` to continue."
    )


def get_groq_client() -> Groq:
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Prompt the user for their API key if not set.
        api_key = input("Enter your Groq API key: ").strip()
    return Groq(api_key=api_key)


class MemoryStore:

    def __init__(self, path: str, max_messages: int = 40) -> None:
        self.path = path
        self.max_messages = max_messages
        self._messages: List[dict] = []
        self.load()

    def load(self) -> None:
        try:
            if not os.path.exists(self.path):
                self._messages = []
                return
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages = data.get("messages", []) if isinstance(data, dict) else []
            if not isinstance(messages, list):
                self._messages = []
                return
            cleaned: List[dict] = []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                content = msg.get("content")
                if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                    cleaned.append({"role": role, "content": content.strip()})
            self._messages = cleaned[-self.max_messages :]
        except Exception:
            self._messages = []

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            payload = {"version": 1, "messages": self._messages[-self.max_messages :]}
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            warnings.warn(f"Memory save failed: {e}", RuntimeWarning, stacklevel=2)

    def get_messages(self) -> List[dict]:
        return list(self._messages)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if user_text:
            self._messages.append({"role": "user", "content": user_text})
        if assistant_text:
            self._messages.append({"role": "assistant", "content": assistant_text})
        self._messages = self._messages[-self.max_messages :]
        self.save()


class DocumentStore:

    def __init__(self, docs_path: str, chunk_size: int = 500, overlap: int = 50) -> None:
        if SentenceTransformer is None or NearestNeighbors is None:
            raise RuntimeError(
                "RAG dependencies are missing. Install sentence_transformers and scikit-learn to use RAG."
            )
        self.docs_path = docs_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._model: Any = SentenceTransformer("all-MiniLM-L6-v2")
        self._texts: List[str] = []
        self._sources: List[str] = []
        # Load documents and build index
        self._load_documents()
        self._build_index()

    def _load_documents(self) -> None:
        txt_files = glob.glob(os.path.join(self.docs_path, "**", "*.txt"), recursive=True)
        pdf_files = glob.glob(os.path.join(self.docs_path, "**", "*.pdf"), recursive=True)
        for path in txt_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                self._chunk_document(text, path)
            except Exception:
                continue
        for path in pdf_files:
            try:
                # Use PyPDF2 for PDF parsing
                import PyPDF2  # type: ignore
            except ImportError:
                raise RuntimeError("PyPDF2 is required to read PDF files. Install it with `pip install PyPDF2`." )
            try:
                text = ""
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() or ""
                self._chunk_document(text, path)
            except Exception:
                continue

    def _chunk_document(self, text: str, source: str) -> None:

        if not text:
            warnings.warn(
            f"Empty document skipped (source={source})",
            UserWarning,
            stacklevel=2
            )
            return

        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
           warnings.warn(
            f"Whitespace-only document skipped (source={source})",
            UserWarning,
            stacklevel=2
            )
           return

        source_name = os.path.basename(source)

        # Paragraph-aware + semantic sentence segmentation.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", normalized) if p and p.strip()]
        for paragraph in paragraphs:
            paragraph = re.sub(r"\s+", " ", paragraph)
            sentences = self._split_sentences(paragraph)
            if not sentences:
                continue

            # Ensure no single unit exceeds chunk_size.
            units: List[str] = []
            for sent in sentences:
                if len(sent) > self.chunk_size:
                    units.extend(self._split_long_text(sent))
                else:
                    units.append(sent)
            if not units:
                continue

            segments = self._semantic_segment_units(
                units,
                similarity_threshold=0.58,
                min_sentences=2,
                min_chars=max(80, int(self.chunk_size * 0.35)),
            )

            # Apply overlap between successive segments (sentence-level overlap budgeted by `self.overlap`).
            prev_segment: Optional[List[str]] = None
            for segment in segments:
                combined = segment
                if prev_segment and self.overlap > 0:
                    overlap_count = self._overlap_sentence_count(prev_segment)
                    if overlap_count > 0:
                        prefix = prev_segment[-overlap_count:]
                        combined = prefix + segment
                        # Ensure combined stays within chunk_size; trim overlap prefix if needed.
                        while prefix and len(" ".join(combined)) > self.chunk_size:
                            prefix = prefix[1:]
                            combined = prefix + segment

                chunk_text = " ".join(combined).strip()
                if chunk_text:
                    self._texts.append(chunk_text)
                    self._sources.append(source_name)
                prev_segment = segment

    def _split_sentences(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        # Heuristic sentence splitter.
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p and p.strip()]

    def _normalize_rows(self, mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return mat / norms

    def _semantic_segment_units(
        self,
        units: List[str],
        similarity_threshold: float,
        min_sentences: int,
        min_chars: int,
    ) -> List[List[str]]:

        if not units:
            return []
        if len(units) == 1:
            return [units]

        emb = np.asarray(self._model.encode(units, show_progress_bar=False))
        if emb.ndim != 2 or emb.shape[0] != len(units):
            # Fallback to non-semantic segmentation if embeddings look unexpected.
            return self._budget_segment_units(units)

        emb = self._normalize_rows(emb)

        segments: List[List[str]] = []
        current: List[str] = [units[0]]
        current_len = len(units[0])
        sum_vec = emb[0].copy()

        for i in range(1, len(units)):
            unit = units[i]
            unit_len = len(unit) + (1 if current else 0)

            # Hard boundary: character budget exceeded.
            if current and current_len + unit_len > self.chunk_size:
                segments.append(current)
                current = [unit]
                current_len = len(unit)
                sum_vec = emb[i].copy()
                continue

            centroid = sum_vec
            centroid_norm = float(np.linalg.norm(centroid))
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            sim = float(np.dot(emb[i], centroid))

            topic_shift = sim < similarity_threshold
            if topic_shift and len(current) >= min_sentences and current_len >= min_chars:
                segments.append(current)
                current = [unit]
                current_len = len(unit)
                sum_vec = emb[i].copy()
                continue

            current.append(unit)
            current_len += unit_len
            sum_vec = sum_vec + emb[i]

        if current:
            segments.append(current)
        return segments

    def _budget_segment_units(self, units: List[str]) -> List[List[str]]:
        segments: List[List[str]] = []
        current: List[str] = []
        current_len = 0
        for unit in units:
            add_len = len(unit) + (1 if current else 0)
            if current and current_len + add_len > self.chunk_size:
                segments.append(current)
                current = [unit]
                current_len = len(unit)
            else:
                current.append(unit)
                current_len += add_len
        if current:
            segments.append(current)
        return segments

    def _overlap_sentence_count(self, segment_sentences: List[str]) -> int:
        if self.overlap <= 0:
            return 0
        overlap_len = 0
        count = 0
        for sentence in reversed(segment_sentences):
            if overlap_len >= self.overlap:
                break
            overlap_len += len(sentence) + (1 if count > 0 else 0)
            count += 1
        if count >= len(segment_sentences):
            return max(len(segment_sentences) - 1, 0)
        return count

    def _split_long_text(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            chunks.append(text[start:end])
            start = max(end - self.overlap, start + 1)
        return chunks

    def _build_index(self) -> None:
       
        if not self._texts:
            self._embeddings = np.array([])
            self._nn = None
            return
        # Compute embeddings
        self._embeddings = np.vstack(
            self._model.encode(self._texts, show_progress_bar=False)
        )
        # Fit nearest neighbour model
        self._nn: Any = NearestNeighbors(n_neighbors=5, metric="cosine")
        self._nn.fit(self._embeddings)

    def query(self, query: str, top_k: int = 3) -> Tuple[str, List[str]]:
        
        if not self._texts or self._nn is None:
            return "", []
        query_emb = self._model.encode([query])[0].reshape(1, -1)
        distances, indices = self._nn.kneighbors(query_emb, n_neighbors=min(top_k, len(self._texts)))
        context_chunks = []
        sources = []
        for idx in indices[0]:
            context_chunks.append(self._texts[idx])
            sources.append(self._sources[idx])
        return "\n\n".join(context_chunks), sources


def choose_model() -> str:
    
    models = {
        "1": ("openai/gpt-oss-20b", "GPT‑OSS 20B"),
        "2": ("openai/gpt-oss-120b", "GPT‑OSS 120B"),
        "3": ("custom", "Enter custom model name")
    }
    print("\nSelect the language model:")
    for key, (_, label) in models.items():
        print(f"  {key}. {label}")
    choice = input("Model [1/2/3]: ").strip()
    model_choice = models.get(choice, models["1"])[0]
    if model_choice == "custom":
        custom = input(
            "Enter the full model identifier (e.g. openai/gpt-oss-120b): "
        ).strip()
        return custom
    return model_choice


def choose_environment() -> str:
    
    print("\nSelect the environment:")
    print("  1. Server")
    print("  2. Network")
    env_choice = input("Environment [1/2]: ").strip()
    return "server" if env_choice == "1" else "network"


def choose_vendor(environment: str) -> str:
    
    if environment == "server":
        print("\nSelect the server platform:")
        print("  1. Windows")
        print("  2. Linux")
        choice = input("Platform [1/2]: ").strip()
        return "windows" if choice == "1" else "linux"
    else:
        print("\nSelect the network vendor:")
        print("  1. Cisco")
        print("  2. MikroTik")
        choice = input("Vendor [1/2]: ").strip()
        return "cisco" if choice == "1" else "mikrotik"


def choose_mode() -> str:
    
    print("\nSelect the mode:")
    print("  1. Build (create new configuration)")
    print("  2. Troubleshoot (diagnose and fix issues)")
    print("  3. Diagram (generate Mermaid diagram)")
    mode_choice = input("Mode [1/2/3]: ").strip()
    if mode_choice == "1":
        return "build"
    if mode_choice == "2":
        return "troubleshoot"
    return "diagram"


def _extract_mermaid(text: str) -> str:

    if not text:
        return ""
    match = re.search(r"```\s*mermaid\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        if re.match(r"^(flowchart|graph|sequenceDiagram|stateDiagram|erDiagram|classDiagram)\b", candidate):
            return candidate

    return text.strip()


def _save_mermaid(mermaid: str) -> str:

    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "diagrams")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"diagram_{ts}.mmd")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(mermaid.rstrip() + "\n")
    return out_path


def call_llm(
    client: Groq,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    memory_messages: Optional[List[dict]] = None,
) -> str:
    
    messages: List[dict] = [{"role": "system", "content": system_prompt}]
    if memory_messages:
        messages.extend(memory_messages)
    messages.append({"role": "user", "content": user_prompt})
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return completion.choices[0].message.content.strip()


def build_server(
    platform: str,
    description: str,
    client: Groq,
    model: str,
    docstore: Optional[DocumentStore] = None,
    top_k: int = 3,
    memory_messages: Optional[List[dict]] = None,
) -> str:
    
    system_prompt = (
        "You are an experienced systems engineer tasked with designing a "
        f"{'Windows' if platform == 'windows' else 'Linux'} server deployment. "
        "Produce a clear, step‑by‑step plan and include any shell or PowerShell commands "
        "required to perform the installation."
    )
    user_prompt = description
    # Append retrieved context if RAG is enabled
    if docstore:
        context, sources = docstore.query(description, top_k=top_k)
        if context:
            # Prepend context instructions to system prompt
            system_prompt = (
                "You are an experienced systems engineer. Use the following context to inform your answer.\n"
                f"Context:\n{context}\n\nSources: {', '.join(sources)}\n\n"
                + system_prompt
            )
    return call_llm(client, model, system_prompt, user_prompt, memory_messages=memory_messages)


def troubleshoot_server(
    platform: str,
    issue: str,
    client: Groq,
    model: str,
    docstore: Optional[DocumentStore] = None,
    top_k: int = 3,
    memory_messages: Optional[List[dict]] = None,
) -> str:
    
    system_prompt = (
        "You are a senior system administrator. Given a problem description, "
        f"diagnose the issue on a {'Windows' if platform == 'windows' else 'Linux'} "
        "server and provide step‑by‑step actions to resolve it. Include any relevant "
        "commands (PowerShell, bash, etc.) and explain why each step is necessary."
    )
    user_prompt = issue
    if docstore:
        context, sources = docstore.query(issue, top_k=top_k)
        if context:
            system_prompt = (
                "You are a senior system administrator. Use the following context to inform your diagnosis.\n"
                f"Context:\n{context}\n\nSources: {', '.join(sources)}\n\n"
                + system_prompt
            )
    return call_llm(client, model, system_prompt, user_prompt, memory_messages=memory_messages)


def build_network(
    vendor: str,
    description: str,
    client: Groq,
    model: str,
    docstore: Optional[DocumentStore] = None,
    top_k: int = 3,
    memory_messages: Optional[List[dict]] = None,
) -> str:
    
    system_prompt = (
        "You are a professional network architect with deep knowledge of "
        f"{'Cisco IOS' if vendor == 'cisco' else 'MikroTik RouterOS'}. "
        "When given a high‑level description of a desired network, design a full topology "
        "including IP addressing, VLANs and routing. Provide CLI commands for the chosen vendor "
        "to implement the design."
    )
    user_prompt = description
    if docstore:
        context, sources = docstore.query(description, top_k=top_k)
        if context:
            system_prompt = (
                "You are a professional network architect. Use the following context to inform your design.\n"
                f"Context:\n{context}\n\nSources: {', '.join(sources)}\n\n"
                + system_prompt
            )
    return call_llm(client, model, system_prompt, user_prompt, memory_messages=memory_messages)


def troubleshoot_network(
    vendor: str,
    issue: str,
    client: Groq,
    model: str,
    docstore: Optional[DocumentStore] = None,
    top_k: int = 3,
    memory_messages: Optional[List[dict]] = None,
) -> str:
    
    system_prompt = (
        "You are a senior network engineer experienced with "
        f"{'Cisco' if vendor == 'cisco' else 'MikroTik'} equipment. "
        "Given a network issue description, identify likely root causes and provide "
        "troubleshooting steps and configuration commands to resolve the problem. "
        "Explain your reasoning where appropriate."
    )
    user_prompt = issue
    if docstore:
        context, sources = docstore.query(issue, top_k=top_k)
        if context:
            system_prompt = (
                "You are a senior network engineer. Use the following context to inform your troubleshooting.\n"
                f"Context:\n{context}\n\nSources: {', '.join(sources)}\n\n"
                + system_prompt
            )
    return call_llm(client, model, system_prompt, user_prompt, memory_messages=memory_messages)


def generate_diagram(
    environment: str,
    vendor_or_platform: str,
    description: str,
    client: Groq,
    model: str,
    docstore: Optional[DocumentStore] = None,
    top_k: int = 3,
    memory_messages: Optional[List[dict]] = None,
) -> str:

    target = (
        f"{vendor_or_platform.title()} server" if environment == "server" else f"{vendor_or_platform.title()} network"
    )
    system_prompt = (
    "You are a senior infrastructure architect. "
    "Generate a high-level system architecture diagram. "
    "Generate a Mermaid diagram ONLY. "
    "Output must be valid Mermaid syntax and nothing else (no markdown, no explanations). "
    "Use 'flowchart LR'. "
    "Use clear, descriptive node labels. "
    "Represent system components and their interactions (services, APIs, storage). "
    "Do NOT generate logic flowcharts, algorithms, or step-by-step processes."
)

    user_prompt = (
        f"Create a Mermaid diagram for this {target} design.\n"
        f"Description:\n{description}\n\n"
        "Constraints:\n"
        f"- Keep the diagram readable and compact\n"
        f"- Do not include commands or prose\n"
        f"- Use logical groupings (subgraphs) when helpful\n"
    )

    if docstore:
        context, sources = docstore.query(description, top_k=top_k)
        if context:
            user_prompt = (
                f"Use this additional context if relevant:\n{context}\n\nSources: {', '.join(sources)}\n\n" + user_prompt
            )

    raw = call_llm(client, model, system_prompt, user_prompt, temperature=0.1, memory_messages=memory_messages)
    return _extract_mermaid(raw)


def main() -> None:
    
    print("Agentic IT Engineer")
    print("=======================\n")
    client = get_groq_client()
    model = choose_model()

    memory_path = os.path.join(os.path.dirname(__file__), ".it_assistant_memory.json")
    memory = MemoryStore(memory_path, max_messages=40)
    # Ask whether to enable RAG
    rag_choice = input(
        "\nWould you like to enable Retrieval‑Augmented Generation (RAG) using local documents? [y/N]: "
    ).strip().lower()
    docstore: Optional[DocumentStore] = None
    if rag_choice == "y":
        docs_path = input(
            "Enter the path to your documents folder (containing .txt/.pdf files): "
        ).strip()
        try:
            print("Loading documents and building search index... this may take a moment.")
            docstore = DocumentStore(docs_path)
            print(f"Loaded {len(docstore._texts)} text chunks from your documents.")
        except Exception as e:
            print(f"Error loading documents: {e}\nProceeding without RAG.")
            docstore = None
    environment = choose_environment()
    vendor_or_platform = choose_vendor(environment)
    mode = choose_mode()

    user_input = ""
    result = ""

    if mode == "build":
        user_input = input(
            "\nDescribe the desired architecture (e.g. number of users, services, VLANs, etc.):\n"
        ).strip()
        if environment == "server":
            result = build_server(
                vendor_or_platform,
                user_input,
                client,
                model,
                docstore,
                memory_messages=memory.get_messages(),
            )
        else:
            result = build_network(
                vendor_or_platform,
                user_input,
                client,
                model,
                docstore,
                memory_messages=memory.get_messages(),
            )
        print("\n=== Agent Response ===\n")
        print(result)

    elif mode == "troubleshoot":
        user_input = input(
            "\nDescribe the issue you are facing (symptoms, metrics, error messages, etc.):\n"
        ).strip()
        if environment == "server":
            result = troubleshoot_server(
                vendor_or_platform,
                user_input,
                client,
                model,
                docstore,
                memory_messages=memory.get_messages(),
            )
        else:
            result = troubleshoot_network(
                vendor_or_platform,
                user_input,
                client,
                model,
                docstore,
                memory_messages=memory.get_messages(),
            )
        print("\n=== Agent Response ===\n")
        print(result)

    else:
        user_input = input(
            "\nDescribe what you want diagrammed (components, links, VLANs, services, etc.):\n"
        ).strip()
        mermaid = generate_diagram(
            environment,
            vendor_or_platform,
            user_input,
            client,
            model,
            docstore,
            memory_messages=memory.get_messages(),
        )
        out_path = _save_mermaid(mermaid)
        result = f"Mermaid diagram saved to: {out_path}\n\n{mermaid}"
        print("\n=== Diagram (Mermaid) ===\n")
        print(mermaid)
        print(f"\nSaved to: {out_path}")

    memory.add_turn(user_input, result)


if __name__ == "__main__":
    main()