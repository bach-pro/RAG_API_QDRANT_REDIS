# Multi-source RAG API

FastAPI RAG cho tài liệu PDF, DOCX và document JSON. Dịch vụ dùng Qdrant cho vector search, Redis cho lịch sử hội thoại, BM25 cho tìm kiếm từ khóa và Ollama hoặc Google Gemini để sinh câu trả lời.

## Kiến trúc

```text
Client / chat-api
        │ POST /v1/chat hoặc /v1/chat/stream
        ▼
      FastAPI
        │
        ├── Router / rewrite / HyDE ──> Ollama router model
        ├── Embedding ───────────────> Ollama embedding model
        ├── Retrieval ───────────────> Qdrant + BM25 + MMR/RRF
        ├── History ─────────────────> Redis
        └── Answer stream ───────────> Ollama hoặc Google Gemini
```

`/v1/chat/stream` trả Server-Sent Events (SSE). Khi HTTP client ngắt trong lúc model đang stream, task FastAPI bị hủy, exception hủy được lan truyền xuống async provider stream và connection Google/Ollama được đóng trong `finally`. Phần router, embedding và retrieval chạy trước khi answer stream bắt đầu nên không thuộc phạm vi hủy này.

## Yêu cầu

- Python 3.11 hoặc Docker Desktop
- `uv` khi chạy local
- Ollama khi dùng provider/model local
- Google API key khi đặt `RAG_ANSWER_PROVIDER=google`

## Chạy nhanh bằng Docker Compose

```powershell
cd service/RAG_API_QDRANT_REDIS
Copy-Item .env.example .env
docker compose --profile cpu up -d --build
```

Các service CPU mặc định:

| Service | URL / port mặc định |
| --- | --- |
| FastAPI | `http://localhost:9000` |
| Swagger | `http://localhost:9000/docs` |
| Qdrant | `http://localhost:6333` |
| Redis | `localhost:6379` |
| Ollama | `http://localhost:11434` |

Chạy profile GPU nếu máy có NVIDIA Container Toolkit:

```powershell
docker compose --profile gpu0 up -d --build
# Hoặc GPU 1:
docker compose --profile gpu1 up -d --build
```

Các port API GPU mặc định là `9002` (GPU 0) và `9003` (GPU 1). Không chạy nhiều profile cùng lúc nếu các port trong `.env` bị trùng.

Để dừng stack nhưng giữ dữ liệu Qdrant/Redis:

```powershell
docker compose --profile cpu down
```

`docker compose down -v` sẽ xóa volumes Qdrant và Redis; chỉ dùng khi muốn xóa toàn bộ index và history.

## Chạy local

```powershell
cd service/RAG_API_QDRANT_REDIS
Copy-Item .env.example .env
uv sync --all-groups
uv run uvicorn rag_api.main:app --host 127.0.0.1 --port 9000 --reload
```

Khi chạy local, đặt `RAG_QDRANT_URL`, `RAG_REDIS_URL`, `OLLAMA_HOST` và `RAG_LOCAL_OLLAMA_HOST` trỏ tới service tương ứng, ví dụ `localhost`.

Giao diện Gradio là tùy chọn và chạy ngoài Docker:

```powershell
uv run python gradio_app.py
```

Mở `http://localhost:7860`.

## Cấu hình quan trọng

Sao chép `.env.example` rồi chỉ điền secret trong `.env`; không commit file này.

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `RAG_ANSWER_PROVIDER` | `ollama` | Provider answer cuối: `ollama` hoặc `google`. |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama dùng cho answer khi provider là Ollama. |
| `OLLAMA_MODEL` | `qwen3:4b-instruct` | Model sinh answer. |
| `GOOGLE_API_KEY` | rỗng | Bắt buộc khi dùng Google. |
| `GOOGLE_MODEL` | `gemini-3.5-flash` | Model Google dùng cho answer. |
| `RAG_LOCAL_OLLAMA_HOST` | `http://localhost:11434` | Ollama cho embedding. |
| `RAG_EMBEDDING_MODEL` | `embeddinggemma:latest` | Model embedding. |
| `RAG_ROUTER_OLLAMA_HOST` | theo embedding host | Endpoint router/rewrite/HyDE/decomposition. |
| `RAG_ROUTER_MODEL` | `qwen3:1.7b` | Model router. |
| `RAG_QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint. |
| `RAG_REDIS_URL` | `redis://localhost:6379/0` | Redis history endpoint. |
| `RAG_CHAT_MEMORY_TTL_SECONDS` | `3600` | TTL history của mỗi conversation. |
| `RAG_TOP_K` / `RAG_FETCH_K` | `5` / `20` | Số kết quả cuối / số ứng viên retrieval. |

Nếu `OLLAMA_HOST` hoặc `RAG_ROUTER_OLLAMA_HOST` là `https://ollama.com`, cấu hình API key tương ứng. Khi answer dùng Google, Ollama vẫn có thể cần cho embedding và router.

## API

| Method | Endpoint | Mô tả |
| --- | --- | --- |
| `GET` | `/health` | Trạng thái service, Qdrant và Redis. |
| `POST` | `/v1/ingest/documents` | Ingest document JSON. |
| `POST` | `/v1/ingest/files` | Upload PDF/DOCX. |
| `GET` | `/v1/ingest/{job_id}` | Xem ingest job. |
| `POST` | `/v1/chat` | Trả câu trả lời JSON. |
| `POST` | `/v1/chat/stream` | Trả SSE: `metadata`, `token`, `done`, hoặc `error`. |
| `DELETE` | `/v1/chat/{conversation_id}/memory` | Xóa history Redis. |

### Ingest document JSON

```powershell
curl.exe -X POST http://localhost:9000/v1/ingest/documents `
  -H "Content-Type: application/json" `
  -d '{"documents":[{"id":"policy-001","title":"Quy định phép","text":"Nhân viên có 12 ngày phép năm.","source_type":"policy"}]}'
```

### Upload PDF/DOCX

```powershell
curl.exe -X POST http://localhost:9000/v1/ingest/files `
  -F "files=@C:\docs\policy.pdf"
```

Ingest chạy nền và trả `job_id`. Dùng `GET /v1/ingest/{job_id}` để poll trạng thái. Mỗi thời điểm chỉ có một ingest job active.

### Chat JSON

```powershell
curl.exe -X POST http://localhost:9000/v1/chat `
  -H "Content-Type: application/json" `
  -d '{"question":"Tôi có bao nhiêu ngày phép?","mode":"Auto Router","top_k":5,"fetch_k":20,"mmr_lambda":0.5,"conversation_id":"demo"}'
```

### Chat stream và hủy request

```powershell
curl.exe -N -X POST http://localhost:9000/v1/chat/stream `
  -H "Content-Type: application/json" `
  -d '{"question":"Hãy trả lời chi tiết.","mode":"Hybrid RRF"}'
```

Nhấn `Ctrl+C` sau khi nhận token đầu tiên để đóng SSE client. Response sẽ kết thúc mà không có event `done`; log API ghi nhận task stream bị hủy. Input token và output token đã sinh trước thời điểm hủy vẫn có thể bị provider tính phí.

## Retrieval modes

| Mode | Khi dùng |
| --- | --- |
| `Auto Router` | Rule và router model chọn mode. |
| `BM25` | Mã, ID, field, tên API hoặc exact phrase. |
| `Hybrid RRF` | Kết hợp semantic và BM25; lựa chọn an toàn mặc định. |
| `Semantic` | Câu hỏi tự nhiên và paraphrase. |
| `MMR` | Cần kết quả đa dạng. |
| `Decomposition` | Câu hỏi có nhiều ý. |
| `HyDE` | Câu hỏi khái niệm/ít keyword. |

## Kiểm thử

```powershell
uv run python -m pytest -q
uv run python -m compileall -q src gradio_app.py
```

## Bảo mật và vận hành

- FastAPI RAG hiện không tự xác thực người dùng. Đặt service trong private network và để `chat-api` là lớp xác thực/proxy công khai.
- Không gửi `GOOGLE_API_KEY`, Ollama key hoặc token OMS xuống browser.
- Theo dõi log `rag_api` cho lỗi provider và dòng `chat stream task cancelled by client disconnect` khi client ngắt stream.
- Kiểm tra health sau deploy: `curl.exe http://localhost:9000/health`.
