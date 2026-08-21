# Kế hoạch 7 tuần xây dựng MVP AI sinh video tự động

## 1. Mục tiêu và kiến trúc chốt

Xây một hệ thống nội bộ chạy trên Linux VPS, cho phép:

- Nhập brief hoặc tìm 3–5 chủ đề AI/công nghệ mới nổi.
- Sinh một visual master dọc khoảng 38,8 giây.
- Tạo hai bản riêng tiếng Việt và tiếng Anh dùng chung hình ảnh.
- Xem trước, sửa kịch bản/metadata, duyệt hoặc cho chạy tự động theo cấu hình.
- Đăng hoặc bàn giao lên YouTube Shorts, Facebook Page Reels, TikTok và Zalo OA.
- Theo dõi chi phí, trạng thái, lỗi và ID bài đăng theo từng nền tảng.

Luồng chính:

```text
Brief → Chủ đề có nguồn → VideoPlan JSON → Keyframe
→ 5 clip Veo → TTS VI/EN → FFmpeg → QC
→ Duyệt/tự động → Lên lịch → Publisher adapters
```

### Kiến trúc kỹ thuật

- Modular monolith bằng Python 3.12:
  - FastAPI cho REST API, OAuth callback và webhook.
  - Jinja2 + HTMX cho dashboard, không xây SPA riêng.
  - Pydantic, SQLAlchemy và Alembic cho schema/database.
  - PostgreSQL là nguồn trạng thái duy nhất.
  - Celery + Redis với ba queue: `ai`, `render`, `publish`.
  - FFmpeg subprocess cho toàn bộ dựng video; không dùng MoviePy trong đường production.
  - MinIO lưu asset, checksum và tạo URL HTTPS có thời hạn.
  - Caddy cung cấp HTTPS và reverse proxy.
- Docker Compose gồm `web`, `worker-ai`, `worker-render`, `worker-publish`, `scheduler`, PostgreSQL, Redis, MinIO và Caddy.
- VPS tối thiểu: 4 vCPU, 8 GB RAM, 100 GB SSD; không cần GPU.
- Scheduler chạy mỗi phút, lấy job đến hạn bằng khóa hàng PostgreSQL; mọi thời gian lưu UTC và hiển thị theo `Asia/Ho_Chi_Minh`.
- Antigravity IDE/CLI chỉ dùng để hỗ trợ scaffold, viết test và kiểm tra giao diện; không trở thành dependency runtime của hệ thống. [Antigravity IDE](https://antigravity.google/docs/ide/overview/)

### Mô hình AI mặc định

- Nghiên cứu và kịch bản: `gemini-3.7-flash` qua `google-genai` và Interactions API.
- Tìm chủ đề: Gemini Grounding with Google Search; không coi đây là dữ liệu xếp hạng trend chính thức. Google Trends API vẫn là alpha nên chỉ để adapter tương lai. [Google Search Grounding](https://ai.google.dev/gemini-api/docs/google-search), [Google Trends API](https://developers.google.com/search/apis/trends)
- Keyframe và thumbnail: `gemini-3.1-flash-image`—không dùng Imagen vì Imagen 4 đã đến hạn ngừng hoạt động ngày 17/08/2026. [Image generation](https://ai.google.dev/gemini-api/docs/image-generation), [Model lifecycle](https://ai.google.dev/gemini-api/docs/deprecations)
- Video:
  - Sinh 5 cảnh × 8 giây bằng `veo-3.1-lite-generate-preview`, 720p, 9:16.
  - Cảnh lỗi hoặc cảnh hero được nâng lên `veo-3.1-fast-generate-preview`.
  - Ghép bốn crossfade 0,3 giây để tạo video khoảng 38,8 giây.
  - Veo Lite hiện có giá tham chiếu 0,05 USD/giây, nên visual master cơ sở khoảng 2 USD; tổng AI cost bị chặn cứng ở 6 USD. [Veo 3.1 và pricing](https://ai.google.dev/gemini-api/docs/pricing)
- TTS production: Google Cloud Text-to-Speech, mặc định `vi-VN-Neural2-A` và `en-US-Neural2-F`; dùng SSML marks để lấy timestamp caption. [Voices](https://cloud.google.com/text-to-speech/docs/voices), [SSML timepoints](https://docs.cloud.google.com/text-to-speech/docs/ssml)
- Âm thanh native của Veo bị loại bỏ; master hình ảnh không có thoại, chữ, logo hay lip-sync. Voice-over, caption và metadata được ghép riêng cho từng ngôn ngữ.

### Chế độ miễn phí (mặc định từ 21/08/2026)

Ưu tiên API free-tier; các adapter trả phí (Veo, Google Cloud TTS) vẫn giữ trong code và bật lại qua cấu hình:

- Nghiên cứu và kịch bản: Gemini Flash free tier (giới hạn TPM/RPD); Groq `llama-3.3-70b-versatile` là provider thay thế khi hết quota.
- Keyframe/thumbnail: Gemini image — kiểm tra hạn mức free tier lúc probe; nếu hết quota thì xếp hàng chờ ngày kế tiếp thay vì chuyển sang trả phí.
- Video: **Veo không có free tier**, nên `VIDEO_PROVIDER=keyframe_motion` — dựng chuyển động từ keyframe bằng FFmpeg (Ken Burns zoompan/parallax), giữ nguyên cấu trúc 5 cảnh × 8 giây và crossfade 0,3 giây. Veo là tùy chọn trả phí bật lại sau.
- TTS tiếng Anh: Groq Orpheus (`canopylabs/orpheus-v1-english`, free dev tier, giới hạn 200 ký tự/request nên phải chunk theo câu).
- TTS tiếng Việt: Groq **không hỗ trợ tiếng Việt**, dùng `edge-tts` (`vi-VN-HoaiMyNeural`, miễn phí, không cần key; endpoint không chính thức nên phải có provider interface để thay bằng Google Cloud TTS free tier 1 triệu ký tự/tháng khi cần độ ổn định).
- Timestamp caption: không dựa vào SSML marks nữa — audio sinh xong được đưa qua Groq Whisper (`whisper-large-v3`, free tier) lấy word timestamps cho cả VI/EN.
- Cost ledger vẫn ghi mọi lượt gọi với `amount_usd=0` kèm đếm quota (RPM/RPD) để biết khi nào chạm trần free tier; hard cap 6 USD chỉ áp dụng khi bật adapter trả phí.

## 2. Luồng nghiệp vụ, dữ liệu và giao diện

### Nghiên cứu và viết kịch bản

- Người vận hành nhập brief/category; hệ thống tìm trong 72 giờ gần nhất, mở rộng thành 7 ngày nếu thiếu dữ liệu.
- Chỉ giữ chủ đề có ít nhất hai nguồn độc lập, ưu tiên ít nhất một nguồn chính thức.
- Chấm điểm theo độ mới 35%, độ xác thực chéo 25%, phù hợp khán giả 25% và tiềm năng hình ảnh 15%.
- Lưu URL, tiêu đề nguồn, thời điểm truy cập và citation; người dùng chọn một chủ đề.
- Chạy Gemini lần hai, không bật công cụ tìm kiếm, để tạo structured output; cách tách hai lượt tránh phụ thuộc tổ hợp Search + Structured Output đang Preview.
- Mọi factual claim phải ánh xạ tới `source_id`; thiếu nguồn sẽ chặn chế độ auto.

`VideoPlan v1` gồm:

- `schema_version`, topic, angle, source IDs, style guide và chi phí dự kiến.
- Chính xác 5 scene, mỗi scene 8 giây.
- Mỗi scene có `keyframe_prompt_en`, `visual_prompt_en`, negative prompt, continuity note và fact IDs.
- `narration`, `on_screen_text`, title, description và hashtags riêng cho `vi` và `en`.
- Các trường disclosure, made-for-kids và rủi ro nội dung.
- Semantic validator kiểm tra tổng thời lượng, số scene, nguồn, độ dài voice-over và giới hạn metadata.

Nếu TTS của một scene vượt 7,6 giây trên 5%, hệ thống yêu cầu Gemini rút gọn câu. Không time-stretch ngoài khoảng 0,95–1,05.

### Sinh và dựng video

- Sinh style board và keyframe nhất quán trước, sau đó image-to-video từng scene.
- Worker tải output Veo về MinIO ngay khi hoàn tất vì file phía Google chỉ được giữ khoảng hai ngày.
- Mỗi asset bất biến, có SHA-256, model ID, prompt hash, chi phí và kết quả `ffprobe`.
- Chỉ render lại scene lỗi; không tự động render lại toàn bộ video.
- FFmpeg tạo master và derivative theo từng `PlatformProfile`:
  - 1080×1920, 30 fps.
  - H.264 High, `yuv420p`, AAC 48 kHz, `faststart`.
  - Loudness −14 LUFS, true peak tối đa −1,5 dBTP.
  - Caption Noto Sans burn-in trong safe zone và kèm file SRT.
- Tạo thumbnail VI/EN riêng.
- Gemini multimodal kiểm tra độ khớp nội dung; kiểm tra kỹ thuật vẫn dựa trên FFmpeg/FFprobe.
- Không dùng nhạc hoặc video lấy từ nền tảng khác. MVP chỉ dùng voice-over, âm thanh được cấp phép và asset AI sinh mới.

### Trạng thái và khả năng phục hồi

```text
DRAFT → RESEARCHED → SCRIPT_READY → SCRIPT_APPROVED
→ GENERATING → QC_REQUIRED → READY → FINAL_APPROVED
→ SCHEDULED → PUBLISHING
→ PUBLISHED | PARTIAL | NEEDS_ACTION | FAILED
```

- Manual mode yêu cầu duyệt kịch bản và video cuối.
- Auto mode vẫn đi qua schema, nguồn, policy, QC và cost gates; hệ thống ghi audit event thay cho thao tác duyệt.
- TikTok luôn cần thao tác/consent của người dùng trong MVP nội bộ.
- Mỗi nền tảng là một publish target độc lập; lỗi một kênh không rollback kênh đã thành công.
- Trước khi retry sau timeout, adapter phải truy vấn trạng thái remote; không blind retry để tránh đăng trùng.
- Chỉ retry lỗi 408, 429 và 5xx bằng exponential backoff + jitter. Lỗi quyền, policy hoặc validation chuyển sang `NEEDS_ACTION`.

### REST API và dữ liệu chính

Các endpoint nội bộ:

- `POST /api/v1/topics/discover`
- `POST /api/v1/creatives`
- `POST /api/v1/creatives/{id}/generate`
- `PATCH /api/v1/scripts/{id}`
- `POST /api/v1/renditions/{id}/approve`
- `POST /api/v1/publications`
- `GET /api/v1/jobs/{id}`
- `/oauth/{platform}/start|callback`
- `/webhooks/{provider}`

Các lệnh generate/publish nhận `Idempotency-Key`, trả `202 Accepted` cùng `job_id`. Lỗi dùng envelope thống nhất gồm `code`, `message`, `retryable`, `details` và `correlation_id`.

Các bảng chính: user, campaign, research/source, script version, scene, asset, rendition, cost event, connected account, job, publish target/attempt và audit event.

Publisher contract thống nhất: `capabilities`, `validate`, `prepare`, `upload`, `finalize`, `poll_status`, `refresh_credentials`.

### Dashboard

- Đăng nhập và quản lý kết nối kênh.
- Form brief và danh sách chủ đề có nguồn.
- Editor cho kịch bản, scene, voice-over VI/EN.
- Progress theo từng scene, chi phí dự kiến/thực tế và nút retry scene.
- Preview hai rendition, sửa metadata và phê duyệt.
- Chọn kênh, quyền riêng tư, giờ đăng và chế độ manual/auto.
- Trang trạng thái từng nền tảng, lỗi, remote post ID và gói đăng thủ công.

## 3. Phạm vi từng nền tảng

| Nền tảng | Tích hợp MVP | Lên lịch | Khi chưa có quyền |
|---|---|---|---|
| YouTube Shorts | OAuth offline với `youtube.upload`, `videos.insert`, khai báo synthetic media | Dùng `publishAt` | Upload private và cung cấp bundle/manual Studio; project chưa audit không được mặc định public. [YouTube upload](https://developers.google.com/youtube/v3/docs/videos/insert) |
| Facebook Reels | Facebook Page, các quyền `pages_show_list`, `pages_read_engagement`, `pages_manage_posts`; START → upload → FINISH | Dùng lịch native nếu thời gian hợp lệ | Standard Access với Page test hoặc bundle; Page ngoài app roles phụ thuộc App Review. [Reels Publishing API](https://developers.facebook.com/docs/video-api/guides/reels-publishing/) |
| TikTok | Mặc định xuất bundle; bật Upload-to-Inbox nếu được cấp `video.upload` | Người dùng xác nhận rồi hệ thống upload draft | Không triển khai Direct Post im lặng. Công cụ chỉ phục vụ tài khoản nội bộ là use case có nguy cơ không qua audit. [TikTok guidelines](https://developers.tiktok.com/docs/en/content-sharing-guidelines) |
| Zalo OA | Nội dung dạng video/bài viết OA nếu OA trả phí và được cấp quyền tạo nội dung | Scheduler nội bộ gọi API lúc đến hạn | `NEEDS_ACTION` với MP4, caption, thumbnail và deep link OA Manager. Đây không phải consumer Zalo Video. [Zalo OA OpenAPI](https://oa.zalo.me/home/function/extension) |

Mỗi connection lưu capability thực tế: `DIRECT`, `SCHEDULE`, `DRAFT`, `MANUAL` hoặc `BLOCKED`. Auto mode chỉ được bật khi capability probe thành công.

## 4. Lộ trình triển khai 7 tuần

| Tuần | Công việc | Exit gate |
|---|---|---|
| 1 — Nền tảng và quyền truy cập | Khởi tạo Git, `pyproject.toml`, dependency lock, lint/type-check/test/CI; Docker Compose; domain/TLS; spike Gemini Search, JSON, Nano Banana, một clip Veo và TTS VI/EN. Tạo Google/Meta/TikTok/Zalo app, OA và checklist OAuth; chuẩn bị privacy policy, terms, data-deletion page và hồ sơ audit. Cài Docker cho máy Windows; FFmpeg nằm trong container. | `/healthz` chạy trên VPS; CI xanh; một clip được tải và probe thành công; capability matrix và ước tính chi phí được xác nhận. |
| 2 — Research và scripting | Xây schema/database, state machine, dashboard skeleton, topic discovery, citation store, `VideoPlan v1`, versioning và script approval. Thêm semantic validator và policy gate. | Một brief tạo được 3–5 topic và một plan 5 scene với narration VI/EN; JSON sai, thiếu nguồn hoặc double-click đều bị xử lý đúng. |
| 3 — Media generation | Xây MinIO asset store, Celery queues, Nano Banana adapter, Veo LRO polling, tải file, checksum, TTS/SSML, cost ledger và hard cap. | Có 5 clip dọc và hai voice track; restart worker không gọi Veo lại; chi phí projected/actual truy vết được. |
| 4 — FFmpeg và QC | Normalize clip, strip/mix audio, crossfade, caption, thumbnail, platform derivatives, ffprobe/QC và preview UI. | Sinh được hai MP4 khoảng 38,8 giây dùng cùng visual checksums; codec, loudness, caption và safe zone đạt test. |
| 5 — YouTube và Facebook | OAuth/token encryption; YouTube uploader/status; Facebook Page Reels uploader/scheduler; manual bundle dùng chung cho mọi platform. Tạo screencast và nộp các hồ sơ review ngay khi flow hoạt động. | Live smoke ở chế độ private/test trên account được cấp quyền; mock/contract test đầy đủ khi review chưa xong. |
| 6 — TikTok, Zalo và scheduling | TikTok manual/inbox flow, Zalo OA spike, scheduler fan-out, consent/audit trail, token refresh, polling và partial failure. | Lịch vẫn chạy sau restart; token hết hạn và lỗi từng kênh không tạo bài trùng; TikTok/Zalo luôn có fallback hoàn chỉnh. |
| 7 — Ổn định và bàn giao | Failure injection, bảo mật, disk guard, retention, backup/restore, pilot ba brief, tài liệu vận hành, API docs và demo. | Ba brief tạo sáu rendition; kill/restart từng stage đều phục hồi; restore backup thành công; từng kênh được gắn nhãn `LIVE`, `PRIVATE_TEST`, `MANUAL` hoặc `BLOCKED`. |

Vì chỉ có một lập trình viên, thứ tự ưu tiên cố định khi trễ tiến độ là:

1. Pipeline Gemini/Veo/TTS/FFmpeg và dashboard end-to-end.
2. YouTube và Facebook chạy thật ở mức quyền được cấp.
3. TikTok Upload-to-Inbox và Zalo OA nếu quyền sẵn sàng.
4. Bundle thủ công là deliverable bắt buộc cho mọi kênh; Instagram và analytics nằm sau MVP.

## 5. Kiểm thử, nghiệm thu và giả định

### Kiểm thử bắt buộc

- Unit: JSON/semantic schema, state transition, cost cap, caption escaping, metadata platform, encryption và role checks.
- Contract: fixture đã loại secret cho Gemini, Veo, TTS và từng publisher; mô phỏng 429, 5xx, timeout, policy block và token rotation.
- Media: kiểm tra duration, 9:16, codec, fps, audio, loudness, black/freeze frame, subtitle safe zone và visual checksum dùng chung.
- Resilience: kill worker sau khi submit Veo, sau upload và trước finalize; restart không sinh lại clip hoặc đăng trùng.
- Security: OAuth state/PKCE, CSRF, signed URL hết hạn, token không xuất hiện trong log/repo và kiểm thử quyền admin/editor/publisher.
- E2E: brief → hai rendition → manual approval/auto gate → schedule → publish/fallback → remote status.

### Tiêu chí nghiệm thu MVP

- Một brief AI/công nghệ tạo hai MP4 VI/EN khoảng 38,8 giây, 1080×1920, có TTS, caption và metadata riêng.
- Hai bản dùng chung visual master; không sinh lại clip chỉ vì đổi ngôn ngữ.
- Tổng chi phí AI biến đổi không vượt 6 USD nếu không có admin override; override phải có lý do trong audit log.
- Mọi factual claim có nguồn; nội dung thiếu nguồn hoặc vi phạm policy không được auto-publish.
- Manual và auto mode hoạt động trên kênh có capability; TikTok giữ consent/manual.
- Retry, webhook lặp và restart không tạo video hoặc bài đăng trùng.
- Mỗi nền tảng chưa được duyệt vẫn tạo bundle gồm derivative MP4, thumbnail, title, caption, hashtags, disclosure và hướng dẫn hành động.
- Việc bên thứ ba phê duyệt app không phải điều kiện hoàn thành code; nghiệm thu bằng live smoke ở mức quyền hiện có cộng contract test và fallback.

### Giả định và giới hạn

- Workspace hiện trống, chưa có Git, Docker, FFmpeg hoặc credential; toàn bộ bootstrap nằm trong tuần 1.
- Một lập trình viên làm toàn thời gian; MVP phục vụ một đội nội bộ, không có public signup hay multi-tenancy.
- Admin tạo tài khoản nội bộ với vai trò admin, editor hoặc publisher; chưa tích hợp SSO.
- Generation chỉ chạy theo yêu cầu; scheduler dùng cho giờ xuất bản, không tự tạo video định kỳ.
- Mặc định xử lý một visual master tại một thời điểm, tối đa mục tiêu 10 master/ngày.
- Raw/intermediate asset giữ 7 ngày, final rendition 30 ngày; có thể pin thủ công. PostgreSQL và MinIO được backup mã hóa ra storage ngoài VPS mỗi đêm.
- Không voice cloning, avatar talking-head, nhạc trend, browser automation, scraping social platform, analytics tối ưu nội dung hoặc Kubernetes trong MVP.
- Model ID và bảng giá đặt trong cấu hình có version, được probe lúc khởi động vì Veo/TTS Preview và quota có thể thay đổi.
- Chi phí 6 USD không gồm VPS, phí Zalo OA, app review hoặc object-storage backup.
