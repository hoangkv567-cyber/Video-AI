# Hướng dẫn Vận hành & Runbook (Operations & Runbook)

Tài liệu này cung cấp quy trình vận hành chi tiết, bảo trì định kỳ, xử lý sự cố và quản lý xuất bản đa kênh cho hệ thống **Video AI**.

---

## 1. Kiến trúc & Vòng đời Dữ liệu

### 1.1. Luồng xử lý chính
```text
Brief → Khám phá chủ đề (Grounding/Sources) → VideoPlan v1 → Duyệt kịch bản
→ Sinh Keyframe/Motion (hoặc Veo) → TTS VI/EN → FFmpeg Master Render → QC Gate
→ Duyệt video cuối → Scheduler / Publisher Adapters → YouTube/Facebook/TikTok/Zalo
```

### 1.2. Chính sách Lưu trữ & Vòng đời Asset (Retention Policy)
- **Asset trung gian / Raw** (`keyframe`, `clip`, `voice`, `styleboard`): Lưu trữ **7 ngày**.
- **Asset thành phẩm / Final** (`master`, `derivative`, `thumbnail`, `caption`, `srt`): Lưu trữ **30 ngày**.
- **Asset được ghim (`pinned = True`)**: **Không bao giờ bị xóa tự động**, chỉ bị xóa khi quản trị viên thao tác trực tiếp.

---

## 2. Các lệnh Bảo trì & Cron Job Định kỳ

### 2.1. Dọn dẹp Asset Quá hạn (Prune Expired Assets)
Khuyến nghị chạy mỗi đêm lúc 02:00 (UTC+7):
```bash
# Chạy thử nghiệm (Dry-run) kiểm tra số lượng và dung lượng
python -m app.cli prune-assets --dry-run

# Chạy dọn dẹp thực tế
python -m app.cli prune-assets --days-raw 7 --days-final 30
```

### 2.2. Kiểm tra Dung lượng Ổ đĩa (Disk Guard)
Khuyến nghị cấu hình cron chạy mỗi 30 phút hoặc 1 giờ:
```bash
# Kiểm tra phân vùng lưu trữ asset
python -m app.cli disk-guard --min-free-gb 5.0 --min-free-percent 10.0
```
- Nếu dung lượng trống < 5.0 GB hoặc < 10.0%, lệnh trả về exit code `1` kèm thông báo cảnh báo `WARNING: Disk space low` để hệ thống giám sát (Zabbix/Prometheus/Alertmanager) kích hoạt thông báo.

### 2.3. Sao lưu Cơ sở Dữ liệu có Mã hóa (Nightly Encrypted Backup)
Khuyến nghị chạy mỗi đêm lúc 03:00 (UTC+7):
```bash
# Sao lưu tự động vào thư mục backups
python -m app.cli backup-db --output ./backups/db_backup_$(date +%Y%m%d_%H%M%S).enc
```
- Bản sao lưu được mã hóa bằng thuật toán Fernet (AES-128-CBC + HMAC SHA-256) thông qua biến môi trường `TOKEN_ENCRYPTION_KEY` hoặc `BACKUP_ENCRYPTION_KEY`.

### 2.4. Phục hồi Dữ liệu từ Bản sao lưu (Disaster Recovery & Restore)
Khi cần khôi phục dữ liệu sau sự cố:
```bash
# 1. Dừng các service web và worker
docker compose stop web worker-ai worker-render worker-publish scheduler

# 2. Thực hiện phục hồi database
python -m app.cli restore-db --input ./backups/db_backup_target.enc

# 3. Khởi động lại hệ thống
docker compose up -d
```

---

## 3. Quản lý Xuất bản & Ma trận Phân quyền Nền tảng

Mỗi tài khoản kết nối (`ConnectedAccount`) được gán nhãn capability độc lập:

| Nền tảng | Capability Hỗ trợ | Cơ chế Hoạt động | Fallback khi chưa cấp quyền |
|---|---|---|---|
| **YouTube Shorts** | `SCHEDULE`, `DIRECT`, `MANUAL` | OAuth offline, upload API, tự động đặt `publishAt` | Upload ở chế độ `private` hoặc xuất gói Manual Studio Bundle |
| **Facebook Reels** | `SCHEDULE`, `DIRECT`, `MANUAL` | Meta Page Reels API, đặt lịch trực tiếp | Standard Access trên Test Page hoặc xuất Manual Bundle |
| **TikTok** | `DRAFT`, `MANUAL` | Upload-to-Inbox draft khi có **user consent token** | Xuất Manual Bundle (Direct Post bị tắt có chủ đích để bảo vệ chính sách tài khoản) |
| **Zalo OA** | `DIRECT`, `MANUAL` | OA Video Upload & Article Create/Verify | Xuất Manual Bundle kèm Deep Link vào Zalo OA Manager |

### Trạng thái Kênh (Channel Labeling):
- `LIVE`: Kênh đã qua kiểm duyệt/App Review, quyền publish trực tiếp hoạt động ổn định.
- `PRIVATE_TEST`: Đăng tải thành công ở chế độ riêng tư (Private/Draft/Test Page) phục vụ nghiệm thu.
- `MANUAL`: Kênh chưa có API permission; tự động sinh gói bundle đầy đủ (MP4, Thumbnail, Tiêu đề, Mô tả, Hashtag, Khai báo AI) để người dùng đăng thủ công.
- `BLOCKED`: Tài khoản bị khóa, thu hồi token hoặc vi phạm chính sách cần kết nối lại.

---

## 4. Xử lý Sự cố (Troubleshooting & Incident Management)

### 4.1. Worker bị dừng đột ngột giữa chừng (Crash / Restart Recovery)
- Hệ thống hỗ trợ cơ chế **Resume Safety**: Mọi asset (keyframe, motion clip, TTS audio) đã sinh xong đều được lưu vào `AssetStore` kèm SHA-256 và storage key.
- Khi worker khởi động lại và nhận lại job, hàm `find_asset()` sẽ kiểm tra và tái sử dụng các asset đã có, chỉ sinh tiếp các scene chưa hoàn tất, tránh lãng phí chi phí AI và thời gian render.
- Khi người vận hành retry với `scene_ids`, hệ thống chỉ tạo keyframe/clip mới cho đúng cảnh được chọn. Asset cũ vẫn bất biến để audit; hai rendition dùng chung visual master sẽ bị hủy QC/approval và tạo render attempt mới.
- `motion_clip` được định danh theo SHA-256 của keyframe hiện hành, vì vậy retry visual không thể vô tình dùng lại chuyển động sinh từ keyframe cũ.

#### Đối soát lệnh sinh video trả phí không xác định

Nếu worker dừng sau khi nhà cung cấp có thể đã nhận lệnh nhưng trước khi lưu được operation ID, creative chuyển sang `NEEDS_ACTION`. Hệ thống không tự gửi lại lệnh trả phí; riêng Wan cũng chỉ gọi POST tạo task đúng một lần.

Admin đối chiếu provider console/support rồi gọi endpoint sau với `Idempotency-Key` duy nhất:

`POST /api/v1/creatives/{creative_id}/scenes/{scene_id}/video-submission/reconcile`

- Đã tìm thấy task: gửi `{"action":"attach_operation","provider":"veo|wan","operation_name":"...","reason":"..."}`. `VIDEO_PROVIDER` đang cấu hình phải trùng provider để worker poll đúng adapter.
- Provider xác nhận không tạo task: gửi `{"action":"confirm_not_submitted","provider":"veo|wan","reason":"..."}`. Hệ thống hủy intent và chỉ xóa đúng projected cost đã liên kết.
- Nếu chưa chứng minh được một trong hai kết quả, giữ nguyên `NEEDS_ACTION`; không dùng `confirm_not_submitted` để ép retry.

Endpoint khóa Creative/Scene/CostEvent, ghi audit theo admin, chuyển về `GENERATING` và tạo generate outbox job trong cùng transaction. Nếu provider preflight không đạt, toàn bộ thay đổi được rollback.

### 4.2. Khôi phục tác vụ Scheduler bị treo (Stale Claim Recovery)
- Nếu worker publish gặp sự cố trong khi đang xử lý một bài đăng đã lên lịch, trạng thái claim sẽ hết hạn sau `STALE_CLAIM_SECONDS` (10 phút).
- Vòng quét tiếp theo của scheduler sẽ tự động nhận diện stale claim và enqueue lại mà không gây trùng lặp bài đăng.
- Mỗi phase `prepare`, `upload`, `finalize` được checkpoint trước và sau thao tác remote trong `PublishAttempt`; URL/token/session được mã hóa Fernet bằng `TOKEN_ENCRYPTION_KEY`.
- Khi phục hồi, adapter phải chứng minh kết quả bằng status probe. Nếu không chứng minh được, target chuyển sang `NEEDS_ACTION`/manual bundle thay vì blind retry. Sau tối đa bốn lần probe, hệ thống cũng fail-closed để tránh vòng lặp vô hạn.
- Lỗi tiến trình FFmpeg hoặc timeout được đánh dấu retryable để job outbox thử lại theo giới hạn; lỗi QC nội dung vẫn fail-closed và cần retry visual có chủ đích.

### 4.3. Lỗi Token hết hạn hoặc quyền bị thu hồi
- Hệ thống tự động kích hoạt `run_with_token_refresh` để lấy access token mới từ `refresh_token`.
- Nếu refresh thất bại (do người dùng đổi mật khẩu hoặc hủy kết nối trên nền tảng), trạng thái chuyển sang `NEEDS_ACTION` và ghi nhận vào bảng `audit_events` để người vận hành cập nhật lại kết nối trên Dashboard.

### 4.4. Khóa chống worker chạy trùng

- PostgreSQL session advisory lock chặn hai worker hợp tác xử lý cùng job/creative/rendition/publish target; CI kiểm tra contention trên PostgreSQL thật.
- Nếu backend giữ lock bị terminate/failover, lock được PostgreSQL giải phóng và worker cũ báo `ExecutionLockLost` khi thoát vùng bảo vệ. Dựa vào checkpoint/idempotency để lần chạy sau đối soát, không giả định exactly-once.
- Không đặt kết nối giữ session advisory lock sau PgBouncer ở transaction-pooling mode. Advisory lock không phải fencing token; thao tác remote vẫn phải dùng intent/checkpoint/status probe như các luồng generation và publishing ở trên.

---

## 5. Checklist Kiểm tra Sẵn sàng Trước khi Golive (Production Checklist)

1. [ ] Cấu hình `.env.production` đầy đủ `SECRET_KEY`, `TOKEN_ENCRYPTION_KEY`, `DATABASE_URL`, `REDIS_URL`.
2. [ ] Kiểm tra `/healthz` và `/readyz` phản hồi HTTP 200 OK.
3. [ ] Tạo tài khoản Admin đầu tiên thông qua `python -m app.cli bootstrap-admin --email <admin_email>`.
4. [ ] Kiểm tra cấu hình Caddy SSL reverse proxy và domain HTTPS.
5. [ ] Thiết lập cron job định kỳ cho `disk-guard`, `prune-assets`, và `backup-db`.
6. [ ] Chạy bộ kiểm thử tự động toàn diện: `pytest`.
