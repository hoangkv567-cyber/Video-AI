# Triển khai production

Tài liệu này dành cho một máy chủ Linux chạy Docker Compose. Production dùng
file override `compose.production.yaml`; không chạy chỉ với `compose.yaml` vì
file cơ sở giữ các mặc định thuận tiện cho môi trường local.

## 1. Chuẩn bị cấu hình

Trỏ DNS của domain vào máy chủ, mở TCP 80/443, sau đó tạo file cấu hình riêng:

```bash
cp .env.production.example .env.production
chmod 600 .env.production
```

Thay toàn bộ giá trị `CHANGE_ME`. Có thể sinh hai khóa ứng dụng bằng Python:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Dùng kết quả thứ nhất cho `SECRET_KEY`, kết quả thứ hai cho
`TOKEN_ENCRYPTION_KEY`. Không gửi các giá trị này vào chat, issue, log hoặc Git.
Mật khẩu PostgreSQL/Redis chứa ký tự dành riêng như `@`, `:`, `/`, `%` phải được
URL-encode trong `DATABASE_URL`/`REDIS_URL`; giá trị raw tương ứng vẫn đặt ở
`POSTGRES_PASSWORD`/`REDIS_PASSWORD`.

`GEMINI_API_KEY` cần cho research, script và keyframe. `GROQ_API_KEY` chỉ bắt
buộc khi chọn Groq LLM/Whisper/TTS. Orpheus chỉ được bật sau khi admin tổ chức
đã chấp nhận điều khoản model. Cấu hình mặc định dùng Edge TTS và motion từ
keyframe, nên không phát sinh phí Veo.

Với Gemini text, hệ thống dùng `GEMINI_TEXT_MODEL` trước và chỉ khi nhận HTTP 429
mới thử lần lượt các model trong `GEMINI_TEXT_FALLBACK_MODELS` theo đúng thứ tự cấu hình.

Kiểm tra cú pháp mà không in cấu hình đã nội suy ra màn hình:

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.production.yaml config --quiet
```

Không dùng `docker compose config` thiếu `--quiet` trong log CI hoặc ticket vì
output có thể chứa secret đã nội suy.

## 2. Khởi động và tạo admin đầu tiên

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.production.yaml up -d --build

docker compose --env-file .env.production \
  -f compose.yaml -f compose.production.yaml ps

curl --fail https://video.example.com/healthz
curl --fail https://video.example.com/readyz
```

Service `migrate` phải kết thúc với exit code 0 trước khi web/worker chạy. Tạo
admin bằng prompt ẩn; CLI không có tùy chọn truyền password trong command line:

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.production.yaml run --rm web \
  python -m app.cli bootstrap-admin \
  --email owner@example.com --name "Video AI Owner"
```

Lệnh có tính idempotent: nếu email đã là active admin, nó không đổi password.
Nếu email thuộc editor/publisher hoặc account bị khóa, lệnh từ chối tự nâng
quyền; nếu đã có một active admin khác, lệnh cũng từ chối tạo admin thứ hai.
Với secret manager, dùng `--password-stdin` qua pipe tin cậy; không đặt
password trực tiếp trong argv, shell history hoặc biến đã commit.

## 3. Nâng cấp và sao lưu

Trước mỗi lần nâng cấp, sao lưu PostgreSQL và volume/object MinIO. Ví dụ backup
database từ container đang chạy:

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.production.yaml exec -T postgres \
  pg_dump -U videoai -d videoai -Fc > videoai-before-upgrade.dump
```

Sau khi lấy code/image mới:

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.production.yaml up -d --build
```

Compose tự chạy `alembic upgrade head`. Không chạy `alembic downgrade base`
trên production: initial downgrade xóa toàn bộ bảng. Production override pin
release MinIO và bật Redis authentication, AOF, log rotation cùng resource
limits cơ bản; điều chỉnh giới hạn sau khi đo tải thực tế.

## 4. Baseline database legacy

Phần này chỉ áp dụng cho database cũ từng được tạo bằng
`Base.metadata.create_all()` và đã có đầy đủ schema nhưng chưa có bảng
`alembic_version`. Database mới luôn dùng `alembic upgrade head` bình thường.

1. Dừng web/worker ghi dữ liệu và tạo backup đã kiểm tra phục hồi được.
2. Kiểm tra bảng version:

   ```bash
   docker compose --env-file .env.production \
     -f compose.yaml -f compose.production.yaml exec -T postgres \
     psql -U videoai -d videoai -tAc \
     "SELECT to_regclass('public.alembic_version');"
   ```

3. Restore backup sang một database staging cô lập. Trỏ một bản sao file env
   (quyền `0600`) vào database staging đó, rồi chạy `stamp` và `check` **chỉ
   trên bản sao**. Alembic không thể chạy `check` chính xác khi chưa có revision,
   nên không dùng production làm lần thử đầu tiên:

   ```bash
   docker compose --env-file .env.baseline-check \
     -f compose.yaml -f compose.production.yaml run --rm migrate \
     alembic stamp a89e43c131e3

   docker compose --env-file .env.baseline-check \
     -f compose.yaml -f compose.production.yaml run --rm migrate \
     alembic check
   ```

4. Chỉ khi staging `alembic check` báo không có operation mới và schema legacy
   thực sự tương đương initial schema, mới baseline production rồi nâng cấp:

   ```bash
   docker compose --env-file .env.production \
     -f compose.yaml -f compose.production.yaml run --rm migrate \
     alembic stamp a89e43c131e3

   docker compose --env-file .env.production \
     -f compose.yaml -f compose.production.yaml run --rm migrate \
     alembic upgrade head
   ```

Nếu `alembic check` phát hiện khác biệt, dừng lại và viết migration reconciliation;
không dùng `stamp` để che schema thiếu cột/index/constraint. Sau baseline, chạy
`alembic current` và xác nhận revision kết thúc ở `(head)` trước khi mở lại lưu
lượng.

## 5. Kiểm tra vận hành tối thiểu

- `/healthz` chỉ xác nhận process còn sống; `/readyz` kiểm tra PostgreSQL,
  Redis, MinIO, FFmpeg và FFprobe.
- Theo dõi `docker compose ps` và log từng worker; không đưa output environment
  hoặc token vào hệ thống log.
- Kiểm tra dung lượng các volume `pgdata`, `redisdata`, `miniodata` và backup
  định kỳ ngoài máy chủ.
- OAuth callback phải khớp chính xác `PUBLIC_BASE_URL` HTTPS.
- Giữ cổng PostgreSQL, Redis, MinIO nội bộ; chỉ Caddy publish 80/443.
