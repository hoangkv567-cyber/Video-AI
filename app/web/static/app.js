"use strict";

(() => {
  const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

  function idempotencyKey(prefix) {
    const token = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
    return `${prefix}-${token}`;
  }

  function setStatus(target, message, kind = "") {
    if (!target) return;
    target.replaceChildren();
    const paragraph = document.createElement("p");
    paragraph.className = kind;
    paragraph.textContent = message;
    target.append(paragraph);
  }

  function errorMessage(payload, fallback) {
    if (payload && typeof payload.message === "string") return payload.message;
    if (payload?.detail && typeof payload.detail === "string") return payload.detail;
    return fallback;
  }

  async function api(url, options = {}) {
    const method = String(options.method ?? "GET").toUpperCase();
    const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content;
    const response = await fetch(url, {
      credentials: "same-origin",
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(!["GET", "HEAD", "OPTIONS"].includes(method) && csrfToken
          ? { "X-CSRF-Token": csrfToken }
          : {}),
        ...(options.headers ?? {}),
      },
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (response.status === 401) {
      globalThis.location.assign(`/login?next=${encodeURIComponent(globalThis.location.pathname)}`);
      throw new Error("Cần đăng nhập lại");
    }
    if (!response.ok) throw new Error(errorMessage(payload, `HTTP ${response.status}`));
    return payload;
  }

  async function pollJob(jobId, onProgress) {
    for (let attempt = 0; attempt < 180; attempt += 1) {
      const job = await api(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
      onProgress?.(job);
      const waitingForRetry =
        job.status === "FAILED" && job.error?.retryable === true && Number(job.attempts) < 4;
      if (["SUCCEEDED", "FAILED", "CANCELLED"].includes(job.status) && !waitingForRetry) {
        return job;
      }
      await sleep(2000);
    }
    throw new Error("Tác vụ chưa hoàn tất sau 6 phút; có thể mở lại trang để kiểm tra tiếp");
  }

  function bindCreativeDeletion() {
    const result = document.querySelector("#creative-delete-result");
    document.querySelectorAll('[data-action="delete-creative"]').forEach((button) => {
      button.addEventListener("click", async () => {
        const creativeId = String(button.dataset.deleteCreativeId ?? "");
        const title = String(button.dataset.deleteCreativeTitle ?? creativeId);
        const blockedMessage = String(button.dataset.deleteBlockedMessage ?? "");
        if (!creativeId) return;
        if (blockedMessage) {
          const message = `Không thể xóa creative “${title}”: ${blockedMessage}`;
          setStatus(result, message, "error");
          globalThis.alert(message);
          return;
        }
        const confirmed = globalThis.confirm(
          [
            `Xóa creative “${title}”?`,
            "Kịch bản, cảnh, video, âm thanh và chi phí local sẽ bị xóa vĩnh viễn.",
            "Bài hoặc lịch đã tạo trên nền tảng sẽ không bị xóa; hệ thống sẽ chặn nếu phát hiện dữ liệu remote.",
          ].join("\n\n"),
        );
        if (!confirmed) return;

        button.disabled = true;
        setStatus(result, `Đang xóa creative “${title}”…`);
        try {
          await api(`/api/v1/creatives/${encodeURIComponent(creativeId)}`, {
            method: "DELETE",
          });
          globalThis.location.reload();
        } catch (error) {
          setStatus(
            result,
            error instanceof Error ? error.message : "Không thể xóa creative",
            "error",
          );
          button.disabled = false;
        }
      });
    });
  }

  function renderTopics(target, discoveryJob, context) {
    target.replaceChildren();
    const topics = Array.isArray(discoveryJob.result?.topics) ? discoveryJob.result.topics : [];
    if (!topics.length) {
      setStatus(target, "Không tìm thấy chủ đề đủ hai nguồn độc lập", "error");
      return;
    }
    const heading = document.createElement("h3");
    heading.textContent = "Chọn một chủ đề";
    target.append(heading);
    const grid = document.createElement("div");
    grid.className = "topic-grid";
    topics.forEach((topic, index) => {
      const card = document.createElement("article");
      card.className = "topic-card";
      const title = document.createElement("h3");
      title.textContent = String(topic?.title ?? `Chủ đề ${index + 1}`);
      const summary = document.createElement("p");
      summary.textContent = String(topic?.summary ?? "");
      const meta = document.createElement("p");
      meta.className = "muted small";
      const sourceCount = Array.isArray(topic?.sources) ? topic.sources.length : 0;
      const score = Number.isFinite(Number(topic?.score)) ? ` · điểm ${Number(topic.score).toFixed(3)}` : "";
      meta.textContent = `${sourceCount} nguồn${score}`;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "btn btn-primary";
      button.textContent = "Chọn và sinh kịch bản";
      button.addEventListener("click", async () => {
        grid.querySelectorAll("button").forEach((item) => { item.disabled = true; });
        setStatus(target, "Đang tạo creative và xếp hàng sinh kịch bản…");
        try {
          const selection = {
            topic_index: index,
            mode: context.mode,
          };
          if (context.campaignName) selection.campaign_name = context.campaignName;
          const selected = await api(`/api/v1/discoveries/${discoveryJob.id}/select`, {
            method: "POST",
            headers: { "Idempotency-Key": idempotencyKey("select") },
            body: JSON.stringify(selection),
          });
          const scriptJob = await pollJob(selected.job_id, (job) => {
            setStatus(target, `Sinh kịch bản: ${job.status} (lần chạy ${job.attempts})`);
          });
          if (scriptJob.status !== "SUCCEEDED") {
            throw new Error(errorMessage(scriptJob.error, "Sinh kịch bản thất bại"));
          }
          setStatus(target, "Kịch bản đã sẵn sàng; đang mở creative…");
          globalThis.location.assign(`/creatives/${encodeURIComponent(selected.creative_id)}`);
        } catch (error) {
          setStatus(target, error instanceof Error ? error.message : "Không thể chọn chủ đề", "error");
        }
      });
      card.append(title, summary, meta, button);
      grid.append(card);
    });
    target.append(grid);
  }

  function bindBriefForms() {
    const discoverForm = document.querySelector("#discover-form");
    const target = document.querySelector("#topics-result");
    discoverForm?.addEventListener("submit", async (event) => {
      event.preventDefault();
      const submit = discoverForm.querySelector('button[type="submit"]');
      if (submit) submit.disabled = true;
      const data = new FormData(discoverForm);
      const context = {
        campaignName: String(data.get("campaign_name") ?? "").trim(),
        mode: String(data.get("mode") ?? "manual"),
      };
      setStatus(target, "Đang nghiên cứu chủ đề trong 72 giờ gần nhất…");
      try {
        const queued = await api("/api/v1/topics/discover", {
          method: "POST",
          headers: { "Idempotency-Key": idempotencyKey("discover") },
          body: JSON.stringify({
            brief: String(data.get("brief") ?? "").trim(),
            category: String(data.get("category") ?? "").trim(),
          }),
        });
        const job = await pollJob(queued.job_id, (current) => {
          setStatus(target, `Nghiên cứu: ${current.status} (lần chạy ${current.attempts})`);
        });
        if (job.status !== "SUCCEEDED") {
          throw new Error(errorMessage(job.error, "Nghiên cứu chủ đề thất bại"));
        }
        renderTopics(target, job, context);
      } catch (error) {
        setStatus(target, error instanceof Error ? error.message : "Không thể nghiên cứu chủ đề", "error");
      } finally {
        if (submit) submit.disabled = false;
      }
    });

    const directForm = document.querySelector("#direct-creative-form");
    const directTarget = document.querySelector("#creative-result");
    directForm?.addEventListener("submit", async (event) => {
      event.preventDefault();
      const data = new FormData(directForm);
      const body = { topic_title: String(data.get("topic_title") ?? "").trim() };
      const campaignId = String(data.get("campaign_id") ?? "").trim();
      if (campaignId) body.campaign_id = campaignId;
      try {
        const creative = await api("/api/v1/creatives", {
          method: "POST",
          headers: { "Idempotency-Key": idempotencyKey("creative") },
          body: JSON.stringify(body),
        });
        globalThis.location.assign(`/creatives/${encodeURIComponent(creative.id)}`);
      } catch (error) {
        setStatus(directTarget, error instanceof Error ? error.message : "Không thể tạo creative", "error");
      }
    });
  }

  function currentVideoPlan() {
    const node = document.querySelector("#video-plan-data");
    if (!node) return null;
    const plan = JSON.parse(node.textContent);
    for (const locale of ["vi", "en"]) {
      plan.locales[locale].narration = plan.locales[locale].narration.map((value, index) => {
        return document.querySelector(`[name="narration_${locale}_${index}"]`)?.value ?? value;
      });
      plan.locales[locale].on_screen_text = plan.locales[locale].on_screen_text.map((value, index) => {
        return document.querySelector(`[name="on_screen_${locale}_${index}"]`)?.value ?? value;
      });
    }
    return plan;
  }

  function bindCreativeActions() {
    const result = document.querySelector("#script-save-result");
    const scriptForm = document.querySelector("#script-form");
    scriptForm?.addEventListener("submit", async (event) => {
      event.preventDefault();
      try {
        const saved = await api(`/api/v1/scripts/${scriptForm.dataset.scriptId}`, {
          method: "PATCH",
          body: JSON.stringify({ video_plan: currentVideoPlan() }),
        });
        setStatus(result, `Đã lưu phiên bản kịch bản ${saved.version}; đang tải lại…`);
        globalThis.location.reload();
      } catch (error) {
        setStatus(result, error instanceof Error ? error.message : "Không thể lưu kịch bản", "error");
      }
    });

    document.querySelector("[data-action='approve-script']")?.addEventListener("click", async (event) => {
      const button = event.currentTarget;
      button.disabled = true;
      try {
        await api(`/api/v1/scripts/${button.dataset.scriptId}`, {
          method: "PATCH",
          body: JSON.stringify({ approve: true }),
        });
        globalThis.location.reload();
      } catch (error) {
        setStatus(result, error instanceof Error ? error.message : "Không thể duyệt kịch bản", "error");
        button.disabled = false;
      }
    });

    document.querySelectorAll("[data-action='generate']").forEach((button) => {
      button.addEventListener("click", async () => {
        button.disabled = true;
        const sceneId = button.dataset.sceneId;
        try {
          const queued = await api(`/api/v1/creatives/${button.dataset.creativeId}/generate`, {
            method: "POST",
            headers: { "Idempotency-Key": idempotencyKey("generate") },
            body: JSON.stringify({ scene_ids: sceneId ? [sceneId] : [] }),
          });
          setStatus(result, `Đã xếp hàng sinh video: ${queued.job_id}`);
          await pollJob(queued.job_id, (job) => {
            setStatus(result, `Sinh video: ${job.status} (lần chạy ${job.attempts})`);
          });
          globalThis.location.reload();
        } catch (error) {
          setStatus(result, error instanceof Error ? error.message : "Không thể sinh video", "error");
          button.disabled = false;
        }
      });
    });

    const renditionResult = document.querySelector("#rendition-result");
    document.querySelectorAll("[data-action='approve-rendition']").forEach((button) => {
      button.addEventListener("click", async () => {
        button.disabled = true;
        try {
          await api(`/api/v1/renditions/${button.dataset.renditionId}/approve`, {
            method: "POST",
          });
          globalThis.location.reload();
        } catch (error) {
          setStatus(
            renditionResult,
            error instanceof Error ? error.message : "Không thể duyệt rendition",
            "error",
          );
          button.disabled = false;
        }
      });
    });

    const publicationForm = document.querySelector("#publication-form");
    const publishResult = document.querySelector("#publish-result");
    publicationForm?.addEventListener("submit", async (event) => {
      event.preventDefault();
      const data = new FormData(publicationForm);
      const platforms = data.getAll("platforms").map(String);
      if (!platforms.length) {
        setStatus(publishResult, "Hãy chọn ít nhất một nền tảng", "error");
        return;
      }
      const localTime = String(data.get("scheduled_at_local") ?? "");
      const scheduledAt = localTime ? new Date(`${localTime}:00+07:00`).toISOString() : null;
      const targets = platforms.map((platform) => {
        const accountId = String(data.get(`account_${platform}`) ?? "").trim();
        return {
          rendition_id: String(data.get(`rendition_${platform}`) ?? ""),
          platform,
          connected_account_id: accountId || null,
          scheduled_at: scheduledAt,
          privacy: String(data.get("privacy") ?? "private"),
        };
      });
      try {
        const queued = await api("/api/v1/publications", {
          method: "POST",
          headers: { "Idempotency-Key": idempotencyKey("publish") },
          body: JSON.stringify({
            creative_id: publicationForm.dataset.creativeId,
            mode: String(data.get("mode") ?? "manual"),
            targets,
          }),
        });
        await pollJob(queued.job_id, (job) => {
          setStatus(publishResult, `Xuất bản: ${job.status} (lần chạy ${job.attempts})`);
        });
        globalThis.location.reload();
      } catch (error) {
        setStatus(
          publishResult,
          error instanceof Error ? error.message : "Không thể tạo lịch xuất bản",
          "error",
        );
      }
    });

    // Auto-poll if creative is generating or render is in progress
    const stateBadge = document.querySelector(".badge.state-GENERATING");
    if (stateBadge) {
      const creativeId =
        document.querySelector("[data-creative-id]")?.dataset?.creativeId ||
        document.querySelector("#publication-form")?.dataset?.creativeId;
      if (creativeId) {
        const pollInterval = setInterval(async () => {
          try {
            const data = await api(`/api/v1/creatives/${encodeURIComponent(creativeId)}`);
            if (data.state !== "GENERATING") {
              clearInterval(pollInterval);
              globalThis.location.reload();
            }
          } catch {
            // ignore transient poll errors
          }
        }, 3000);
      }
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindCreativeDeletion();
    bindBriefForms();
    bindCreativeActions();
  });
})();
