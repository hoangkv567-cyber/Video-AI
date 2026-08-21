"""Operator dashboard page routes (Jinja2 + HTMX, no SPA).

Pages read the database directly for display; every mutation is submitted by
HTMX to the REST API under ``/api/v1/...`` (those endpoints may not exist yet —
pages must render regardless). All timestamps are stored UTC and converted to
Asia/Ho_Chi_Minh only in the Jinja filters registered here.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import (
    Campaign,
    ConnectedAccount,
    CostEvent,
    Creative,
    PublishTarget,
    Rendition,
    Scene,
    ScriptVersion,
    Source,
    User,
)
from app.states import CreativeState, Platform, Role
from app.web import auth
from app.web.filters import format_hcm_time, hcm_datetime_input, to_display_tz

router = APIRouter(prefix="", include_in_schema=False)

_WEB_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _WEB_DIR / "static"

templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))
templates.env.filters["hcm_time"] = format_hcm_time
templates.env.filters["hcm_input"] = hcm_datetime_input

PLATFORM_LABELS: dict[str, str] = {
    Platform.YOUTUBE.value: "YouTube Shorts",
    Platform.FACEBOOK.value: "Facebook Reels",
    Platform.TIKTOK.value: "TikTok",
    Platform.ZALO.value: "Zalo OA",
}

STATE_LABELS: dict[str, str] = {
    CreativeState.DRAFT.value: "Nháp",
    CreativeState.RESEARCHED.value: "Đã nghiên cứu",
    CreativeState.SCRIPT_READY.value: "Kịch bản sẵn sàng",
    CreativeState.SCRIPT_APPROVED.value: "Kịch bản đã duyệt",
    CreativeState.GENERATING.value: "Đang sinh video",
    CreativeState.QC_REQUIRED.value: "Chờ QC",
    CreativeState.READY.value: "Sẵn sàng duyệt",
    CreativeState.FINAL_APPROVED.value: "Đã duyệt cuối",
    CreativeState.SCHEDULED.value: "Đã lên lịch",
    CreativeState.PUBLISHING.value: "Đang đăng",
    CreativeState.PUBLISHED.value: "Đã đăng",
    CreativeState.PARTIAL.value: "Đăng một phần",
    CreativeState.NEEDS_ACTION.value: "Cần thao tác",
    CreativeState.FAILED.value: "Thất bại",
}

DbDep = Annotated[Session, Depends(get_db)]
UserDep = Annotated[User, Depends(auth.login_required)]
EditorDep = Annotated[User, Depends(auth.require_roles(Role.EDITOR))]


def _page_context(request: Request, user: User | None, **extra: Any) -> dict[str, Any]:
    csrf_bind = user.id if user else auth.ANON_CSRF_BIND
    return {
        "request": request,
        "current_user": user,
        "csrf_token": auth.make_csrf_token(csrf_bind),
        "state_labels": STATE_LABELS,
        "platform_labels": PLATFORM_LABELS,
        **extra,
    }


def _cost_totals(db: Session, creative_ids: list[str]) -> dict[str, dict[str, float]]:
    """Per-creative {'projected': x, 'actual': y} sums from the cost ledger."""
    totals: dict[str, dict[str, float]] = {
        cid: {"projected": 0.0, "actual": 0.0} for cid in creative_ids
    }
    if not creative_ids:
        return totals
    rows = db.execute(
        select(CostEvent.creative_id, CostEvent.projected, func.sum(CostEvent.amount_usd))
        .where(CostEvent.creative_id.in_(creative_ids))
        .group_by(CostEvent.creative_id, CostEvent.projected)
    ).all()
    for creative_id, projected, amount in rows:
        key = "projected" if projected else "actual"
        totals[creative_id][key] = float(amount or 0.0)
    return totals


def _safe_next(next_path: str | None) -> str:
    if next_path and next_path.startswith("/") and not next_path.startswith("//"):
        return next_path
    return "/"


# --- Static assets (served by route so no changes to app.main are needed) ----


@router.get("/static/{asset_path:path}")
def static_file(asset_path: str) -> FileResponse:
    base = _STATIC_DIR.resolve()
    target = (base / asset_path).resolve()
    if not target.is_file() or not target.is_relative_to(base):
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(target)


# --- Auth pages --------------------------------------------------------------


@router.get("/login", response_class=HTMLResponse)
def login_page(
    request: Request, next: Annotated[str | None, Query()] = None
) -> HTMLResponse:
    context = _page_context(request, None, next_path=_safe_next(next), error=None)
    return templates.TemplateResponse(request, "login.html", context)


@router.post("/login")
def login_submit(
    request: Request,
    db: DbDep,
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
    csrf_token: Annotated[str, Form()] = "",
    next: Annotated[str, Form()] = "/",
) -> Response:
    if not auth.verify_csrf_token(csrf_token, auth.ANON_CSRF_BIND):
        context = _page_context(
            request, None, next_path=_safe_next(next), error="Phiên không hợp lệ, hãy thử lại"
        )
        return templates.TemplateResponse(request, "login.html", context, status_code=400)

    user = db.execute(
        select(User).where(User.email == email.strip().lower())
    ).scalar_one_or_none()
    if user is None or not user.is_active or not auth.verify_password(password, user.password_hash):
        context = _page_context(
            request, None, next_path=_safe_next(next), error="Sai email hoặc mật khẩu"
        )
        return templates.TemplateResponse(request, "login.html", context, status_code=401)

    response = RedirectResponse(url=_safe_next(next), status_code=303)
    response.set_cookie(
        auth.SESSION_COOKIE,
        auth.create_session_token(user.id),
        max_age=auth.SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=get_settings().app_env not in {"dev", "test"},
    )
    return response


@router.post("/logout")
def logout(user: UserDep, csrf_token: Annotated[str, Form()] = "") -> Response:
    if not auth.verify_csrf_token(csrf_token, user.id):
        raise HTTPException(status_code=400, detail="CSRF token không hợp lệ")
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(auth.SESSION_COOKIE)
    return response


# --- Overview ----------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def overview(request: Request, db: DbDep, user: UserDep) -> HTMLResponse:
    creatives = (
        db.execute(select(Creative).order_by(Creative.created_at.desc())).scalars().all()
    )
    campaigns = {c.id: c for c in db.execute(select(Campaign)).scalars().all()}
    costs = _cost_totals(db, [c.id for c in creatives])
    state_counts: dict[str, int] = {}
    for c in creatives:
        state_counts[c.state] = state_counts.get(c.state, 0) + 1
    context = _page_context(
        request,
        user,
        creatives=creatives,
        campaigns=campaigns,
        costs=costs,
        state_counts=state_counts,
    )
    return templates.TemplateResponse(request, "index.html", context)


# --- Connections -------------------------------------------------------------


@router.get("/connections", response_class=HTMLResponse)
def connections(request: Request, db: DbDep, user: UserDep) -> HTMLResponse:
    accounts = (
        db.execute(select(ConnectedAccount).order_by(ConnectedAccount.platform))
        .scalars()
        .all()
    )
    context = _page_context(
        request,
        user,
        accounts=accounts,
        platforms=[p.value for p in Platform],
    )
    return templates.TemplateResponse(request, "connections.html", context)


# --- Brief form --------------------------------------------------------------


@router.get("/briefs/new", response_class=HTMLResponse)
def brief_new(request: Request, db: DbDep, user: EditorDep) -> HTMLResponse:
    campaigns = (
        db.execute(select(Campaign).order_by(Campaign.created_at.desc())).scalars().all()
    )
    context = _page_context(request, user, campaigns=campaigns)
    return templates.TemplateResponse(request, "brief_new.html", context)


# --- Creative detail ---------------------------------------------------------


def _scene_rows(
    plan: dict[str, Any] | None, scenes: list[Scene]
) -> list[dict[str, Any]]:
    plan_scenes: dict[int, dict[str, Any]] = {}
    narr: dict[str, list[str]] = {"vi": [], "en": []}
    on_screen: dict[str, list[str]] = {"vi": [], "en": []}
    if plan:
        plan_scenes = {int(s.get("index", i)): s for i, s in enumerate(plan.get("scenes", []))}
        locales = plan.get("locales", {})
        for loc in ("vi", "en"):
            narr[loc] = list(locales.get(loc, {}).get("narration", []))
            on_screen[loc] = list(locales.get(loc, {}).get("on_screen_text", []))
    db_scenes = {s.index: s for s in scenes}
    indices = sorted(set(plan_scenes) | set(db_scenes))
    rows: list[dict[str, Any]] = []
    for i in indices:
        ps = plan_scenes.get(i, {})
        rows.append(
            {
                "index": i,
                "plan": ps,
                "scene": db_scenes.get(i),
                "narration_vi": narr["vi"][i] if i < len(narr["vi"]) else "",
                "narration_en": narr["en"][i] if i < len(narr["en"]) else "",
                "on_screen_vi": on_screen["vi"][i] if i < len(on_screen["vi"]) else "",
                "on_screen_en": on_screen["en"][i] if i < len(on_screen["en"]) else "",
            }
        )
    return rows


@router.get("/creatives/{creative_id}", response_class=HTMLResponse)
def creative_detail(
    request: Request, creative_id: str, db: DbDep, user: UserDep
) -> HTMLResponse:
    creative = db.get(Creative, creative_id)
    if creative is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy creative")

    sources = (
        db.execute(
            select(Source).where(Source.creative_id == creative_id).order_by(Source.accessed_at)
        )
        .scalars()
        .all()
    )
    script_version = db.execute(
        select(ScriptVersion)
        .where(ScriptVersion.creative_id == creative_id)
        .order_by(ScriptVersion.version.desc())
    ).scalars().first()
    scenes = (
        db.execute(
            select(Scene).where(Scene.creative_id == creative_id).order_by(Scene.index)
        )
        .scalars()
        .all()
    )
    renditions = (
        db.execute(
            select(Rendition).where(Rendition.creative_id == creative_id).order_by(Rendition.locale)
        )
        .scalars()
        .all()
    )
    targets = (
        db.execute(select(PublishTarget).where(PublishTarget.creative_id == creative_id))
        .scalars()
        .all()
    )
    accounts = db.execute(select(ConnectedAccount)).scalars().all()
    capability_by_platform = {a.platform: a.capability for a in accounts}

    plan = script_version.video_plan if script_version else None
    costs = _cost_totals(db, [creative_id])[creative_id]
    default_schedule = hcm_datetime_input(datetime.now(UTC) + timedelta(hours=1))

    context = _page_context(
        request,
        user,
        creative=creative,
        sources=sources,
        script_version=script_version,
        plan=plan,
        scene_rows=_scene_rows(plan, scenes),
        renditions=renditions,
        targets=targets,
        costs=costs,
        platforms=[p.value for p in Platform],
        capability_by_platform=capability_by_platform,
        default_schedule=default_schedule,
        can_edit=user.role in {Role.ADMIN.value, Role.EDITOR.value},
        can_publish=user.role in {Role.ADMIN.value, Role.PUBLISHER.value},
    )
    return templates.TemplateResponse(request, "creative_detail.html", context)


# --- Publishing board --------------------------------------------------------


@router.get("/publishing", response_class=HTMLResponse)
def publishing(request: Request, db: DbDep, user: UserDep) -> HTMLResponse:
    rows = db.execute(
        select(PublishTarget, Creative)
        .join(Creative, PublishTarget.creative_id == Creative.id)
        .order_by(PublishTarget.platform, PublishTarget.created_at.desc())
    ).all()
    by_platform: dict[str, list[dict[str, Any]]] = {p.value: [] for p in Platform}
    for target, creative in rows:
        by_platform.setdefault(target.platform, []).append(
            {"target": target, "creative": creative}
        )
    context = _page_context(request, user, by_platform=by_platform)
    return templates.TemplateResponse(request, "publishing.html", context)


__all__ = ["router", "templates", "to_display_tz"]
