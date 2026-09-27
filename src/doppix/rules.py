"""Treklar + sahna + svetofor -> hodisa segmentlari (Part A).

Har bir klass alohida funksiya: `(ctx) -> [(boshi, oxiri), ...]`. Chegaralar `P` da —
namuna videolardagi o'z labellarimiz bo'yicha sozlangan (tools/dev_eval.py).
Pikselli qiymatlar FRAME_W=1920 kadr uchun.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config
from .perception import Perception
from .scene import H, Scene, inside, inside_any, side_of_line
from .segments import finalize, mask_to_intervals, merge
from .tracks import Track, build_tracks, light_state

P = {
    # umumiy
    "stop_speed": 12.0,          # px/s dan sekin — "turibdi"
    "edge_px": 12.0,             # box kadr pastki chetiga shuncha px yaqin — kesilgan (oyog'i ko'rinmaydi)
    "move_speed": 25.0,          # px/s dan tez — "yuryapti"
    # jaywalking
    "jw_min_sec": 1.0,
    "jw_core_sec": 0.5,          # orolcha orqali o'tganda ham qatnov qismining o'zida kamida shuncha s
    "jw_along_near_h": 1.2,      # zebraga shuncha bo'y yaqin ...
    "jw_along_cos": 0.85,        # ... va zebra o'qiga parallel (|cos| dan katta) yurgan kadrlar ...
    "jw_along_max": 0.7,         # ... ulushi shundan ko'p bo'lsa — zebra yoqalab yurgan, jaywalking emas
    "ped_max_rel_speed": 2.0,    # bo'y/s: piyoda ~0.3-1, samokat/skuterdagi odam 2.5+ (detektorda samokat klassi yo'q)
    "jw_cross_margin": 60.0,     # zebradan ~1 mashina kengligi uzoqda bo'lsin (zebra cheti hisoblanmaydi)
    "jw_road_margin": 15.0,      # yo'l chegarasidan shuncha px ichkarida
    "jw_min_height": 45.0,       # box balandligi shundan kichik (juda uzoqdagi) odam hisobga olinmaydi
    # failure_to_yield
    "fty_ahead": 120.0,          # piyoda mashina oldida shuncha px gacha bo'lsa — yo'lida
    "fty_lat_margin": 25.0,      # yon tomonga: mashina yarim kengligi + shuncha px
    "fty_lat_approach": 120.0,   # yo'lga qarab kelayotgan piyoda uchun kengroq oraliq
    "fty_approach_speed": 20.0,  # px/s — piyoda yo'lga qarab shundan tez yursa
    "fty_min_sec": 0.5,
    "fty_ped_margin": -10.0,     # piyoda zebra ichida (musbat — chetidan shuncha px ichkarida)
    "fty_ped_min_rel": 0.4,      # bo'y/s: piyoda yurib ketayotgan bo'lsin (yo'l o'rtasida/chetda kutib turgan emas)
    "fty_min_hits": 1,           # "yo'lida" holati kamida shuncha kadrda
    "fty_veh_min_rel": 0.3,      # mashina tezligi (o'z uzunligi/s) shundan past — sekin siljib kutyapti, yo'l bermoqda
    "fty_signal_cw": (0,),       # svetofor C boshqaradigan zebra(lar): bu yerda faqat mashinaga QIZIL paytida
    "fty_red_guard": 2.0,        # qizil yonganidan keyin shuncha s — piyoda svetofori ~2 s farq bilan almashadi
    "fty_max_sec": 8.0,
    # stopped_vehicle / congestion
    "sv_min_sec": 10.0,
    "sv_yield_share": 0.8,       # to'xtash vaqtining shuncha qismida yonidagi zebrada yurayotgan piyoda bo'lsa ...
    "sv_yield_max_sec": 20.0,    # ... va to'xtash shundan qisqa bo'lsa — piyodaga yo'l bermoqda, turib qolmagan
    "cg_min_vehicles": 3,
    "cg_min_sec": 8.0,
    # stop_line
    "sl_min_sec": 3.0,
    "sl_x_ext": 150.0,           # stop-chiziqning x oralig'ini shuncha kengaytirish
    # red_light
    "rl_phase_guard": 1.0,       # svetofor almashgan paytdan shuncha s ichida — hisoblanmaydi (holat kechikishi)
    "rl_max_sec": 8.0,           # mashina chorrahadan chiqmaguncha, lekin ko'pi bilan shuncha
}


@dataclass
class Ctx:
    p: Perception
    scene: Scene
    tracks: list[Track]
    light: np.ndarray            # har qayta ishlangan kadr: 1 qizil, 0 qizil emas, -1 noma'lum
    times: np.ndarray
    who: list = field(default_factory=list)   # (klass, boshi, oxiri, {trek id}) — hodisa ishtirokchilari (sayt uchun)

    def mark(self, label: str, s: float, e: float, *tids) -> None:
        self.who.append((label, float(s), float(e), {int(t) for t in tids}))

    def light_at(self, t: float) -> float:
        i = int(np.clip(np.searchsorted(self.times, t), 0, len(self.times) - 1))
        return float(self.light[i])

    def by_cls(self, classes) -> list[Track]:
        return [tr for tr in self.tracks if tr.cls in classes]


def _runs(tr: Track, mask: np.ndarray, min_sec: float, gap: float = 1.0) -> list[tuple[float, float]]:
    return [(s, e) for s, e in merge(mask_to_intervals(tr.t, mask), gap) if e - s >= min_sec]


def _riders(ctx: Ctx):
    """Vaqt bo'yicha mototsikl/velosiped boxlari — ularning ustidagi odam piyoda emas."""
    out: dict[float, list[np.ndarray]] = {}
    for tr in ctx.by_cls((config.BICYCLE, config.MOTORCYCLE)):
        for t, b in zip(tr.t, tr.box):
            out.setdefault(round(float(t), 3), []).append(b)
    return out


def _is_rider(riders, t: float, foot: np.ndarray) -> bool:
    for b in riders.get(round(float(t), 3), ()):
        if b[0] - 10 <= foot[0] <= b[2] + 10 and b[1] - 20 <= foot[1] <= b[3] + 30:
            return True
    return False


# ── Piyodalar ─────────────────────────────────────────────────────────

def jaywalking(ctx: Ctx):
    """Piyoda qatnov qismida, zebradan tashqarida, va harakatda (orolchada kutib turgan emas)."""
    riders = _riders(ctx)
    out = []
    for tr in ctx.by_cls((config.PERSON,)):
        f = tr.foot
        # kadr pastki chetida kesilgan odam: box pasti — oyoq emas, kadr cheti (joylashuv noto'g'ri)
        whole = tr.box[:, 3] < H - P["edge_px"]
        on_road = ctx.scene.on_road(f, P["jw_road_margin"]) & (tr.height >= P["jw_min_height"]) & whole
        off_cross = ~inside_any(ctx.scene.crosswalks, f, -P["jw_cross_margin"])
        moving = tr.speed() > P["move_speed"] * 0.6
        not_rider = np.array([not _is_rider(riders, t, x) for t, x in zip(tr.t, f)])
        core = on_road & off_cross & moving & not_rider
        # orolcha ustidan yurib o'tish kesib o'tishni ikkiga bo'lmasin: u "ko'prik" — o'zi hodisa emas,
        # lekin ikki tomonidagi yo'l qismlarini bitta hodisaga ulaydi
        bridge = inside_any(ctx.scene.islands, f) & moving & (tr.height >= P["jw_min_height"]) if ctx.scene.islands \
            else np.zeros(len(f), bool)
        rel = _rel_speed(tr)
        for s, e in merge(mask_to_intervals(tr.t, core | bridge), 1.0):
            m = (tr.t >= s) & (tr.t <= e)
            tc = tr.t[m & core]
            # faqat orolchada yurish — hodisa emas: yo'lning o'zida ham kamida jw_core_sec bo'lsin
            if e - s >= P["jw_min_sec"] and len(tc) and tc[-1] - tc[0] >= P["jw_core_sec"]                     and np.median(rel[m]) <= P["ped_max_rel_speed"]                     and _along_share(ctx.scene.crosswalks, tr, m) <= P["jw_along_max"]:
                out.append((s, e))
                ctx.mark("jaywalking", s, e, tr.tid)
    return out


def _along_share(crosswalks, tr: Track, m: np.ndarray) -> float:
    """Oraliqdagi kadrlarning qancha qismida piyoda zebra yonida (jw_along_near_h bo'y ichida) va unga
    parallel yurgan. Zebra yoqalab yurish (samokat/odamlar ko'p paytda chetdan) — zebradan foydalanish;
    haqiqiy jaywalking esa yo'lni qiya kesadi (namunalarda <= 0.67, zebra yoqalab yurganlarda >= 0.72)."""
    import cv2
    if not crosswalks or not m.any():
        return 0.0
    f, v, h = tr.foot[m], _velocity(tr)[m], tr.height[m]
    d = np.array([[cv2.pointPolygonTest(cw.reshape(-1, 1, 2), (float(x), float(y)), True) for cw in crosswalks]
                  for x, y in f])
    axes = []
    for cw in crosswalks:                       # zebraning uzun o'qi — piyodalar yuradigan yo'nalish
        c = cw - cw.mean(0)
        axes.append(np.linalg.svd(c, full_matrices=False)[2][0])
    ax = np.array(axes)[d.argmax(1)]
    sp = np.linalg.norm(v, axis=1)
    cos = np.abs((v * ax).sum(1)) / np.maximum(sp, 1e-6)
    return float(((d.max(1) > -P["jw_along_near_h"] * h) & (cos > P["jw_along_cos"])).mean())


def _rel_speed(tr: Track) -> np.ndarray:
    """Tezlik box balandligiga nisbatan (bo'y/s) — kameradan uzoq-yaqinlikka bog'liq emas."""
    return tr.speed() / np.maximum(tr.height, 1.0)


def _velocity(tr: Track, window: float = 0.5) -> np.ndarray:
    """(n,2) px/s — `window` soniyalik siljish bo'yicha (markaz nuqta)."""
    c = tr.center
    v = np.zeros_like(c)
    j0 = 0
    for i in range(len(c)):
        while tr.t[i] - tr.t[j0] > window:
            j0 += 1
        if tr.t[i] > tr.t[j0]:
            v[i] = (c[i] - c[j0]) / (tr.t[i] - tr.t[j0])
    if len(v) > 1:
        v[0] = v[1]
    return v


def _in_path(vc, vv, vbox, pf, pv) -> bool:
    """Piyoda mashinaning yo'lidami: mashina oldida (harakat yo'nalishi bo'yicha) va yon masofasi
    mashina kengligining yarmiga yaqin, YOKI yo'lga qarab kelyapti. Piyoda allaqachon o'tib,
    yo'ldan uzoqlashib borayotgan bo'lsa (mashina uning orqasidan o'tyapti) — yo'q."""
    sp = float(np.linalg.norm(vv))
    if sp < 1e-6:
        return False
    d = vv / sp
    n = np.array([-d[1], d[0]])
    rel = pf - vc
    along, lat = float(rel @ d), float(rel @ n)
    half_len = 0.5 * float(np.hypot(vbox[2] - vbox[0], vbox[3] - vbox[1]))
    half_w = 0.5 * min(vbox[2] - vbox[0], vbox[3] - vbox[1])
    if along < -0.2 * half_len or along > half_len + P["fty_ahead"]:
        return False
    if abs(lat) <= half_w + P["fty_lat_margin"]:
        return True
    lat_speed = float(pv @ n) * (1 if lat < 0 else -1)   # musbat — yo'lga qarab kelyapti
    return abs(lat) <= half_w + P["fty_lat_approach"] and lat_speed > P["fty_approach_speed"]


def _red_for(ctx: Ctx, t: float, guard: float) -> bool:
    """t paytida svetofor C qizil va qizil yonganiga kamida `guard` s bo'lgan."""
    i = int(np.clip(np.searchsorted(ctx.times, t), 0, len(ctx.times) - 1))
    if ctx.light[i] != 1:
        return False
    j = np.searchsorted(ctx.times, t - guard)
    return bool(np.all(ctx.light[j:i + 1] == 1))


def failure_to_yield(ctx: Ctx):
    """Mashina zebrani kesib o'tyapti, shu paytda zebrada piyoda uning YO'LIDA (oldida yoki yo'lga
    chiqib kelayotgan). Segment — mashina zebraga kirgan paytdan chiqqan paytgacha."""
    riders = _riders(ctx)
    persons = []
    for tr in ctx.by_cls((config.PERSON,)):
        f = tr.foot
        rel = _rel_speed(tr)
        ok = (rel <= P["ped_max_rel_speed"]) & (rel >= P["fty_ped_min_rel"])
        on_cw = [inside(cw, f, P["fty_ped_margin"]) & ok for cw in ctx.scene.crosswalks]
        persons.append((tr, f, _velocity(tr), on_cw))
    out = []
    for veh in ctx.by_cls(config.VEHICLES):
        vc, vv = veh.center, _velocity(veh)
        in_cw_any = [inside(cw, veh.foot, -5) for cw in ctx.scene.crosswalks]
        moving = np.linalg.norm(vv, axis=1) > P["move_speed"]
        for ci in range(len(ctx.scene.crosswalks)):
            for s, e in _runs(veh, in_cw_any[ci], 0.2, gap=0.5):
                hits, ped = 0, None
                for i in np.nonzero((veh.t >= s) & (veh.t <= e) & moving)[0]:
                    t = veh.t[i]
                    # svetoforli zebra: mashinaga yashil bo'lsa yo'l huquqi mashinada (piyoda kechikkan/kutib turibdi)
                    if ci in P["fty_signal_cw"] and not _red_for(ctx, t, P["fty_red_guard"]):
                        continue
                    b = veh.box[i]
                    if np.linalg.norm(vv[i]) / max(np.hypot(b[2] - b[0], b[3] - b[1]), 1.0) < P["fty_veh_min_rel"]:
                        continue
                    for tr, f, pv, on_cw in persons:
                        j = np.searchsorted(tr.t, t)
                        if j >= len(tr.t) or abs(tr.t[j] - t) > 0.05 or not on_cw[ci][j]:
                            continue
                        if _is_rider(riders, t, f[j]):
                            continue
                        if _in_path(vc[i], vv[i], veh.box[i], f[j], pv[j]):
                            hits += 1
                            ped = tr.tid
                            break
                    if hits >= P["fty_min_hits"]:
                        break
                hit = hits >= P["fty_min_hits"]
                # zebradan o'tish bir necha soniya; uzoq turib qolish — bu stop_line/tirbandlik
                if hit and P["fty_min_sec"] <= e - s <= P["fty_max_sec"]:
                    out.append((s, e))
                    ctx.mark("failure_to_yield", s, e, veh.tid, *([ped] if ped is not None else []))
    return out


# ── To'xtagan mashinalar ─────────────────────────────────────────────

def _intersection_zone(ctx: Ctx, pts: np.ndarray) -> np.ndarray:
    """Chorraha ichi: qatnov qismida va 1-zebradan (yuqori) pastda — chapdan kelayotgan navbat emas."""
    road = ctx.scene.on_road(pts, 10)
    if not ctx.scene.crosswalks:
        return road
    cw = ctx.scene.crosswalks[0]
    below = pts[:, 1] > np.interp(pts[:, 0], np.sort(cw[:, 0]), cw[np.argsort(cw[:, 0]), 1]) + 20
    return road & below


def _stopped_runs(ctx: Ctx, min_sec: float):
    res = []
    for veh in ctx.by_cls(config.VEHICLES):
        stopped = (veh.speed(2.0) < P["stop_speed"]) & _intersection_zone(ctx, veh.foot)
        for s, e in _runs(veh, stopped, min_sec, gap=1.5):
            res.append((veh, s, e))
    return res


def congestion_and_stopped(ctx: Ctx):
    """Bir vaqtda kamida N ta mashina chorrahada turib qolsa — tirbandlik;
    bitta mashina >=10 s turib qolsa (tirbandlik bo'lmagan paytda) — turib qolgan mashina."""
    stops = _stopped_runs(ctx, 2.0)
    times = ctx.times
    count = np.zeros(len(times), dtype=int)
    for _, s, e in stops:
        count += (times >= s) & (times <= e)
    cong = [(s, e) for s, e in merge(mask_to_intervals(times, count >= P["cg_min_vehicles"]), 2.0)
            if e - s >= P["cg_min_sec"]]
    for cs, ce in cong:
        ctx.mark("congestion", cs, ce, *[v.tid for v, s, e in stops if s < ce and e > cs])
    single = []
    peds = None
    for veh, s, e in stops:
        if e - s < P["sv_min_sec"]:
            continue
        overlap = sum(max(0.0, min(e, ce) - max(s, cs)) for cs, ce in cong)
        if overlap >= 0.5 * (e - s):
            continue
        if e - s < P["sv_yield_max_sec"]:
            peds = _walking_on_crosswalks(ctx) if peds is None else peds
            if _yield_share(veh, s, e, peds) >= P["sv_yield_share"]:
                continue                        # piyodalarga yo'l berib kutyapti
        single.append((s, e))
        ctx.mark("stopped_vehicle", s, e, veh.tid)
    return cong, single


def _walking_on_crosswalks(ctx: Ctx) -> dict:
    """Vaqt (0.01 s) -> zebra ustida yurib ketayotgan piyodalarning oyoq nuqtalari."""
    out: dict[float, list[np.ndarray]] = {}
    for tr in ctx.by_cls((config.PERSON,)):
        rel = _rel_speed(tr)
        on = np.zeros(len(tr.t), bool)
        for cw in ctx.scene.crosswalks:
            on |= inside(cw, tr.foot, -10)
        for t, f in zip(tr.t[on & (rel > 0.2) & (rel < P["ped_max_rel_speed"])], tr.foot[on & (rel > 0.2) & (rel < P["ped_max_rel_speed"])]):
            out.setdefault(round(float(t), 2), []).append(f)
    return out


def _yield_share(veh: Track, s: float, e: float, peds: dict) -> float:
    """To'xtash vaqtining qancha qismida mashina yonida (boxidan 1 mashina o'lchami ichida) zebrada piyoda yurgan."""
    m = (veh.t >= s) & (veh.t <= e)
    hit = []
    for t, b in zip(veh.t[m], veh.box[m]):
        w = max(b[2] - b[0], b[3] - b[1])
        hit.append(any(b[0] - w <= f[0] <= b[2] + w and b[1] - w <= f[1] <= b[3] + w for f in peds.get(round(float(t), 2), ())))
    return float(np.mean(hit)) if hit else 0.0


# ── Svetofor bilan bog'liq ────────────────────────────────────────────

def stop_line(ctx: Ctx):
    """Qizilda stop-chiziqdan o'tib, chorrahaga kirmasdan to'xtab turgan mashina.
    Boshi — to'xtagan payt; oxiri — yashil yongan (yoki mashina yurib ketgan) payt."""
    if not ctx.scene.stop_lines:
        return []
    line = ctx.scene.stop_lines[0]
    x0, x1 = line[:, 0].min() - P["sl_x_ext"], line[:, 0].max() + P["sl_x_ext"]
    cw = ctx.scene.crosswalks[0] if ctx.scene.crosswalks else None
    out = []
    for veh in ctx.by_cls(config.VEHICLES):
        f = veh.foot
        past = side_of_line(line, f) > 0
        in_x = (f[:, 0] >= x0) & (f[:, 0] <= x1)
        not_through = np.ones(len(f), dtype=bool) if cw is None else inside(cw, f, -40) | (side_of_line(line, f) > 0) & (
            f[:, 1] < np.interp(f[:, 0], np.sort(cw[:, 0]), cw[np.argsort(cw[:, 0]), 1]) + 60)
        stopped = veh.speed(2.0) < P["stop_speed"]
        red = np.array([ctx.light_at(t) == 1 for t in veh.t])
        for s, e in _runs(veh, past & in_x & not_through & stopped & red, P["sl_min_sec"], gap=1.5):
            # oxiri: svetofor yashilga o'tgan payt
            later = ctx.times[(ctx.times > s) & (ctx.light == 0)]
            green = later[0] if len(later) else e
            out.append((s, max(e, min(green, e + 60))))
            ctx.mark("stop_line", *out[-1], veh.tid)
    return out


def red_light(ctx: Ctx):
    """Qizilda stop-chiziqni kesib o'tish. Boshi — mashina oldi (box pastki cheti, kameraga qarab
    kelyapti) chiziqni kesgan payt; oxiri — trek tugagan (chorrahadan/kadrdan chiqqan) payt."""
    if not ctx.scene.stop_lines:
        return []
    line = ctx.scene.stop_lines[0]
    x0, x1 = line[:, 0].min() - P["sl_x_ext"], line[:, 0].max() + P["sl_x_ext"]
    # svetofor almashgan paytlar
    changes = ctx.times[1:][np.diff(ctx.light) != 0]
    out = []
    for veh in ctx.by_cls(config.VEHICLES):
        f = veh.foot
        side = side_of_line(line, f)
        for i in range(1, len(f)):
            if not (side[i - 1] <= 0 < side[i] and x0 <= f[i, 0] <= x1):
                continue
            t = veh.t[i]
            if ctx.light_at(t) != 1:
                continue
            if len(changes) and np.min(np.abs(changes - t)) < P["rl_phase_guard"]:
                continue
            out.append((t, min(veh.t[-1], t + P["rl_max_sec"])))
            ctx.mark("red_light", *out[-1], veh.tid)
    return out


# ── Hammasi ───────────────────────────────────────────────────────────

def aligned_scene(p: Perception, scene: Scene | None = None) -> Scene:
    """Etalon sahnani shu videoning kamera holatiga moslaydi (fon bo'yicha)."""
    from .align import estimate, load_ref
    scene = scene or Scene.load()
    ref = load_ref()
    if ref is None or p.bg is None:
        return scene
    return scene.transformed(estimate(ref, p.bg))


def detect(p: Perception, scene: Scene | None = None, detail: bool = False):
    """Hodisalar [[boshi, oxiri, klass], ...]. detail=True — qo'shimcha har hodisaning ishtirokchi trek
    id'lari (saytdagi videoda ajratib ko'rsatish uchun; topshiriq natijasiga ta'sir qilmaydi)."""
    scene = aligned_scene(p, scene)
    ctx = Ctx(p, scene, build_tracks(p), light_state(p, scene), p.times)
    dur = p.duration
    events = []

    def add(label, intervals, **kw):
        for s, e in finalize(intervals, dur, **kw):
            events.append([s, e, label])

    add("jaywalking", jaywalking(ctx), min_len=1.0, gap=1.5)
    add("failure_to_yield", failure_to_yield(ctx), min_len=0.5, gap=0.5)
    cong, single = congestion_and_stopped(ctx)
    add("congestion", cong, min_len=P["cg_min_sec"], gap=3.0)
    add("stopped_vehicle", single, min_len=P["sv_min_sec"], gap=2.0)
    add("stop_line", stop_line(ctx), min_len=P["sl_min_sec"], gap=2.0)
    add("red_light", red_light(ctx), min_len=0.5, gap=0.5)
    events = sorted(events)
    if not detail:
        return events
    involved = [sorted(set().union(*[w[3] for w in ctx.who if w[0] == lab and w[1] < e and w[2] > s]))
                for s, e, lab in events]
    return events, involved
