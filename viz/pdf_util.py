from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_bgr_pdf(path: str | Path, pages: list[np.ndarray]) -> None:
    """
    Write a multi-page PDF from BGR uint8 images (JPEG-compressed per page).

    Uses only the standard library and OpenCV (no matplotlib / Pillow).
    """
    if not pages:
        raise ValueError("write_bgr_pdf requires at least one page")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    jpeg_pages: list[bytes] = []
    widths: list[int] = []
    heights: list[int] = []
    for img in pages:
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
            raise TypeError("each page must be uint8 BGR with shape (H, W, 3)")
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise RuntimeError("JPEG encode failed for PDF page")
        jpeg_pages.append(buf.tobytes())
        heights.append(int(img.shape[0]))
        widths.append(int(img.shape[1]))

    objects: list[bytes] = [b""]  # 1-based object numbers

    def add_obj(body: bytes) -> int:
        objects.append(body)
        return len(objects) - 1

    xobject_ids: list[int] = []
    content_ids: list[int] = []
    for jpeg, w, h in zip(jpeg_pages, widths, heights):
        xobj = (
            f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 "
            f"/Filter /DCTDecode /Length {len(jpeg)} >>\nstream\n".encode("latin-1")
            + jpeg
            + b"\nendstream"
        )
        xobject_ids.append(add_obj(xobj))
        draw = f"q {w} 0 0 {h} 0 0 cm /Im1 Do Q".encode("latin-1")
        content_ids.append(
            add_obj(
                f"<< /Length {len(draw)} >>\nstream\n".encode("latin-1") + draw + b"\nendstream"
            )
        )

    page_specs: list[tuple[int, int, int, int]] = list(
        zip(widths, heights, content_ids, xobject_ids)
    )
    page_ids: list[int] = []
    pages_id = 0
    for w, h, cid, xid in page_specs:
        page_ids.append(
            add_obj(
                (
                    f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {w} {h}] "
                    f"/Contents {cid} 0 R "
                    f"/Resources << /XObject << /Im1 {xid} 0 R >> >> >>"
                ).encode("latin-1")
            )
        )

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_id = add_obj(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")
    )
    catalog_id = add_obj(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"))

    for i, (w, h, cid, xid) in enumerate(page_specs):
        objects[page_ids[i]] = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {w} {h}] "
            f"/Contents {cid} 0 R "
            f"/Resources << /XObject << /Im1 {xid} 0 R >> >> >>"
        ).encode("latin-1")

    parts: list[bytes] = [b"%PDF-1.4\n"]
    offsets: list[int] = [0]
    for i, body in enumerate(objects):
        if i == 0:
            continue
        offsets.append(sum(len(p) for p in parts))
        parts.append(f"{i} 0 obj\n".encode("latin-1") + body + b"\nendobj\n")

    xref_start = sum(len(p) for p in parts)
    parts.append(f"xref\n0 {len(objects)}\n".encode("latin-1"))
    parts.append(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        parts.append(f"{off:010d} 00000 n \n".encode("latin-1"))
    parts.append(
        (
            f"trailer\n<< /Size {len(objects)} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("latin-1")
    )

    out_path.write_bytes(b"".join(parts))
