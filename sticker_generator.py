"""
sticker_generator.py
Clean anime sticker creation — no emotion labels, no classification.
User poses freely, captures the frame, background removed, saved as sticker.
WhatsApp/Telegram compatible transparent PNG + WebP.
"""
import cv2, numpy as np, os, zipfile
from PIL import Image, ImageFilter
from datetime import datetime

_ROOT            = os.path.dirname(os.path.abspath(__file__))
STICKER_DIR      = os.path.join(_ROOT, "stickers", "custom")
os.makedirs(STICKER_DIR, exist_ok=True)

SIZES = {
    "whatsapp":  (512, 512),
    "telegram":  (512, 512),
    "large":     (800, 800),
}

class StickerGenerator:
    def _bgr_to_pil(self, frame):
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")

    def _remove_bg(self, pil_img, thresh=235):
        data = np.array(pil_img)
        r,g,b,a = data[...,0],data[...,1],data[...,2],data[...,3]
        mask = (r>thresh)&(g>thresh)&(b>thresh)
        data[mask,3] = 0
        # Feather edges
        result = Image.fromarray(data,'RGBA')
        alpha  = result.split()[3]
        alpha  = alpha.filter(ImageFilter.GaussianBlur(radius=1))
        result.putalpha(alpha)
        return result

    def _circle_crop(self, pil_img):
        size = pil_img.size
        mask = Image.new('L', size, 0)
        from PIL import ImageDraw
        ImageDraw.Draw(mask).ellipse((4,4,size[0]-4,size[1]-4), fill=255)
        result = pil_img.copy(); result.putalpha(mask)
        return result

    def _square_crop(self, pil_img, radius=40):
        from PIL import ImageDraw
        mask = Image.new('L', pil_img.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle([0,0,*pil_img.size], radius=radius, fill=255)
        result = pil_img.copy(); result.putalpha(mask)
        return result

    def make(self, frame_bgr, shape="circle", size=(512,512)):
        """
        frame_bgr : OpenCV BGR frame (anime or raw)
        shape     : "circle" | "square" | "raw"
        Returns   : (PIL RGBA image, png_path, webp_path)
        """
        pil  = self._bgr_to_pil(frame_bgr)
        pil  = pil.resize((size[0]-20, size[1]-20), Image.LANCZOS)
        pil  = self._remove_bg(pil)

        if shape == "circle":
            pil = self._circle_crop(pil)
        elif shape == "square":
            pil = self._square_crop(pil)

        # Final canvas with transparent padding
        canvas = Image.new('RGBA', size, (0,0,0,0))
        offset = ((size[0]-pil.width)//2, (size[1]-pil.height)//2)
        canvas.paste(pil, offset, pil)

        ts       = int(datetime.now().timestamp())
        png_path = os.path.join(STICKER_DIR, f"sticker_{ts}.png")
        webp_path= os.path.join(STICKER_DIR, f"sticker_{ts}.webp")
        canvas.save(png_path,  "PNG",  optimize=True)
        canvas.save(webp_path, "WEBP", quality=90)
        return canvas, png_path, webp_path

    def make_pack(self, png_paths, pack_name=None):
        ts       = int(datetime.now().timestamp())
        name     = pack_name or f"sticker_pack_{ts}"
        zip_path = os.path.join(STICKER_DIR, f"{name}.zip")
        with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as zf:
            for p in png_paths:
                if os.path.exists(p):
                    zf.write(p, os.path.basename(p))
        return zip_path

    def load_saved(self):
        """Return list of saved sticker PNG paths sorted newest first."""
        pngs = [os.path.join(STICKER_DIR,f)
                for f in os.listdir(STICKER_DIR) if f.endswith('.png')]
        return sorted(pngs, key=os.path.getmtime, reverse=True)