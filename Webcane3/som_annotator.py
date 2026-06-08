from PIL import Image, ImageDraw, ImageFont
import io
from typing import List, Dict, Tuple, Optional


class SoMAnnotator:
    def __init__(self):
        self.box_color = "#FF0000"
        self.text_color = "#FFFFFF"
        self.bg_color = "#FF0000"
        self.font_size = 14
        self.font = self._load_font()
        # display_index -> element_id mapping, updated each annotate() call
        self.index_to_id_map = {}

    def _load_font(self) -> ImageFont.FreeTypeFont:
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for path in font_paths:
            try:
                return ImageFont.truetype(path, self.font_size)
            except:
                continue
        return ImageFont.load_default()

    def annotate(self, screenshot_bytes: bytes, elements: List[Dict], max_elements: int = 80) -> Tuple[bytes, List[Dict]]:
        image = Image.open(io.BytesIO(screenshot_bytes))
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size
        self.index_to_id_map = {}
        filtered = []
        for el in elements:
            bbox = el['bbox']
            if bbox['w'] < 10 or bbox['h'] < 10:
                continue
            if bbox['x'] < 0 or bbox['y'] < 0:
                continue
            if bbox['x'] + bbox['w'] > img_width or bbox['y'] + bbox['h'] > img_height:
                continue
            filtered.append(el)
            if len(filtered) >= max_elements:
                break

        for display_idx, el in enumerate(filtered):
            self.index_to_id_map[display_idx] = el['id']
            self._draw_box(draw, el, str(display_idx))

        print(f"[SoM] Created {len(filtered)} annotations with index->id mapping")
        output = io.BytesIO()
        image.save(output, format='PNG')
        return output.getvalue(), filtered

    def get_element_id(self, display_index: int) -> int:
        element_id = self.index_to_id_map.get(display_index, -1)
        if element_id >= 0:
            print(f"[SoM] Mapping: display index {display_index} -> element ID {element_id}")
        else:
            print(f"[SoM] WARNING: No mapping for display index {display_index}")
        return element_id

    def _draw_box(self, draw: ImageDraw.ImageDraw, element: Dict, label: str):
        bbox = element['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        draw.rectangle([(x, y), (x + w, y + h)], outline=self.box_color, width=2)
        label_bbox = draw.textbbox((0, 0), label, font=self.font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        padding = 2
        label_x = x
        label_y = y - label_height - padding * 2
        if label_y < 0:
            label_y = y + padding
        draw.rectangle(
            [(label_x, label_y), (label_x + label_width + padding * 2, label_y + label_height + padding * 2)],
            fill=self.bg_color
        )
        draw.text((label_x + padding, label_y + padding), label, fill=self.text_color, font=self.font)
