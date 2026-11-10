from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPORT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = REPORT_ROOT / "img"
OUTPUT_PATH = IMAGE_ROOT / "ucihar_loss_curves_2x2.png"

PANELS = (
    ("ucihar_microbi_curve.png", "(a) MicroBiConvLSTM"),
    ("ucihar_deepconvlstm_curve.png", "(b) DeepConvLSTM"),
    ("ucihar_tinyhar_curve.png", "(c) TinyHAR"),
    ("ucihar_tinierhar_curve.png", "(d) TinierHAR"),
)

HORIZONTAL_GAP = 24
VERTICAL_GAP = 24
LABEL_HEIGHT = 42
LABEL_FONT_SIZE = 26


def load_font() -> ImageFont.FreeTypeFont:
    for font_name in ("DejaVuSerif.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, LABEL_FONT_SIZE)
        except OSError:
            continue
    raise RuntimeError("A DejaVu font is required to label the panels.")


def main() -> None:
    loaded = []
    for file_name, label in PANELS:
        with Image.open(IMAGE_ROOT / file_name) as source:
            loaded.append((source.convert("RGB"), label))

    panel_size = loaded[0][0].size
    if any(image.size != panel_size for image, _ in loaded):
        raise ValueError("All Figure 4 source panels must have equal dimensions.")

    panel_width, panel_height = panel_size
    cell_height = panel_height + LABEL_HEIGHT
    canvas_width = 2 * panel_width + HORIZONTAL_GAP
    canvas_height = 2 * cell_height + VERTICAL_GAP
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font()

    for index, (panel, label) in enumerate(loaded):
        row, column = divmod(index, 2)
        x = column * (panel_width + HORIZONTAL_GAP)
        y = row * (cell_height + VERTICAL_GAP)
        canvas.paste(panel, (x, y))
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0]
        label_x = x + (panel_width - label_width) // 2
        label_y = y + panel_height + 4
        draw.text((label_x, label_y), label, fill="black", font=font)

    canvas.save(OUTPUT_PATH, format="PNG", optimize=True, dpi=(300, 300))
    print(f"Saved {OUTPUT_PATH} at {canvas.size[0]}x{canvas.size[1]} pixels.")


if __name__ == "__main__":
    main()
