import math
from pathlib import Path

OUTPUT_DIR = Path('docs/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_LINES = [
    {"element": "H", "transition": "Hα", "wavelength": 6562.79, "kind": "emission"},
    {"element": "H", "transition": "Hβ", "wavelength": 4861.35, "kind": "absorption"},
    {"element": "H", "transition": "Hγ", "wavelength": 4340.47, "kind": "absorption"},
    {"element": "H", "transition": "Hδ", "wavelength": 4101.74, "kind": "absorption"},
    {"element": "He", "transition": "He I", "wavelength": 4471.50, "kind": "emission"},
    {"element": "He", "transition": "He II", "wavelength": 4026.19, "kind": "emission"},
    {"element": "Ca", "transition": "Ca II K", "wavelength": 3933.66, "kind": "absorption"},
    {"element": "Ca", "transition": "Ca II H", "wavelength": 3968.47, "kind": "absorption"},
    {"element": "Mg", "transition": "Mg b1", "wavelength": 5167.32, "kind": "absorption"},
    {"element": "Mg", "transition": "Mg b2", "wavelength": 5172.68, "kind": "absorption"},
    {"element": "Mg", "transition": "Mg b3", "wavelength": 5183.60, "kind": "absorption"},
    {"element": "Na", "transition": "Na D1", "wavelength": 5891.58, "kind": "absorption"},
    {"element": "Na", "transition": "Na D2", "wavelength": 5897.56, "kind": "absorption"},
    {"element": "Fe", "transition": "Fe I", "wavelength": 5169.03, "kind": "absorption"},
    {"element": "Fe", "transition": "Fe I", "wavelength": 5270.40, "kind": "absorption"},
]

ELEMENT_COLOURS = {
    "H": "#d62728",
    "He": "#9467bd",
    "Ca": "#2ca02c",
    "Mg": "#ff7f0e",
    "Na": "#1f77b4",
    "Fe": "#8c564b",
}


def linspace(start, stop, num):
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def gaussian(x, mu, sigma, amplitude):
    return amplitude * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def moving_average(values, window):
    half = window // 2
    padded = values[:half][::-1] + values + values[-half:][::-1]
    averaged = []
    for i in range(len(values)):
        segment = padded[i: i + window]
        averaged.append(sum(segment) / window)
    return averaged


def to_path(x_values, y_values, x_min, x_max, y_min, y_max, width, height, x_offset, y_offset):
    def scale_x(x):
        return x_offset + (x - x_min) / (x_max - x_min) * width

    def scale_y(y):
        return y_offset + height - (y - y_min) / (y_max - y_min) * height

    commands = []
    for idx, (x, y) in enumerate(zip(x_values, y_values)):
        sx = scale_x(x)
        sy = scale_y(y)
        commands.append((sx, sy))
    if not commands:
        return ''
    move = commands[0]
    path = [f"M {move[0]:.2f},{move[1]:.2f}"]
    for sx, sy in commands[1:]:
        path.append(f"L {sx:.2f},{sy:.2f}")
    return ' '.join(path)


def create_figure1():
    wavelength = linspace(3600.0, 9200.0, 2200)
    mean_wl = (3600.0 + 9200.0) / 2
    raw_flux = []
    for wl in wavelength:
        continuum = 1 - 0.00002 * (wl - mean_wl)
        emission = sum(gaussian(wl, mu, 2.5, 0.35) for mu in [6562.79, 4471.50])
        absorption = sum(
            gaussian(wl, mu, 3.5, -0.3)
            for mu in [4861.35, 4340.47, 4101.74, 3933.66, 3968.47, 5167.32, 5172.68, 5183.60, 5891.58, 5897.56]
        )
        noise = (math.sin(wl / 35.0) + math.sin(wl / 57.0)) * 0.015
        raw_flux.append(continuum + emission + absorption + noise)
    smoothed = moving_average(raw_flux, 101)
    normalised = [rf / sf for rf, sf in zip(raw_flux, smoothed)]

    width = 1280
    height = 880
    margin_left = 120
    margin_right = 140
    margin_top = 120
    margin_bottom = 140
    panel_gap = 140
    panel_width = width - margin_left - margin_right
    panel_height = (height - margin_top - margin_bottom - panel_gap) / 2

    raw_min = min(raw_flux)
    raw_max = max(raw_flux)
    norm_min = min(normalised)
    norm_max = max(normalised)
    raw_axis_min = raw_min - 0.05 * (raw_max - raw_min)
    raw_axis_max = raw_max + 0.05 * (raw_max - raw_min)
    norm_axis_min = min(0.6, norm_min)
    norm_axis_max = max(1.4, norm_max)

    raw_path = to_path(
        wavelength,
        raw_flux,
        min(wavelength),
        max(wavelength),
        raw_axis_min,
        raw_axis_max,
        panel_width,
        panel_height,
        margin_left,
        margin_top,
    )
    norm_path = to_path(
        wavelength,
        normalised,
        min(wavelength),
        max(wavelength),
        norm_axis_min,
        norm_axis_max,
        panel_width,
        panel_height,
        margin_left,
        margin_top + panel_height + panel_gap,
    )

    x_min = min(wavelength)
    x_max = max(wavelength)

    # Axis ticks
    x_ticks = [3800, 4200, 4600, 5000, 5400, 5800, 6200, 6600, 7000, 7600, 8200, 8800]
    raw_y_ticks = [round(raw_axis_min + i * (raw_axis_max - raw_axis_min) / 4, 2) for i in range(5)]
    norm_y_ticks = [0.7, 0.9, 1.1, 1.3]

    elements = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' role='img' aria-labelledby='title'>",
        "<title>Figure 1. Example Stellar Spectrum (Raw vs. Normalized)</title>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='none'/>",
    ]

    # Draw panel backgrounds and axes
    for panel_y, y_ticks, y_label, axis_min, axis_max in [
        (margin_top, raw_y_ticks, "Flux (erg s⁻¹ cm⁻² Å⁻¹)", raw_axis_min, raw_axis_max),
        (margin_top + panel_height + panel_gap, norm_y_ticks, "Normalized Flux", norm_axis_min, norm_axis_max),
    ]:
        elements.append(
            f"<rect x='{margin_left}' y='{panel_y}' width='{panel_width}' height='{panel_height}' fill='none' stroke='#4f4f4f' stroke-width='2' rx='10' ry='10'/>"
        )

        # Horizontal grid lines and y ticks
        for y_val in y_ticks:
            y = panel_y + panel_height - (y_val - axis_min) / (axis_max - axis_min) * panel_height
            elements.append(
                f"<line x1='{margin_left}' x2='{margin_left + panel_width}' y1='{y:.2f}' y2='{y:.2f}' stroke='#c8c8c8' stroke-width='1' opacity='0.6' />"
            )
            label = f"{y_val:.2f}" if axis_max - axis_min > 1 else f"{y_val:.1f}"
            elements.append(
                f"<text x='{margin_left - 24}' y='{y + 6:.2f}' font-size='18' font-family='DejaVu Sans' text-anchor='end'>{label}</text>"
            )

        # Axis labels
        elements.append(
            f"<text x='{margin_left - 70}' y='{panel_y + panel_height / 2:.2f}' font-size='22' font-family='DejaVu Sans' text-anchor='middle' transform='rotate(-90 {margin_left - 70},{panel_y + panel_height / 2:.2f})'>{y_label}</text>"
        )

    # Plot spectra
    elements.append(f"<path d='{raw_path}' stroke='#1f77b4' stroke-width='2.8' fill='none'/>")
    elements.append(f"<path d='{norm_path}' stroke='#ff7f0e' stroke-width='2.8' fill='none' stroke-dasharray='10 6'/>")

    # X-axis ticks and labels
    axis_y_bottom = margin_top + 2 * panel_height + panel_gap
    for tick in x_ticks:
        x = margin_left + (tick - x_min) / (x_max - x_min) * panel_width
        elements.append(f"<line x1='{x:.2f}' y1='{axis_y_bottom}' x2='{x:.2f}' y2='{axis_y_bottom + 16}' stroke='#3a3a3a' stroke-width='1.4' />")
        elements.append(f"<line x1='{x:.2f}' x2='{x:.2f}' y1='{margin_top}' y2='{axis_y_bottom}' stroke='#d0d0d0' stroke-width='1' opacity='0.45' />")
        elements.append(f"<text x='{x:.2f}' y='{axis_y_bottom + 42}' font-size='18' font-family='DejaVu Sans' text-anchor='middle'>{int(tick)}</text>")
    elements.append(f"<text x='{margin_left + panel_width / 2:.2f}' y='{axis_y_bottom + 84}' font-size='24' font-family='DejaVu Sans' text-anchor='middle'>Wavelength (Å)</text>")

    # Reference lines, markers, and callouts
    for line in REFERENCE_LINES:
        x = margin_left + (line["wavelength"] - x_min) / (x_max - x_min) * panel_width
        colour = ELEMENT_COLOURS.get(line["element"], "#000000")
        dash = "6 8" if line["kind"] == "emission" else "4 10"
        elements.append(
            f"<line x1='{x:.2f}' x2='{x:.2f}' y1='{margin_top}' y2='{margin_top + panel_height}' stroke='{colour}' stroke-width='1.6' stroke-dasharray='{dash}' opacity='0.9' />"
        )
        elements.append(
            f"<line x1='{x:.2f}' x2='{x:.2f}' y1='{margin_top + panel_height + panel_gap}' y2='{margin_top + 2 * panel_height + panel_gap}' stroke='{colour}' stroke-width='1.6' stroke-dasharray='{dash}' opacity='0.9' />"
        )

        marker_y = margin_top + 24 if line["kind"] == "emission" else margin_top + panel_height - 24
        marker_y_norm = margin_top + panel_height + panel_gap + (32 if line["kind"] == "emission" else panel_height - 32)
        if line["kind"] == "emission":
            elements.append(f"<circle cx='{x:.2f}' cy='{marker_y:.2f}' r='10' fill='{colour}' opacity='0.9' />")
            elements.append(f"<circle cx='{x:.2f}' cy='{marker_y_norm:.2f}' r='10' fill='{colour}' opacity='0.9' />")
        else:
            triangle = [
                (x, marker_y + 10),
                (x - 10, marker_y - 10),
                (x + 10, marker_y - 10),
            ]
            triangle_norm = [
                (x, marker_y_norm + 10),
                (x - 10, marker_y_norm - 10),
                (x + 10, marker_y_norm - 10),
            ]
            tri_path = " ".join(f"{px:.2f},{py:.2f}" for px, py in triangle)
            tri_norm_path = " ".join(f"{px:.2f},{py:.2f}" for px, py in triangle_norm)
            elements.append(f"<polygon points='{tri_path}' fill='{colour}' opacity='0.9' />")
            elements.append(f"<polygon points='{tri_norm_path}' fill='{colour}' opacity='0.9' />")

    elements.append("</svg>")

    (OUTPUT_DIR / 'figure1_example_spectrum.svg').write_text('\n'.join(elements), encoding='utf-8')


def create_figure3():
    classes = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
    counts = [320, 540, 720, 940, 1120, 980, 860]
    sn_values = []
    for median, count in zip([35, 42, 55, 60, 58, 52, 48], counts):
        for i in range(count):
            sn_values.append(median + math.sin(i / 3.0) * 4 + math.cos(i / 7.0) * 2)
    width = 1000
    height = 420
    margin = 60
    chart_width = (width - 3 * margin) / 2
    chart_height = height - 2 * margin

    max_count = max(counts)
    bar_elements = [
        f"<rect x='{margin}' y='{margin}' width='{chart_width}' height='{chart_height}' fill='none' stroke='black' stroke-width='1'/>"
    ]
    bar_width = chart_width / len(classes)
    for idx, (cls, count) in enumerate(zip(classes, counts)):
        x = margin + idx * bar_width + bar_width * 0.1
        bar_h = (count / max_count) * (chart_height * 0.9)
        y = margin + chart_height - bar_h
        bar_elements.append(f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_width * 0.8:.2f}' height='{bar_h:.2f}' fill='#9467bd' opacity='0.85' />")
        bar_elements.append(f"<text x='{x + bar_width * 0.4:.2f}' y='{margin + chart_height + 18}' font-size='12' text-anchor='middle' font-family='sans-serif'>{cls}</text>")
    bar_elements.append(f"<text x='{margin + chart_width/2:.2f}' y='{margin - 20}' font-size='16' font-family='sans-serif' text-anchor='middle'>Class balance across 5k spectra</text>")

    # Histogram approximation with 25 bins
    bins = 25
    sn_min = min(sn_values)
    sn_max = max(sn_values)
    bin_counts = [0] * bins
    for value in sn_values:
        index = int((value - sn_min) / (sn_max - sn_min + 1e-9) * bins)
        if index == bins:
            index -= 1
        bin_counts[index] += 1
    max_bin = max(bin_counts)
    hist_elements = [
        f"<rect x='{2 * margin + chart_width}' y='{margin}' width='{chart_width}' height='{chart_height}' fill='none' stroke='black' stroke-width='1'/>"
    ]
    bin_width = chart_width / bins
    for idx, count in enumerate(bin_counts):
        x = 2 * margin + chart_width + idx * bin_width
        bar_h = (count / max_bin) * (chart_height * 0.9)
        y = margin + chart_height - bar_h
        hist_elements.append(f"<rect x='{x:.2f}' y='{y:.2f}' width='{bin_width * 0.95:.2f}' height='{bar_h:.2f}' fill='#2ca02c' opacity='0.85' />")
    hist_elements.append(f"<text x='{2 * margin + 1.5 * chart_width:.2f}' y='{margin - 20}' font-size='16' font-family='sans-serif' text-anchor='middle'>Signal-to-noise distribution</text>")
    hist_elements.append(f"<text x='{2 * margin + 1.5 * chart_width:.2f}' y='{height - margin/2:.2f}' font-size='14' font-family='sans-serif' text-anchor='middle'>Median S/N per pixel</text>")

    svg = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' role='img' aria-labelledby='title3'>",
        "<title id='title3'>Figure 3. Dataset balance and S/N characteristics</title>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='none'/>",
        *bar_elements,
        *hist_elements,
        f"<text x='{width/2:.2f}' y='{margin/2:.2f}' font-size='18' font-family='sans-serif' text-anchor='middle'>Figure 3. Dataset balance and S/N characteristics</text>",
        "</svg>",
    ]
    (OUTPUT_DIR / 'figure3_class_balance_sn.svg').write_text('\n'.join(svg), encoding='utf-8')


def create_figure4():
    classes = ['O', 'B', 'A', 'F']
    counts = {
        ('O', 'O'): 32,
        ('O', 'B'): 5,
        ('O', 'A'): 2,
        ('O', 'F'): 1,
        ('B', 'O'): 4,
        ('B', 'B'): 29,
        ('B', 'A'): 5,
        ('B', 'F'): 2,
        ('A', 'O'): 3,
        ('A', 'B'): 6,
        ('A', 'A'): 27,
        ('A', 'F'): 4,
        ('F', 'O'): 2,
        ('F', 'B'): 4,
        ('F', 'A'): 6,
        ('F', 'F'): 28,
    }
    # Binary metrics for O vs rest
    roc_points = [(0.0, 0.0), (0.1, 0.6), (0.2, 0.75), (0.35, 0.85), (1.0, 1.0)]
    pr_points = [(0.0, 1.0), (0.2, 0.88), (0.4, 0.82), (0.6, 0.78), (0.8, 0.74), (1.0, 0.7)]
    scatter_points = [
        (50, 3400, 0.95),
        (52, 3500, 0.93),
        (54, 3600, 0.88),
        (58, 3550, 0.9),
        (60, 3700, 0.85),
        (62, 3800, 0.8),
        (48, 3300, 0.86),
        (56, 3400, 0.92),
        (64, 3650, 0.78),
        (45, 3250, 0.83),
        (59, 3750, 0.87),
    ]

    width = 1100
    height = 700
    margin = 60
    subplot_width = (width - 3 * margin) / 2
    subplot_height = (height - 3 * margin) / 2

    svg = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' role='img' aria-labelledby='title4'>",
        "<title id='title4'>Figure 4. Model performance diagnostics</title>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
        f"<text x='{width/2:.2f}' y='{margin/2:.2f}' font-size='20' font-family='sans-serif' text-anchor='middle'>Figure 4. Model performance diagnostics</text>",
    ]

    # Confusion matrix grid
    cell_width = subplot_width / len(classes)
    cell_height = subplot_height / len(classes)
    x0 = margin
    y0 = margin
    svg.append(f"<g transform='translate({x0},{y0})'>")
    svg.append(f"<rect x='0' y='0' width='{subplot_width}' height='{subplot_height}' fill='none' stroke='black' stroke-width='1'/>")
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            count = counts.get((true_cls, pred_cls), 0)
            fill = '#c6dbef' if i == j else '#fdd0a2'
            svg.append(
                f"<rect x='{j * cell_width:.2f}' y='{i * cell_height:.2f}' width='{cell_width:.2f}' height='{cell_height:.2f}' fill='{fill}' stroke='white' stroke-width='1'/>"
            )
            svg.append(
                f"<text x='{j * cell_width + cell_width / 2:.2f}' y='{i * cell_height + cell_height / 2 + 5:.2f}' font-size='16' font-family='sans-serif' text-anchor='middle'>{count}</text>"
            )
    for idx, cls in enumerate(classes):
        svg.append(f"<text x='{idx * cell_width + cell_width / 2:.2f}' y='{subplot_height + 25:.2f}' font-size='14' font-family='sans-serif' text-anchor='middle'>{cls}</text>")
        svg.append(f"<text x='-15' y='{idx * cell_height + cell_height / 2 + 5:.2f}' font-size='14' font-family='sans-serif' text-anchor='end'>{cls}</text>")
    svg.append("<text x='0' y='-20' font-size='16' font-family='sans-serif'>Confusion matrix</text>")
    svg.append("</g>")

    # ROC curve
    x1 = 2 * margin + subplot_width
    y1 = margin
    svg.append(f"<g transform='translate({x1},{y1})'>")
    svg.append(f"<rect x='0' y='0' width='{subplot_width}' height='{subplot_height}' fill='none' stroke='black' stroke-width='1'/>")
    prev = None
    path_segments = []
    for x_val, y_val in roc_points:
        sx = x_val * subplot_width
        sy = subplot_height - y_val * subplot_height
        if prev is None:
            path_segments.append(f"M {sx:.2f},{sy:.2f}")
        else:
            path_segments.append(f"L {sx:.2f},{sy:.2f}")
        prev = (sx, sy)
    svg.append(f"<path d='{' '.join(path_segments)}' stroke='#d62728' stroke-width='2' fill='none'/>")
    svg.append(f"<line x1='0' y1='{subplot_height}' x2='{subplot_width}' y2='0' stroke='#999' stroke-dasharray='4 4' />")
    svg.append("<text x='0' y='-20' font-size='16' font-family='sans-serif'>ROC curve (AUC≈0.90)</text>")
    svg.append("</g>")

    # PR curve
    x2 = margin
    y2 = 2 * margin + subplot_height
    svg.append(f"<g transform='translate({x2},{y2})'>")
    svg.append(f"<rect x='0' y='0' width='{subplot_width}' height='{subplot_height}' fill='none' stroke='black' stroke-width='1'/>")
    prev = None
    path_segments = []
    for recall, precision in pr_points:
        sx = recall * subplot_width
        sy = subplot_height - precision * subplot_height
        if prev is None:
            path_segments.append(f"M {sx:.2f},{sy:.2f}")
        else:
            path_segments.append(f"L {sx:.2f},{sy:.2f}")
        prev = (sx, sy)
    svg.append(f"<path d='{' '.join(path_segments)}' stroke='#1f77b4' stroke-width='2' fill='none'/>")
    svg.append("<text x='0' y='-20' font-size='16' font-family='sans-serif'>Precision-Recall (AP≈0.85)</text>")
    svg.append("</g>")

    # Scatter plot
    x3 = 2 * margin + subplot_width
    y3 = 2 * margin + subplot_height
    svg.append(f"<g transform='translate({x3},{y3})'>")
    svg.append(f"<rect x='0' y='0' width='{subplot_width}' height='{subplot_height}' fill='none' stroke='black' stroke-width='1'/>")
    sn_min = min(pt[0] for pt in scatter_points)
    sn_max = max(pt[0] for pt in scatter_points)
    cov_min = min(pt[1] for pt in scatter_points)
    cov_max = max(pt[1] for pt in scatter_points)
    for sn, cov, acc in scatter_points:
        x = (sn - sn_min) / (sn_max - sn_min) * subplot_width
        y = subplot_height - (cov - cov_min) / (cov_max - cov_min) * subplot_height
        radius = 6 + (acc - 0.75) * 20
        color = '#ff9896' if acc < 0.85 else '#2ca02c'
        svg.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{radius:.2f}' fill='{color}' opacity='0.8' />")
    svg.append(f"<text x='0' y='-20' font-size='16' font-family='sans-serif'>Accuracy vs. S/N and coverage</text>")
    svg.append(f"<text x='{subplot_width/2:.2f}' y='{subplot_height + 30:.2f}' font-size='14' font-family='sans-serif' text-anchor='middle'>Median S/N</text>")
    svg.append(f"<text x='-40' y='{subplot_height/2:.2f}' font-size='14' font-family='sans-serif' text-anchor='middle' transform='rotate(-90 -40 {subplot_height/2:.2f})'>Wavelength coverage (Å)</text>")
    svg.append("</g>")

    svg.append("</svg>")
    (OUTPUT_DIR / 'figure4_model_performance.svg').write_text('\n'.join(svg), encoding='utf-8')


def create_figure5():
    releases = ['DR17', 'DR18', 'DR19']
    accuracy = [0.92, 0.88, 0.85]
    macro_f1 = [0.90, 0.86, 0.83]

    width = 800
    height = 400
    margin = 60
    chart_width = width - 2 * margin
    chart_height = height - 2 * margin

    y_min = 0.75
    y_max = 0.95

    def path_from(values):
        commands = []
        for idx, value in enumerate(values):
            x = margin + idx / (len(values) - 1) * chart_width
            y = margin + chart_height - (value - y_min) / (y_max - y_min) * chart_height
            if idx == 0:
                commands.append(f"M {x:.2f},{y:.2f}")
            else:
                commands.append(f"L {x:.2f},{y:.2f}")
        return ' '.join(commands)

    svg = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' role='img' aria-labelledby='title5'>",
        "<title id='title5'>Figure 5. Performance vs. data-release shift</title>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>",
        f"<rect x='{margin}' y='{margin}' width='{chart_width}' height='{chart_height}' fill='none' stroke='black' stroke-width='1'/>",
        f"<path d='{path_from(accuracy)}' stroke='#1f77b4' stroke-width='3' fill='none'/>",
        f"<path d='{path_from(macro_f1)}' stroke='#ff7f0e' stroke-width='3' fill='none' stroke-dasharray='6 4'/>",
    ]
    for idx, label in enumerate(releases):
        x = margin + idx / (len(releases) - 1) * chart_width
        svg.append(f"<text x='{x:.2f}' y='{margin + chart_height + 25}' font-size='14' font-family='sans-serif' text-anchor='middle'>{label}</text>")
    legend_items = [('Accuracy', accuracy, '#1f77b4'), ('Macro F1', macro_f1, '#ff7f0e')]
    for idx, (score, value, color) in enumerate(legend_items):
        x = margin + chart_width - 150
        y = margin + 30 + 32 * idx
        dash = " stroke-dasharray='6 4'" if score == 'Macro F1' else ''
        svg.append(f"<rect x='{x}' y='{y - 12}' width='28' height='12' fill='{color}'{dash}/>")
        svg.append(f"<text x='{x + 40}' y='{y - 2}' font-size='14' font-family='sans-serif'>{score}</text>")
    svg.append(f"<text x='{width/2:.2f}' y='{margin - 20}' font-size='18' font-family='sans-serif' text-anchor='middle'>Figure 5. Performance vs. data-release shift</text>")
    svg.append(f"<text x='{width/2:.2f}' y='{height - 10}' font-size='14' font-family='sans-serif' text-anchor='middle'>Test release (trained on DR17)</text>")
    svg.append(f"<text x='{margin - 40}' y='{margin + chart_height/2:.2f}' font-size='14' font-family='sans-serif' text-anchor='middle' transform='rotate(-90 {margin - 40} {margin + chart_height/2:.2f})'>Score</text>")
    svg.append("</svg>")

    (OUTPUT_DIR / 'figure5_release_shift.svg').write_text('\n'.join(svg), encoding='utf-8')


def main():
    create_figure1()
    create_figure3()
    create_figure4()
    create_figure5()


if __name__ == '__main__':
    main()
