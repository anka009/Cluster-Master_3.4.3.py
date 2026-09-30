import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Voronoi, ConvexHull

# ============================================================
# AT2 SPATIAL ANALYSIS – CLUSTER-MASTER 3.4.2
#
# Drei Modi:
#   🔬 Hauptanalyse  = Cluster-Master 3.4
#   🧭 EPS-Kalibrierung = objektiver EPS-Vorschlag pro Image/ROI
#   🐦 Colibri        = visuelle DBSCAN-Kontrolle
#
# Hauptanalyse und Colibri verwenden dieselbe DBSCAN-Engine.
# ============================================================

st.set_page_config(
    page_title="AT2 Spatial Analysis 3.4.2",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AT2 Spatial Analysis")
st.caption("Cluster-Master 3.4.2  |  EPS-Kalibrierung + Hauptanalyse + Colibri")

# ============================================================
# GEMEINSAME FUNKTIONEN
# ============================================================

def prepare_coordinates(image_df, calibration_mode, manual_pixel_um):
    """Einheitliche Koordinatenlogik für Hauptanalyse und Colibri."""
    if (
        calibration_mode == "auto"
        and "X_um" in image_df.columns
        and "Y_um" in image_df.columns
    ):
        coord = image_df[["X_um", "Y_um"]].dropna()
        if len(coord) > 0:
            return coord.to_numpy(dtype=float), "X_um / Y_um"

    if "X_pixel" in image_df.columns and "Y_pixel" in image_df.columns:
        pw = manual_pixel_um
        ph = manual_pixel_um

        if calibration_mode == "auto":
            if "PixelWidth_um" in image_df.columns:
                v = pd.to_numeric(
                    image_df["PixelWidth_um"], errors="coerce"
                ).dropna()
                v = v[v > 0]
                if len(v):
                    pw = float(v.median())

            if "PixelHeight_um" in image_df.columns:
                v = pd.to_numeric(
                    image_df["PixelHeight_um"], errors="coerce"
                ).dropna()
                v = v[v > 0]
                if len(v):
                    ph = float(v.median())

        coord = image_df[["X_pixel", "Y_pixel"]].dropna()
        xy_pixel = coord.to_numpy(dtype=float)

        xy_um = np.column_stack([
            xy_pixel[:, 0] * pw,
            xy_pixel[:, 1] * ph
        ])

        return xy_um, f"X_pixel / Y_pixel → µm ({pw:.4f} × {ph:.4f})"

    return np.empty((0, 2)), "Keine gültigen Koordinaten"


def run_dbscan(xy_um, eps_um, min_samples):
    """Gemeinsame DBSCAN-Engine für alle Modi."""
    if len(xy_um) < 2:
        return np.full(len(xy_um), -1, dtype=int)

    return DBSCAN(
        eps=float(eps_um),
        min_samples=int(min_samples)
    ).fit_predict(xy_um)


def recommend_eps_kdistance(xy_um, min_samples=3):
    """
    Ermittelt einen objektiven EPS-Vorschlag aus der k-distance-Kurve.

    Für min_samples=3 wird die Distanz zum 3. nächsten Punkt
    (einschließlich des Punktes selbst) verwendet. Die sortierte
    Distanzkurve wird über die maximale Abweichung von der
    Verbindungslinie zwischen Anfang und Ende ausgewertet (Knie/Elbow).

    Rückgabe: EPS in µm. Es werden keine künstlichen oberen Grenzen
    verwendet; die Skalierung ergibt sich ausschließlich aus den Daten.
    """
    n = len(xy_um)
    if n < min_samples or min_samples < 2:
        return np.nan

    try:
        nn = NearestNeighbors(
            n_neighbors=int(min_samples)
        )
        nn.fit(xy_um)
        distances, _ = nn.kneighbors(xy_um)

        k_distances = np.sort(
            distances[:, -1].astype(float)
        )

        if len(k_distances) == 0:
            return np.nan

        if len(k_distances) == 1:
            eps = k_distances[0]
        else:
            x = np.linspace(0.0, 1.0, len(k_distances))
            y = k_distances

            y_min = float(np.min(y))
            y_max = float(np.max(y))
            y_range = y_max - y_min

            if y_range <= 0:
                eps = float(np.median(y))
            else:
                y_norm = (y - y_min) / y_range
                line_y = y_norm[0] + (y_norm[-1] - y_norm[0]) * x
                distances_to_line = np.abs(y_norm - line_y)
                knee_index = int(np.argmax(distances_to_line))
                eps = float(y[knee_index])

        if not np.isfinite(eps) or eps <= 0:
            return np.nan

        return float(np.round(eps, 1))

    except Exception:
        return np.nan



def select_calibration_units(units, n_units):
    """Reproduzierbare, gleichmäßig verteilte Auswahl aus allen Einheiten."""
    if len(units) <= n_units:
        return units

    indices = np.linspace(
        0, len(units) - 1, n_units, dtype=int
    )
    return [units[i] for i in indices]


def cluster_metrics(xy_um, labels):
    cluster_ids = sorted(
        x for x in np.unique(labels) if x != -1
    )

    total_count = len(xy_um)
    clustered_count = int((labels != -1).sum())
    clustered_percent = (
        clustered_count / total_count * 100
        if total_count > 0 else np.nan
    )

    cluster_sizes = [
        int((labels == cid).sum())
        for cid in cluster_ids
    ]

    median_cluster_size = (
        float(np.median(cluster_sizes))
        if cluster_sizes else np.nan
    )

    return (
        cluster_ids,
        clustered_count,
        clustered_percent,
        cluster_sizes,
        median_cluster_size
    )


def empty_result(image_name, roi_id, roi_area_mm2, n_cells,
                 at2_per_mm2, pixel_width_um, pixel_height_um):
    return {
        "Image": image_name,
        "ROI_ID": roi_id,
        "ROI_Area_mm2": roi_area_mm2,
        "AT2_Count": n_cells,
        "AT2_per_mm2": at2_per_mm2,
        "Clustered_AT2_percent": np.nan,
        "Clusters_per_mm2": np.nan,
        "Median_AT2_per_Cluster": np.nan,
        "Median_Cluster_Area_um2": np.nan,
        "Median_Voronoi_Area_um2": np.nan,
        "Voronoi_CV": np.nan,
        "Voronoi_Mean_um2": np.nan,
        "Voronoi_SD_um2": np.nan,
        "Voronoi_Cutoff_um2": np.nan,
        "Voronoi_N_Total": 0,
        "Voronoi_N_Used": 0,
        "Voronoi_N_Excluded": 0,
        "Voronoi_Excluded_percent": np.nan,
        "Cluster_Count": 0,
        "PixelWidth_um": pixel_width_um,
        "PixelHeight_um": pixel_height_um
    }


def analyze_at2(
    image_df,
    eps_um,
    min_samples,
    calibration_mode,
    manual_pixel_um,
    voronoi_mode,
    manual_voronoi_area_um2
):
    image_name = str(image_df["Image"].iloc[0])
    roi_id = str(image_df["ROI_ID"].iloc[0])
    n_cells = len(image_df)
    roi_area_mm2 = float(image_df["ROI_Area_mm2"].iloc[0])

    pixel_width_um = manual_pixel_um
    pixel_height_um = manual_pixel_um

    if calibration_mode == "auto":
        if "PixelWidth_um" in image_df.columns:
            v = pd.to_numeric(
                image_df["PixelWidth_um"], errors="coerce"
            ).dropna()
            v = v[v > 0]
            if len(v):
                pixel_width_um = float(v.median())

        if "PixelHeight_um" in image_df.columns:
            v = pd.to_numeric(
                image_df["PixelHeight_um"], errors="coerce"
            ).dropna()
            v = v[v > 0]
            if len(v):
                pixel_height_um = float(v.median())

    at2_per_mm2 = (
        n_cells / roi_area_mm2
        if roi_area_mm2 > 0 else np.nan
    )

    xy_um, _ = prepare_coordinates(
        image_df,
        calibration_mode,
        manual_pixel_um
    )

    if len(xy_um) == 0:
        return empty_result(
            image_name, roi_id, roi_area_mm2, n_cells,
            at2_per_mm2, pixel_width_um, pixel_height_um
        )

    if len(xy_um) < 3:
        return {
            "Image": image_name,
            "ROI_ID": roi_id,
            "ROI_Area_mm2": roi_area_mm2,
            "AT2_Count": n_cells,
            "AT2_per_mm2": at2_per_mm2,
            "Clustered_AT2_percent": 0,
            "Clusters_per_mm2": 0,
            "Median_AT2_per_Cluster": np.nan,
            "Median_Cluster_Area_um2": np.nan,
            "Median_Voronoi_Area_um2": np.nan,
            "Voronoi_CV": np.nan,
            "Voronoi_Mean_um2": np.nan,
            "Voronoi_SD_um2": np.nan,
            "Voronoi_Cutoff_um2": np.nan,
            "Voronoi_N_Total": len(xy_um),
            "Voronoi_N_Used": 0,
            "Voronoi_N_Excluded": 0,
            "Voronoi_Excluded_percent": 0.0,
            "Cluster_Count": 0,
            "PixelWidth_um": pixel_width_um,
            "PixelHeight_um": pixel_height_um
        }

    # ========================================================
    # DBSCAN – gemeinsame Engine
    # ========================================================
    labels = run_dbscan(
        xy_um,
        eps_um,
        min_samples
    )

    (
        cluster_ids,
        clustered_count,
        clustered_percent,
        cluster_sizes,
        median_cluster_size
    ) = cluster_metrics(xy_um, labels)

    cluster_count = len(cluster_ids)

    clusters_per_mm2 = (
        cluster_count / roi_area_mm2
        if roi_area_mm2 > 0 else np.nan
    )

    cluster_areas = []

    for cid in cluster_ids:
        points = xy_um[labels == cid]

        if len(points) >= 3:
            try:
                cluster_areas.append(
                    float(ConvexHull(points).volume)
                )
            except Exception:
                pass

    median_cluster_area = (
        float(np.median(cluster_areas))
        if cluster_areas else np.nan
    )

    # ========================================================
    # VORONOI – unverändert aus Cluster-Master 3.4
    # ========================================================
    raw_areas = []

    try:
        vor = Voronoi(xy_um)

        for region_index in vor.point_region:
            region = vor.regions[region_index]

            if len(region) == 0 or -1 in region:
                raw_areas.append(np.nan)
                continue

            vertices = vor.vertices[region]

            if len(vertices) < 3:
                raw_areas.append(np.nan)
                continue

            x = vertices[:, 0]
            y = vertices[:, 1]

            area = 0.5 * abs(
                np.dot(x, np.roll(y, 1))
                -
                np.dot(y, np.roll(x, 1))
            )

            raw_areas.append(
                float(area)
                if np.isfinite(area) and area > 0
                else np.nan
            )

        raw_areas = np.asarray(raw_areas, dtype=float)

        finite_areas = raw_areas[
            np.isfinite(raw_areas) &
            (raw_areas > 0)
        ]

        voronoi_n_total = len(finite_areas)

        if len(finite_areas) >= 2:
            voronoi_mean = float(np.mean(finite_areas))
            voronoi_sd = float(
                np.std(finite_areas, ddof=1)
            )
        elif len(finite_areas) == 1:
            voronoi_mean = float(finite_areas[0])
            voronoi_sd = 0.0
        else:
            voronoi_mean = np.nan
            voronoi_sd = np.nan

        if voronoi_mode == "auto":
            voronoi_cutoff = (
                voronoi_mean +
                2.0 * voronoi_sd
                if np.isfinite(voronoi_mean)
                and np.isfinite(voronoi_sd)
                else np.nan
            )
        else:
            voronoi_cutoff = float(
                manual_voronoi_area_um2
            )

        if np.isfinite(voronoi_cutoff):
            used_areas = finite_areas[
                finite_areas <= voronoi_cutoff
            ]
        else:
            used_areas = np.array([], dtype=float)

        voronoi_n_used = len(used_areas)
        voronoi_n_excluded = (
            voronoi_n_total -
            voronoi_n_used
        )

        voronoi_excluded_percent = (
            voronoi_n_excluded /
            voronoi_n_total *
            100
            if voronoi_n_total > 0
            else np.nan
        )

        median_voronoi_area = (
            float(np.median(used_areas))
            if voronoi_n_used > 0
            else np.nan
        )

        if voronoi_n_used >= 3:
            used_mean = float(
                np.mean(used_areas)
            )
            used_sd = float(
                np.std(used_areas, ddof=1)
            )
            voronoi_cv = (
                used_sd / used_mean
                if used_mean > 0
                else np.nan
            )
        else:
            voronoi_cv = np.nan

    except Exception:
        median_voronoi_area = np.nan
        voronoi_cv = np.nan
        voronoi_mean = np.nan
        voronoi_sd = np.nan
        voronoi_cutoff = np.nan
        voronoi_n_total = 0
        voronoi_n_used = 0
        voronoi_n_excluded = 0
        voronoi_excluded_percent = np.nan

    return {
        "Image": image_name,
        "ROI_ID": roi_id,
        "ROI_Area_mm2": roi_area_mm2,
        "AT2_Count": n_cells,
        "AT2_per_mm2": at2_per_mm2,
        "Clustered_AT2_percent": clustered_percent,
        "Clusters_per_mm2": clusters_per_mm2,
        "Median_AT2_per_Cluster": median_cluster_size,
        "Median_Cluster_Area_um2": median_cluster_area,
        "Median_Voronoi_Area_um2": median_voronoi_area,
        "Voronoi_CV": voronoi_cv,
        "Voronoi_Mean_um2": voronoi_mean,
        "Voronoi_SD_um2": voronoi_sd,
        "Voronoi_Cutoff_um2": voronoi_cutoff,
        "Voronoi_N_Total": voronoi_n_total,
        "Voronoi_N_Used": voronoi_n_used,
        "Voronoi_N_Excluded": voronoi_n_excluded,
        "Voronoi_Excluded_percent": voronoi_excluded_percent,
        "Cluster_Count": cluster_count,
        "PixelWidth_um": pixel_width_um,
        "PixelHeight_um": pixel_height_um
    }


# ============================================================
# SIDEBAR – MODUS
# ============================================================

mode = st.sidebar.radio(
    "Werkzeugmodus",
    [
        "🔬 Hauptanalyse",
        "🧭 EPS-Kalibrierung",
        "🐦 Colibri"
    ],
    index=0
)

st.sidebar.markdown("---")

# ============================================================
# GEMEINSAME CSV
# ============================================================

uploaded_file = st.file_uploader(
    "📂 QuPath MASTER-CSV laden",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Bitte deine **Positive_Centroids_MASTER.csv** laden."
    )
    st.stop()

try:
    df = pd.read_csv(
        uploaded_file,
        sep=None,
        engine="python"
    )
except Exception as e:
    st.error(f"CSV konnte nicht gelesen werden:\n{e}")
    st.stop()

df.columns = df.columns.astype(str).str.strip()

required_basic = [
    "Image",
    "ROI_ID",
    "ROI_Area_mm2"
]

missing_basic = [
    c for c in required_basic
    if c not in df.columns
]

if missing_basic:
    st.error(
        "Diese Spalten fehlen:\n\n"
        + "\n".join(missing_basic)
    )
    st.stop()

numeric_columns = [
    "ROI_Area_pixel2",
    "ROI_Area_um2",
    "ROI_Area_mm2",
    "PixelWidth_um",
    "PixelHeight_um",
    "Positive_Count",
    "X_pixel",
    "Y_pixel",
    "X_um",
    "Y_um"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# ============================================================
# GEMEINSAME KALIBRIERUNG
# ============================================================

st.sidebar.subheader("📏 Kalibrierung")

calibration_mode_label = st.sidebar.radio(
    "Koordinaten/Kalibrierung",
    [
        "Automatisch aus QuPath",
        "Manuell"
    ],
    index=0
)

calibration_mode = (
    "auto"
    if calibration_mode_label ==
    "Automatisch aus QuPath"
    else "manual"
)

manual_pixel_um = st.sidebar.number_input(
    "Pixelgröße (µm / Pixel)",
    min_value=0.0001,
    max_value=10.0,
    value=0.2128,
    step=0.001,
    format="%.4f"
)

# ============================================================
# EPS-KALIBRIERUNG
# ============================================================

if mode == "🧭 EPS-Kalibrierung":

    st.header("🧭 EPS-Kalibrierung")
    st.markdown(
        """
        **Freie Auswahl von Image / ROI-Einheiten**

        Aus der gesamten MASTER-CSV werden nach und nach beliebige Image / ROI-Einheiten
        ausgewählt. Für jede ausgewählte Einheit wird ein EPS-Vorschlag berechnet.
        Die Auswahl kann jederzeit um weitere, beliebig verteilte Einheiten ergänzt werden.
        Der Mittelwert dient ausschließlich als Orientierung. Die endgültige Wahl erfolgt
        anschließend visuell mit **Colibri**.
        """
    )

    units = [
        (str(image), str(roi))
        for image, image_group
        in df.groupby("Image", sort=True)
        for roi, roi_group in image_group.groupby("ROI_ID", sort=True)
    ]

    total_units = len(units)

    if total_units == 0:
        st.error("Keine Image/ROI-Einheiten gefunden.")
        st.stop()

    # Die Auswahl gehört zum aktuell geladenen Datensatz.
    units_signature = tuple(units)
    if st.session_state.get("eps_calibration_units_signature") != units_signature:
        st.session_state.eps_calibration_units_signature = units_signature
        st.session_state.eps_calibration_selected = []

    if "eps_calibration_selected" not in st.session_state:
        st.session_state.eps_calibration_selected = []

    c1, c2 = st.columns([3, 1])

    with c1:
        selected_candidate = st.selectbox(
            "Image / ROI auswählen",
            options=units,
            format_func=lambda u: f"Image {u[0]}  |  ROI {u[1]}",
            key="eps_calibration_candidate"
        )

    with c2:
        st.write("")
        st.write("")
        add_clicked = st.button(
            "➕ Zur Auswahl hinzufügen",
            width="stretch"
        )

    if add_clicked:
        if selected_candidate in st.session_state.eps_calibration_selected:
            st.warning(
                f"⚠️ Bild/ROI **Image {selected_candidate[0]} | ROI {selected_candidate[1]}** "
                "steht schon in der Auswahl."
            )
        else:
            st.session_state.eps_calibration_selected.append(
                selected_candidate
            )

    c1, c2 = st.columns([3, 1])

    with c1:
        calibration_min_samples = st.number_input(
            "min_samples",
            min_value=2,
            value=3,
            step=1,
            help="Wert für die EPS-Empfehlung. Die Hauptanalyse/Colibri bleibt davon unabhängig einstellbar."
        )

    with c2:
        if st.button("🗑️ Auswahl zurücksetzen"):
            st.session_state.eps_calibration_selected = []
            st.rerun()

    selected_units = st.session_state.eps_calibration_selected

    st.caption(
        f"{len(selected_units)} von {total_units} Image/ROI-Einheiten ausgewählt. "
        "Weitere Einheiten können jederzeit ergänzt werden."
    )

    if not selected_units:
        st.info(
            "Noch keine Einheit ausgewählt. Wähle oben beliebige Bilder/ROIs aus und "
            "füge sie nacheinander zur Kalibrierung hinzu."
        )
    else:
        calibration_rows = []
        progress = st.progress(0)

        for i, (current_image, current_roi) in enumerate(selected_units):
            roi_df = df[
                (df["Image"].astype(str) == current_image) &
                (df["ROI_ID"].astype(str) == current_roi)
            ].copy()

            xy, coordinate_mode = prepare_coordinates(
                roi_df,
                calibration_mode,
                manual_pixel_um
            )

            recommended_eps = recommend_eps_kdistance(
                xy,
                int(calibration_min_samples)
            )

            calibration_rows.append({
                "Nr.": i + 1,
                "Image": current_image,
                "ROI_ID": current_roi,
                "AT2": len(xy),
                "EPS-Vorschlag (µm)": recommended_eps,
                "Koordinaten": coordinate_mode
            })

            progress.progress(
                int((i + 1) / len(selected_units) * 100)
            )

        progress.empty()

        calibration_df = pd.DataFrame(calibration_rows)

        valid_eps = calibration_df[
            "EPS-Vorschlag (µm)"
        ].dropna()

        st.subheader("📋 EPS-Vorschläge pro ausgewähltem Image / ROI")
        st.dataframe(
            calibration_df,
            width="stretch",
            hide_index=True
        )

        if len(valid_eps) > 0:
            mean_eps = float(valid_eps.mean())
            st.metric(
                "Mittelwert der EPS-Vorschläge",
                f"{mean_eps:.1f} µm"
            )
        else:
            st.warning(
                "Für die bisher ausgewählten Einheiten konnte kein EPS-Vorschlag "
                "berechnet werden."
            )

    st.info(
        "Der Mittelwert ist nur ein Orientierungswert. Er wird nicht automatisch übernommen. "
        "Wähle so lange weitere, beliebige Image/ROI-Einheiten aus, bis du die Variabilität "
        "des Datensatzes ausreichend erfasst hast. Anschließend in **Colibri** Werte in der "
        "Umgebung des Orientierungswertes visuell prüfen und den endgültigen EPS-Wert für die "
        "Hauptanalyse selbst festlegen."
    )

    st.markdown(
        """
        **Prinzip:** Die EPS-Empfehlung basiert auf der k-distance-Kurve der
        AT2-Zellzentren. Der berechnete Kniepunkt liefert einen objektiven
        Startwert. Die Auswahl der Bilder und die wissenschaftliche Entscheidung
        über den endgültig verwendeten EPS-Wert bleiben beim Anwender.
        """
    )

# ============================================================
# COLIBRI
# ============================================================

elif mode == "🐦 Colibri":

    st.header("🐦 Colibri")
    st.markdown(
        """
        **DBSCAN-Kalibrierung für AT2-Zellcluster**

        Bild für Bild durch die komplette MASTER-CSV.
        `eps` und `min_samples` können visuell geprüft werden.
        """
    )

    units = [
        (str(image), str(roi))
        for image, image_group
        in df.groupby("Image", sort=True)
        for roi, roi_group
        in image_group.groupby("ROI_ID", sort=True)
    ]

    total_units = len(units)

    if total_units == 0:
        st.error("Keine Image/ROI-Einheiten gefunden.")
        st.stop()

    if "colibri_index" not in st.session_state:
        st.session_state.colibri_index = 0

    if "colibri_eps" not in st.session_state:
        st.session_state.colibri_eps = 12.0

    if "colibri_min_samples" not in st.session_state:
        st.session_state.colibri_min_samples = 3

    st.sidebar.subheader("🔵 DBSCAN")

    # ------------------------------------------------------------
    # EPS / min_samples: Slider + direkte Zahleneingabe
    # Beide Bedienelemente bleiben synchron.
    # ------------------------------------------------------------
    if "colibri_eps_input" not in st.session_state:
        st.session_state.colibri_eps_input = float(
            st.session_state.colibri_eps
        )
    if "colibri_eps_slider" not in st.session_state:
        st.session_state.colibri_eps_slider = float(
            st.session_state.colibri_eps
        )

    if "colibri_min_samples_input" not in st.session_state:
        st.session_state.colibri_min_samples_input = int(
            st.session_state.colibri_min_samples
        )
    if "colibri_min_samples_slider" not in st.session_state:
        st.session_state.colibri_min_samples_slider = int(
            st.session_state.colibri_min_samples
        )

    def sync_eps_from_slider():
        value = float(st.session_state.colibri_eps_slider)
        st.session_state.colibri_eps_input = value
        st.session_state.colibri_eps = value

    def sync_eps_from_input():
        value = float(st.session_state.colibri_eps_input)
        st.session_state.colibri_eps_slider = value
        st.session_state.colibri_eps = value

    def sync_min_samples_from_slider():
        value = int(st.session_state.colibri_min_samples_slider)
        st.session_state.colibri_min_samples_input = value
        st.session_state.colibri_min_samples = value

    def sync_min_samples_from_input():
        value = int(st.session_state.colibri_min_samples_input)
        st.session_state.colibri_min_samples_slider = value
        st.session_state.colibri_min_samples = value

    eps_c1, eps_c2 = st.sidebar.columns([2.2, 1])

    with eps_c1:
        st.slider(
            "EPS – maximaler Abstand (µm)",
            min_value=1.0,
            max_value=100.0,
            step=1.0,
            key="colibri_eps_slider",
            on_change=sync_eps_from_slider
        )

    with eps_c2:
        st.number_input(
            "EPS",
            min_value=1.0,
            max_value=100.0,
            step=1.0,
            format="%.0f",
            key="colibri_eps_input",
            on_change=sync_eps_from_input
        )

    min_c1, min_c2 = st.sidebar.columns([2.2, 1])

    with min_c1:
        st.slider(
            "min_samples",
            min_value=2,
            max_value=10,
            step=1,
            key="colibri_min_samples_slider",
            on_change=sync_min_samples_from_slider
        )

    with min_c2:
        st.number_input(
            "min_samples",
            min_value=2,
            max_value=10,
            step=1,
            format="%d",
            key="colibri_min_samples_input",
            on_change=sync_min_samples_from_input
        )

    eps_um = float(st.session_state.colibri_eps)
    min_samples = int(st.session_state.colibri_min_samples)

    current_index = min(
        st.session_state.colibri_index,
        total_units - 1
    )

    current_image, current_roi = units[current_index]

    roi_df = df[
        (df["Image"].astype(str) == current_image) &
        (df["ROI_ID"].astype(str) == current_roi)
    ].copy()

    xy, coordinate_mode = prepare_coordinates(
        roi_df,
        calibration_mode,
        manual_pixel_um
    )

    labels = run_dbscan(
        xy,
        eps_um,
        min_samples
    )

    (
        cluster_ids,
        clustered_count,
        clustered_percent,
        cluster_sizes,
        median_cluster_size
    ) = cluster_metrics(
        xy,
        labels
    )

    st.subheader(
        f"🔬 {current_image}  |  ROI {current_roi}"
    )

    st.progress(
        (current_index + 1) / total_units
    )

    st.caption(
        f"Bild/ROI {current_index + 1} von {total_units}"
        f" | Koordinaten: {coordinate_mode}"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("AT2 gesamt", len(xy))
    c2.metric("AT2 im Cluster", clustered_count)
    c3.metric(
        "Clustered AT2",
        f"{clustered_percent:.1f} %"
    )
    c4.metric("Cluster", len(cluster_ids))
    c5.metric(
        "Median AT2 / Cluster",
        (
            f"{median_cluster_size:.1f}"
            if not np.isnan(median_cluster_size)
            else "—"
        )
    )

    st.markdown("---")

    fig, ax = plt.subplots(figsize=(11, 8))

    non_clustered = labels == -1

    if np.any(non_clustered):
        ax.scatter(
            xy[non_clustered, 0],
            xy[non_clustered, 1],
            s=25,
            alpha=0.45,
            label="nicht geclustert"
        )

    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        points = xy[mask]

        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=55,
            label=f"Cluster {cluster_id + 1}"
        )

        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points = np.vstack(
                    [hull_points, hull_points[0]]
                )

                ax.plot(
                    hull_points[:, 0],
                    hull_points[:, 1],
                    linewidth=1.5
                )
            except Exception:
                pass

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")

    # QuPath/ImageJ image coordinates have their origin at the top-left
    # (Y increases downward). Matplotlib uses a bottom-left origin by
    # default, so invert Y to make the Colibri plot match the source image.
    ax.invert_yaxis()

    ax.set_title(
        f"DBSCAN: EPS = {eps_um:.0f} µm | "
        f"min_samples = {min_samples}"
    )
    ax.set_aspect("equal", adjustable="box")

    if len(cluster_ids) <= 15:
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

    # ------------------------------------------------------------
    # Colibri-Vollbild: derselbe Plot, aber separat und groß.
    # Der normale Plot bleibt in der Hauptansicht sichtbar.
    # ------------------------------------------------------------
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    if "colibri_fullscreen" not in st.session_state:
        st.session_state.colibri_fullscreen = False

    if st.button(
        "⛶ Plot im Vollbild öffnen",
        key="colibri_fullscreen_button",
        width="stretch"
    ):
        st.session_state.colibri_fullscreen = True
        st.rerun()

    if st.session_state.colibri_fullscreen:
        @st.dialog("🐦 Colibri – Plot im Vollbild", width="large")
        def show_colibri_fullscreen():
            # Dialog deutlich breiter machen; der Plot soll den verfügbaren
            # Bildschirmbereich tatsächlich ausnutzen.
            st.markdown(
                """
                <style>
                [data-testid="stDialog"] > div {
                    width: 94vw !important;
                    max-width: 94vw !important;
                }
                [data-testid="stDialog"] [role="dialog"] {
                    max-height: 94vh !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            fig_full, ax_full = plt.subplots(figsize=(20, 13), dpi=120)

            if np.any(non_clustered):
                ax_full.scatter(
                    xy[non_clustered, 0],
                    xy[non_clustered, 1],
                    s=35,
                    alpha=0.45,
                    label="nicht geclustert"
                )

            for cluster_id in cluster_ids:
                mask = labels == cluster_id
                points = xy[mask]

                ax_full.scatter(
                    points[:, 0],
                    points[:, 1],
                    s=75,
                    label=f"Cluster {cluster_id + 1}"
                )

                if len(points) >= 3:
                    try:
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        hull_points = np.vstack(
                            [hull_points, hull_points[0]]
                        )
                        ax_full.plot(
                            hull_points[:, 0],
                            hull_points[:, 1],
                            linewidth=1.8
                        )
                    except Exception:
                        pass

            ax_full.set_xlabel("X (µm)")
            ax_full.set_ylabel("Y (µm)")

            # Same orientation correction as the normal Colibri plot:
            # image coordinates use a top-left origin, Matplotlib a bottom-left origin.
            ax_full.invert_yaxis()

            ax_full.set_title(
                f"DBSCAN: EPS = {eps_um:.0f} µm | "
                f"min_samples = {min_samples}"
            )
            ax_full.set_aspect("equal", adjustable="box")

            if len(cluster_ids) <= 15:
                ax_full.legend(
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left"
                )

            st.pyplot(fig_full, width="stretch")
            plt.close(fig_full)

            if st.button(
                "↩ Vollbild schließen",
                key="colibri_fullscreen_close",
                width="stretch"
            ):
                st.session_state.colibri_fullscreen = False
                st.rerun()

        show_colibri_fullscreen()

    st.markdown("---")
    st.subheader("🔎 Erkannte Cluster")

    if cluster_ids:
        cluster_table = []

        for cluster_id in cluster_ids:
            mask = labels == cluster_id
            points = xy[mask]
            area = np.nan

            if len(points) >= 3:
                try:
                    area = float(
                        ConvexHull(points).volume
                    )
                except Exception:
                    pass

            cluster_table.append({
                "Cluster": cluster_id + 1,
                "AT2": int(mask.sum()),
                "Fläche_µm²": area
            })

        st.dataframe(
            pd.DataFrame(cluster_table).round(2),
            width="stretch",
            hide_index=True
        )
    else:
        st.info(
            "Bei diesen Parametern wurde kein Cluster erkannt."
        )

    st.markdown("---")
    st.subheader("🐦 Colibri-Rundflug")

    nav1, nav2, nav3, nav4 = st.columns(4)

    with nav1:
        if st.button(
            "⬅️ Vorheriges",
            width="stretch",
            disabled=(current_index == 0)
        ):
            st.session_state.colibri_index -= 1
            st.rerun()

    with nav2:
        if st.button(
            "➡️ Nächstes Bild",
            width="stretch",
            disabled=(current_index >= total_units - 1)
        ):
            st.session_state.colibri_index += 1
            st.rerun()

    with nav3:
        if st.button(
            "⏮️ Zum Anfang",
            width="stretch"
        ):
            st.session_state.colibri_index = 0
            st.rerun()

    with nav4:
        if st.button(
            "⏭️ Zum letzten Bild",
            width="stretch"
        ):
            st.session_state.colibri_index = total_units - 1
            st.rerun()

    jump_options = [
        f"{i + 1}: {img} | ROI {roi}"
        for i, (img, roi) in enumerate(units)
    ]

    selected_jump = st.selectbox(
        "Direkt zu Image / ROI",
        jump_options,
        index=current_index
    )

    jump_index = jump_options.index(selected_jump)

    if jump_index != current_index:
        st.session_state.colibri_index = jump_index
        st.rerun()

    st.markdown("---")

    st.info(
        """
        💡 **Colibri-Regel**

        Colibri dient zur visuellen Kontrolle von `eps` und
        `min_samples`. Nach Festlegung eines Wertes sollte dieser
        für die komplette Serie unverändert verwendet werden.

        Der endgültig gewählte EPS-Wert wird anschließend im
        Hauptanalysemodus **manuell eingetragen**.
        """
    )

    st.caption(
        f"🐦 Aktuelle visuelle Prüfung: "
        f"eps = {eps_um:.1f} µm | "
        f"min_samples = {min_samples}"
    )

# ============================================================
# HAUPTANALYSE
# ============================================================

else:

    st.header("🔬 Cluster-Master 3.4.2")

    st.markdown(
        """
        **Surfactant-C+ AT2-Zellen – räumliche Analyse**

        Jede Kombination aus **Image + ROI_ID** wird als eigene
        Analyseeinheit behandelt.
        """
    )

    st.sidebar.subheader("🔵 Clusterdefinition")

    default_eps = 10.0
    default_min_samples = 3

    eps_um = st.sidebar.number_input(
        "Clusterabstand eps (µm)",
        min_value=1.0,
        max_value=1000.0,
        value=default_eps,
        step=1.0,
        help=(
            "Maximaler Abstand zwischen AT2-Zellen "
            "innerhalb eines DBSCAN-Clusters."
        )
    )

    min_samples = st.sidebar.number_input(
        "Minimale AT2-Zellen pro Cluster",
        min_value=2,
        max_value=50,
        value=default_min_samples,
        step=1
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔷 Voronoi")

    voronoi_mode_label = st.sidebar.radio(
        "Maximale Voronoi-Fläche",
        [
            "Automatisch: Mittelwert + 2 SD",
            "Benutzerdefiniert"
        ],
        index=0,
        help=(
            "Die Grenze betrifft ausschließlich die "
            "Voronoi-Auswertung."
        )
    )

    voronoi_mode = (
        "auto"
        if voronoi_mode_label ==
        "Automatisch: Mittelwert + 2 SD"
        else "manual"
    )

    manual_voronoi_area_um2 = st.sidebar.number_input(
        "Eigenes Voronoi-Maximum (µm²)",
        min_value=1.0,
        max_value=1e12,
        value=2000.0,
        step=100.0,
        format="%.0f",
        disabled=(voronoi_mode != "manual")
    )

    st.markdown("---")

    results = []

    grouped = df.groupby(
        ["Image", "ROI_ID"],
        sort=True
    )

    total = len(grouped)

    progress = st.progress(0)

    for i, (_, image_df) in enumerate(grouped):
        results.append(
            analyze_at2(
                image_df,
                eps_um,
                min_samples,
                calibration_mode,
                manual_pixel_um,
                voronoi_mode,
                manual_voronoi_area_um2
            )
        )

        progress.progress(
            int((i + 1) / total * 100)
        )

    progress.empty()

    results_df = pd.DataFrame(results)

    desired_columns = [
        "Image",
        "ROI_ID",
        "ROI_Area_mm2",
        "AT2_Count",
        "AT2_per_mm2",
        "Clustered_AT2_percent",
        "Cluster_Count",
        "Clusters_per_mm2",
        "Median_AT2_per_Cluster",
        "Median_Cluster_Area_um2",
        "Median_Voronoi_Area_um2",
        "Voronoi_CV",
        "Voronoi_Mean_um2",
        "Voronoi_SD_um2",
        "Voronoi_Cutoff_um2",
        "Voronoi_N_Total",
        "Voronoi_N_Used",
        "Voronoi_N_Excluded",
        "Voronoi_Excluded_percent",
        "PixelWidth_um",
        "PixelHeight_um"
    ]

    results_df = results_df[desired_columns]

    st.success(
        f"{len(results_df)} Image/ROI-Einheiten analysiert."
    )

    st.subheader("🔬 AT2-Ergebnisse")

    display_df = results_df.copy()

    for col in [
        "ROI_Area_mm2",
        "AT2_per_mm2",
        "Clustered_AT2_percent",
        "Clusters_per_mm2",
        "Median_AT2_per_Cluster",
        "Median_Cluster_Area_um2",
        "Median_Voronoi_Area_um2",
        "Voronoi_CV",
        "Voronoi_Mean_um2",
        "Voronoi_SD_um2",
        "Voronoi_Cutoff_um2",
        "Voronoi_Excluded_percent"
    ]:
        display_df[col] = display_df[col].round(3)

    st.dataframe(
        display_df,
        width="stretch",
        height=650
    )

    st.subheader("📊 Übersicht")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Image / ROI", len(results_df))
    c2.metric(
        "AT2 gesamt",
        int(results_df["AT2_Count"].sum())
    )
    c3.metric(
        "Ø AT2/mm²",
        f"{results_df['AT2_per_mm2'].mean():.1f}"
    )
    c4.metric(
        "Ø Clustered AT2",
        f"{results_df['Clustered_AT2_percent'].mean():.1f}%"
    )
    c5.metric(
        "Ø Cluster/mm²",
        f"{results_df['Clusters_per_mm2'].mean():.2f}"
    )
    c6.metric(
        "Ø Voronoi CV",
        f"{results_df['Voronoi_CV'].mean():.2f}"
    )

    st.markdown("---")
    st.subheader("📥 Ergebnisse speichern")

    csv_output = results_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 AT2-Ergebnisse als CSV speichern",
        data=csv_output,
        file_name="AT2_spatial_results.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("ℹ️ Parameter")

    st.markdown(
        """
        **AT2/mm²**

        Anzahl Surfactant-C-positiver AT2-Zellen pro mm²
        analysierter ROI-Fläche.

        **Clustered AT2 (%)**

        Prozentualer Anteil der AT2-Zellen, die DBSCAN
        einem Cluster zuordnet.

        **Cluster Count**

        Anzahl der erkannten AT2-Cluster.

        **Cluster/mm²**

        Anzahl der AT2-Cluster pro mm².

        **Median AT2/Cluster**

        Median der AT2-Zellzahl innerhalb der erkannten Cluster.

        **Median Clusterfläche (µm²)**

        Median der Fläche der konvexen Hülle der AT2-Zellzentren.

        **Median Voronoi Area (µm²)**

        Median der endlichen Voronoi-Flächen nach Anwendung
        der Voronoi-Ausreißergrenze.

        **Voronoi CV**

        Variationskoeffizient der verwendeten Voronoi-Flächen.

        **Voronoi-Ausreißergrenze**

        Automatisch: **Mittelwert + 2 SD** der endlichen
        Voronoi-Flächen der jeweiligen Image/ROI-Einheit.

        Benutzerdefiniert: frei wählbares maximales Voronoi-Flächenmaß.

        **WICHTIG:** Die Voronoi-Ausreißergrenze betrifft
        ausschließlich die Voronoi-Auswertung. Keine AT2-Zelle
        wird aus AT2_Count, AT2/mm², DBSCAN, Cluster Count,
        Clustergröße oder Clusterfläche entfernt.

        **Kalibrierung**

        Wenn QuPath `X_um/Y_um` liefert, werden diese Koordinaten
        direkt verwendet. Wenn nur `X_pixel/Y_pixel` vorhanden
        sind, werden sie mit der Pixelkalibrierung in µm
        umgerechnet.

        Der `eps`-Wert von DBSCAN wird immer in **µm** angegeben.

        **EPS-Kalibrierung**

        Die EPS-Kalibrierung liefert aus einer repräsentativen Auswahl
        der Image/ROI-Einheiten einen objektiven EPS-Vorschlag pro
        Einheit und dessen Mittelwert. Der Mittelwert ist nur ein
        Orientierungswert. Die endgültige EPS-Wahl erfolgt nach visueller
        Prüfung mit Colibri und wird anschließend manuell in der
        Hauptanalyse eingetragen.
        """
    )
