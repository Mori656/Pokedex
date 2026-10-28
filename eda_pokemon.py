"""
Projekt: Klasyfikacja Pokemonów I generacji (ResNet-18)
Skrypt: eda_pokemon.py - Eksploracyjna Analiza Danych (EDA)
Autor: Marcel Nosek
Akademia Tarnowska, Katedra Informatyki (2026)

Oczekiwana struktura folderów:
    katalog_z_projektem/
    │
    ├── eda_pokemon.py
    │
    └── PokemonData/
        ├── Bulbasaur/
        │   ├── 001.jpg
        │   └── 002.png
        ├── Charmander/
        │   ├── 001.jpg
        │   └── ...
        ├── Pikachu/
        │   └── ...
        └── ... (pozostałe 149 folderów Pokémonów)

Uruchomienie:
    pip install numpy matplotlib pillow scikit-learn
    python eda_pokemon.py

Wykresy i tabela CSV zostaną zapisane w folderze './wykresy_eda/'.
"""

import sys
import os
import csv
import hashlib
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ==============================================================================
# 1. KONFIGURACJA ŚCIEŻEK
# ==============================================================================
# Jeśli podano ścieżkę w terminalu (np. python eda_pokemon.py "C:/dane/PokemonData"),
# użyj jej. W przeciwnym razie domyślnie szukaj folderu ./PokemonData
if len(sys.argv) > 1:
    POKEMON_DATA_DIR = Path(sys.argv[1])
else:
    POKEMON_DATA_DIR = Path("C:\\Users\\marce\\.cache\\kagglehub\\datasets\\adamstanley537\\pokemon-gen-1-image-dataset-27000-images\\versions\\1\\PokemonData")

OUTPUT_DIR = Path("./wykresy_eda")
TARGET_SIZE = (224, 224)  # Docelowy wymiar wejściowy dla ResNet-18 (szerokość, wysokość)
MIN_SIDE_FOR_OUTLIER = 64
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MAX_TSNE_PER_CLASS = 6     # Maksymalna liczba próbek z jednej klasy do szybkiego t-SNE
MAX_TSNE_SAMPLES = 900     # Łączny limit próbek do t-SNE (by nie obciążać pamięci RAM)


# ==============================================================================
# 2. FUNKCJE POMOCNICZE: LETTERBOX PADDING I DESKRYPTORY
# ==============================================================================
def letterbox(image, size=TARGET_SIZE, fill=(0, 0, 0)):
    """
    Zmienia rozmiar obrazu z zachowaniem proporcji (aspect ratio)
    i dopełnia brakujące miejsce czarnym tłem (paddingiem) do wymiaru 224x224.
    Chroni sylwetkę Pokémona przed deformacją.
    """
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, fill)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def extract_color_features(image, bins=8):
    """
    Wyciąga znormalizowany histogram barw RGB ze znormalizowanego obrazu.
    Pozwala na analizę przestrzeni cech (PCA / t-SNE) bez ładowania ciężkich modeli.
    """
    arr = np.asarray(letterbox(image).resize((64, 64)), dtype=np.uint8)
    features = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        features.extend(hist / max(hist.sum(), 1))
    return np.asarray(features, dtype=np.float32)


# ==============================================================================
# 3. GŁÓWNA PROCEDURA ANALIZY
# ==============================================================================
def main():
    print(f"--> Sprawdzam ścieżkę do danych: {POKEMON_DATA_DIR.resolve()}")

    # Sprawdzenie czy folder PokemonData istnieje; jeśli użytkownik uruchomił
    # skrypt z poziomu wnętrza folderu PokemonData, automatycznie to wykrywamy.
    data_dir = POKEMON_DATA_DIR
    if not data_dir.is_dir():
        if Path(".").is_dir() and len([p for p in Path(".").iterdir() if p.is_dir() and p.name != "wykresy_eda"]) >= 20:
            print("--> Wykryto foldery Pokémonów bezpośrednio w bieżącym katalogu.")
            data_dir = Path(".")
        else:
            print(f"\nBŁĄD: Nie znaleziono folderu: {data_dir.resolve()}")
            print("Upewnij się, że folder 'PokemonData' znajduje się obok tego skryptu,")
            print("lub podaj ścieżkę jako argument: python eda_pokemon.py \"twoja/sciezka/do/PokemonData\"")
            sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Pobieramy podfoldery każdego z Pokémonów
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name != "wykresy_eda"])
    if not class_dirs:
        print(f"BŁĄD: Folder {data_dir.resolve()} nie zawiera podfolderów z klasami Pokémonów.")
        sys.exit(1)

    print(f"--> Znaleziono {len(class_dirs)} podfolderów Pokémonów.")
    print("--> Trwa skanowanie i analiza jakości zdjęć...")

    counts = Counter()
    sizes = []
    ratios = []
    unreadable = []
    non_rgb = []
    small_images = []
    seen_hashes = {}
    duplicate_files = []
    sample_images = []
    features = []
    feature_labels = []

    for class_id, class_dir in enumerate(class_dirs):
        pokemon_name = class_dir.name
        files = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS])
        counts[pokemon_name] = len(files)
        tsne_count_class = 0

        for path in files:
            try:
                # Weryfikacja integralności pliku
                with Image.open(path) as check:
                    check.verify()

                # Ponowne otwarcie do odczytu danych
                with Image.open(path) as im:
                    w, h = im.size
                    mode = im.mode
                    if mode != "RGB":
                        non_rgb.append((str(path), mode))

                    sizes.append((w, h))
                    ratios.append(w / h if h else float("nan"))

                    if min(w, h) < MIN_SIDE_FOR_OUTLIER:
                        small_images.append(str(path))

                    # Wykrywanie duplikatów 1:1 za pomocą hasza SHA-256
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if digest in seen_hashes:
                        duplicate_files.append((str(path), seen_hashes[digest]))
                    else:
                        seen_hashes[digest] = str(path)

                    # Próbki do podglądu letterboxingu (po 1 z pierwszych 4 klas)
                    if len(sample_images) < 4 and tsne_count_class == 0:
                        sample_images.append((pokemon_name, im.copy().convert("RGB")))

                    # Próbki cech do wykresu t-SNE
                    if tsne_count_class < MAX_TSNE_PER_CLASS and len(features) < MAX_TSNE_SAMPLES:
                        features.append(extract_color_features(im))
                        feature_labels.append(class_id)
                        tsne_count_class += 1

            except Exception as exc:
                unreadable.append((str(path), str(exc)))

    # Zapis tabeli liczebności do CSV
    csv_file = OUTPUT_DIR / "liczebnosc_klas.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Pokemon", "Liczba_Zdjec"])
        writer.writerows(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    ordered_classes = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    vals = [n for _, n in ordered_classes]

    # ==========================================================================
    # WYKRES 1: ROZKŁAD LICZEBNOŚCI KLAS
    # ==========================================================================
    print("--> Generowanie Wykresu 1: Rozkład liczebności klas...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)
    ax1.bar(range(len(ordered_classes)), vals, color="#2563eb", width=0.85, alpha=0.85)
    ax1.set(title=f"Liczba zdjęć w {len(ordered_classes)} klasach Pokémonów", xlabel="Klasy Pokémonów (malejąco)", ylabel="Liczba zdjęć")
    ax1.axhline(np.mean(vals), color="#dc2626", linestyle="--", linewidth=1.5, label=f"Średnia: {np.mean(vals):.1f}")
    ax1.axhline(np.median(vals), color="#16a34a", linestyle=":", linewidth=1.5, label=f"Mediana: {np.median(vals):.1f}")
    if min(vals) >= 150:
        ax1.set_ylim(min(vals) - 20, max(vals) + 20)
    ax1.legend(loc="lower left")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2.hist(vals, bins=min(15, max(5, len(set(vals)))), color="#38bdf8", edgecolor="#0369a1", alpha=0.85)
    ax2.set(title="Histogram liczebności klas", xlabel="Liczba zdjęć w folderze Pokémona", ylabel="Liczba klas")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "1_rozklad_klas.png", dpi=160)
    plt.close(fig)

    # ==========================================================================
    # WYKRES 2: WYMIARY I PROPORCJE SUROWYCH ZDJĘĆ
    # ==========================================================================
    print("--> Generowanie Wykresu 2: Wymiary i współczynniki proporcji...")
    arr_sizes = np.asarray(sizes, dtype=float)
    arr_ratios = np.asarray(ratios, dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    if len(arr_sizes):
        hb = ax1.hexbin(arr_sizes[:, 0], arr_sizes[:, 1], gridsize=30, mincnt=1, cmap="Blues")
        ax1.axline((0, 0), slope=1, linestyle="--", color="#dc2626", label="Format kwadratowy 1:1")
        ax1.set(title="Wymiary oryginalnych zdjęć (Szerokość vs Wysokość)", xlabel="Szerokość (px)", ylabel="Wysokość (px)")
        ax1.legend(loc="lower right")
        fig.colorbar(hb, ax=ax1, label="Liczba zdjęć")

        ax2.hist(arr_ratios[np.isfinite(arr_ratios)], bins=25, color="#0284c7", edgecolor="#075985", alpha=0.85)
        ax2.axvline(1.0, color="#dc2626", linestyle="--", linewidth=1.5, label="Kwadrat (1.0)")
        ax2.set(title="Współczynnik proporcji (Szerokość / Wysokość)", xlabel="Aspect Ratio", ylabel="Liczba zdjęć")
        ax2.legend(loc="upper right")
        ax2.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2_wymiary_i_proporcje.png", dpi=160)
    plt.close(fig)

    # ==========================================================================
    # WYKRES 3: PRZYKŁADY PRZETWARZANIA (LETTERBOX PADDING DO 224x224)
    # ==========================================================================
    print("--> Generowanie Wykresu 3: Przykłady zdjęć przed i po letterboxingu...")
    if sample_images:
        fig, axes = plt.subplots(len(sample_images), 2, figsize=(8, 2.6 * len(sample_images)), dpi=150)
        if len(sample_images) == 1:
            axes = np.asarray([axes])
        for row, (label, img) in enumerate(sample_images):
            axes[row, 0].imshow(img)
            axes[row, 0].set_title(f"Oryginał z PokemonData: {label} ({img.width}×{img.height})")
            axes[row, 0].axis("off")

            padded = letterbox(img, size=TARGET_SIZE)
            axes[row, 1].imshow(padded)
            axes[row, 1].set_title("Po letterboxingu (Tensor: 224×224 px)")
            axes[row, 1].axis("off")

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "3_przyklady_letterbox_224x224.png", dpi=160)
        plt.close(fig)

    # ==========================================================================
    # WYKRES 4: RZUT t-SNE / PCA PRZESTRZENI CECH BARWNYCH
    # ==========================================================================
    print("--> Generowanie Wykresu 4: Wizualizacja przestrzeni cech (PCA i t-SNE)...")
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(feature_labels)
    if len(X) >= 5:
        # Obliczenie PCA (z użyciem sklearn lub czystego numpy SVD jako fallback)
        if HAS_SKLEARN:
            pca = PCA(n_components=2, random_state=42)
            pca_xy = pca.fit_transform(X)
        else:
            X_centered = X - np.mean(X, axis=0)
            u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
            pca_xy = np.dot(X_centered, vt[:2].T)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
        cmap = plt.get_cmap("tab20", min(20, len(class_dirs)))

        axes[0].scatter(pca_xy[:, 0], pca_xy[:, 1], c=y, cmap=cmap, s=22, alpha=0.75)
        axes[0].set(title="Rzut PCA (analiza barw RGB)", xlabel="Główna składowa 1", ylabel="Główna składowa 2")
        axes[0].grid(True, linestyle="--", alpha=0.3)

        if HAS_SKLEARN and len(X) >= 10:
            perplexity = min(30, max(2, (len(X) - 1) // 3))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42)
                tsne_xy = tsne.fit_transform(X)
            axes[1].scatter(tsne_xy[:, 0], tsne_xy[:, 1], c=y, cmap=cmap, s=22, alpha=0.75)
            axes[1].set(title="Rzut t-SNE (podobieństwo kolorystyczne)", xlabel="Wymiar t-SNE 1", ylabel="Wymiar t-SNE 2")
        else:
            # Gdy brak sklearn lub za mało próbek: drugi rzut na wariancję barw
            axes[1].scatter(X[:, 0], X[:, 8], c=y, cmap=cmap, s=22, alpha=0.75)
            axes[1].set(title="Korelacja barw (Kanał Czerwony vs Zielony)", xlabel="Intensywność czerwieni", ylabel="Intensywność zieleni")
        
        axes[1].grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "4_pca_tsne_przestrzen_cech.png", dpi=160)
        plt.close(fig)

    # ==========================================================================
    # PODSUMOWANIE W KONSOLI
    # ==========================================================================
    total_imgs = sum(counts.values())
    print("\n" + "="*70)
    print("PODSUMOWANIE ANALIZY EDA:")
    print(f"• Folder źródłowy:            {data_dir.resolve()}")
    print(f"• Liczba znalezionych klas:    {len(class_dirs)}")
    print(f"• Łączna liczba zdjęć:        {total_imgs}")
    print(f"• Średnia zdjęć na klasę:     {np.mean(vals):.1f}")
    print(f"• Min / Max w klasach:        {min(vals)} / {max(vals)}")
    print(f"• Uszkodzone pliki (do kosza):{len(unreadable)}")
    print(f"• Pliki z przezroczystością:  {len(non_rgb)}")
    print(f"• Wykryte duplikaty SHA-256:  {len(duplicate_files)}")
    print(f"• Zapisano wykresy w:         {OUTPUT_DIR.resolve()}/")
    print(f"• Zapisano plik CSV:          {csv_file.resolve()}")
    print("="*70)


if __name__ == "__main__":
    main()
