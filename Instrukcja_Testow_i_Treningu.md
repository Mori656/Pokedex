# Instrukcja wdrożenia i realizacji wymagań projektu (Etap 4 / Pkt 5)
**Projekt:** Klasyfikacja 149 klas Pokémonów z użyciem ResNet-18 od podstaw  
**Przedmiot:** Sieci neuronowe i uczenie głębokie – Akademia Tarnowska (Dr inż. Marcin Skobel, 2026)  
**Autor:** Marcel Nosek  

---

## 1. Szybki start: Jak zrobić pierwsze testy?

W skrypcie `train_pokemon.py` przygotowano dwa dedykowane tryby testowe, które pozwalają natychmiast zweryfikować poprawność kodu przed uruchomieniem pełnego, wielogodzinnego treningu:

### Krok 1. Sanity Check (Test samej architektury – trwa 1 sekundę)
Sprawdza liczbę parametrów, wymiary tensorów wejściowych/wyjściowych, brak wartości NaN/Inf oraz działanie funkcji straty:
```bash
python train_pokemon.py --sanity-check
```
**Oczekiwany rezultat:**
* Parametry: **11 252 949** (wszystkie trenowalne).
* Wejście: `[32, 3, 224, 224]`, Wyjście: `[32, 149]` (logity).
* Testowa strata `CrossEntropyLoss`: ~5.00–5.33 (idealna dla losowej inicjalizacji 149 klas, ponieważ $-\ln(1/149) \approx 5.00$).

---

### Krok 2. Smoke Test (Próbny bieg treningu – trwa ~1-2 minuty)
Uruchamia 1 epokę na małej próbce danych, sprawdzając cały potok: odczyt folderów `PokemonData/`, deduplikację, letterboxing, pętlę treningową, walidację, obliczanie metryk (Top-1, Top-5, Macro F1), zapis checkpointu `best_model.pt` oraz wygenerowanie wykresów `wykresy_treningu.png`:
```bash
python train_pokemon.py --smoke-test --data-dir ./PokemonData
```

---

### Krok 3. Pełny trening docelowy (po zaliczeniu testów)
```bash
python train_pokemon.py --data-dir ./PokemonData --epochs 25 --batch-size 32 --lr 0.0003
```

---

## 2. Pokrycie wymagań z punktu 5 wytycznych

Poniższa tabela przedstawia, jak przygotowane pliki realizują wymagane rezultaty projektu:

| Wymagany element (Pkt 5) | Realizacja w projekcie |
| :--- | :--- |
| **(1) Skrypt lub notebook** | `train_pokemon.py` – kompletny, samodzielny skrypt PyTorch realizujący pełen potok od surowych folderów po ewaluację testową. |
| **(2) Plik requirements.txt** | `requirements.txt` ze spisem bibliotek: `torch`, `torchvision`, `numpy`, `Pillow`, `scikit-learn`, `matplotlib`. |
| **(3) Opis danych i podziału** | 26 786 obrazów, 149 klas, podział Stratified Holdout 70% / 15% / 15% z deduplikacją 67 duplikatów SHA-256 (plik `split_indices.json`). |
| **(4) Model i uzasadnienie strategii** | Własna implementacja ResNet-18 z blokami `BasicBlock` i losową inicjalizacją Kaiming He (patrz sekcja 3 poniżej). |
| **(5) Konfiguracja i checkpoint** | AdamW (`lr=3e-4`, `weight_decay=1e-4`), `CosineAnnealingLR`, Early Stopping (patience=5), zapis `best_model.pt`. |
| **(6) Raport, wykresy i metryki** | Liczba parametrów: **11 252 949**, wykresy strat i dokładności w `wykresy_treningu.png`, metryki: Top-1 Acc, Top-5 Acc, Macro F1. |

---

## 3. Uzasadnienie wyboru strategii (Odpowiedzi na pytania z Pkt 6)

### Pytanie 1: Dlaczego strategia „od podstaw” (From Scratch), a nie fine-tuning lub model generatywny?
* **Alternatywa 1 (Fine-tuning):** Wstępnie wytrenowany model (np. na ImageNet) uczył się na zdjęciach rzeczywistych obiektów (psy, samochody, krajobrazy). Wizerunki Pokémonów to rysunki anime, grafiki wektorowe i karty TCG o specyficznej kresce i kolorystyce. Zbiór liczy aż **~26 800 zdjęć**, co stanowi solidną bazę do nauczenia wag od zera, bez narzucania biasu z fotografii ImageNet.
* **Alternatywa 2 (Model generatywny):** Zadaniem projektu jest precyzyjna dyskryminacja (klasyfikacja 1 ze 149 klas), a nie synteza nowych Pokémonów czy rekonstrukcja tła. Zastosowanie GAN/Diffusion byłoby niezgodne z celem biznesowym zadania.

### Pytanie 2: Jak kontrolować przeuczenie (Overfitting)?
* Zastosowano **Dropout(0.3)** przed warstwą klasyfikującą.
* Wprowadzono **Weight Decay ($10^{-4}$)** w optymalizatorze AdamW.
* Zastosowano mocną augmentację w locie (obroty, losowe odbicia, jitter kolorów) wyłącznie dla zbioru treningowego.
* Zaimplementowano **Early Stopping** z monitorowaniem straty walidacyjnej (`val_loss`).

### Pytanie 3: Dlaczego sama metryka Accuracy nie wystarcza?
* **Top-5 Accuracy:** Wiele Pokémonów należy do tej samej linii ewolucyjnej (np. *Pikachu ↔ Raichu*, *Gastly ↔ Haunter ↔ Gengar*) i dzieli zbliżone kolory oraz sylwetki. Top-5 pozwala ocenić, czy sieć poprawnie zawęziła wybór do właściwej rodziny cech.
* **Macro F1-Score:** Sprawdza jakość klasyfikacji z równą wagą dla każdej ze 149 klas, gwarantując, że model nie ignoruje klas mniej licznych (np. Persian z 138 zdjęciami).
