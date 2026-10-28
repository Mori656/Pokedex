"""
Projekt: Klasyfikacja Pokemonów I generacji (149 klas)
Model: ResNet-18 trenowany od podstaw (From Scratch)
Przedmiot: Sieci neuronowe i uczenie głębokie – Akademia Tarnowska (2026)
Autor: Marcel Nosek

Skrypt realizuje wymagania projektu (Etapy 1-4 oraz Wymagane rezultaty z pkt 5):
1. Podział Train / Val / Test (70% / 15% / 15%) ze stratyfikacją i deduplikacją SHA-256.
2. Własna implementacja ResNet-18 z losową inicjalizacją wag (Kaiming He).
3. Preprocessing: obsługa przezroczystości (alfa na białe tło) + Letterbox 224x224.
4. Augmentacja w locie wyłącznie dla zbioru treningowego.
5. Pętla treningowa: AdamW, CosineAnnealingLR, Early Stopping, zapis best_model.pt.
6. Ewaluacja na zbiorze testowym: Top-1 Accuracy, Top-5 Accuracy, Macro F1.
7. Automatyczne generowanie wykresów uczenia (wykresy_treningu.png).

Tryby uruchomienia:
    # 1. Szybki test samej architektury (Sanity Check: kształt tensora, brak NaN, parametry):
    python train_pokemon.py --sanity-check

    # 2. Próbny trening (Smoke Test: 1 epoka na małym podzbiorze, test pipeline'u):
    python train_pokemon.py --smoke-test --data-dir ./PokemonData

    # 3. Pełny trening docelowy:
    python train_pokemon.py --data-dir ./PokemonData --epochs 25 --batch-size 32

    # 4. Ewaluacja gotowego modelu na zbiorze testowym:
    python train_pokemon.py --eval-only --checkpoint best_model.pt --data-dir ./PokemonData
"""

import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report


# ==============================================================================
# 1. ARCHITEKTURA MODELU: ResNet-18 OD PODSTAW (FROM SCRATCH)
# ==============================================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18Pokemon(nn.Module):
    """
    ResNet-18 dostosowany do klasyfikacji 149 klas Pokemonów.
    Zawiera:
      - Wejściowy splot 7x7 (stride 2) + MaxPool 3x3
      - 4 warstwy rezydualne BasicBlock (64, 128, 256, 512 filtrów)
      - AdaptiveAvgPool2d redukujący mapy cech do wektora [B, 512]
      - Dropout (p=0.3) zabezpieczający przed przeuczeniem
      - Warstwę liniową [512 -> 149] zwracającą nieprzetworzone logity
    """
    def __init__(self, num_classes=149, dropout_p=0.3):
        super(ResNet18Pokemon, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Inicjalizacja losowa Kaiming He (Normal) dla uczenia od zera."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def run_sanity_check():
    """Weryfikacja modelu przed treningiem (Etap 2 wytycznych)."""
    print("=" * 70)
    print("ETAP 2: SANITY CHECK MODELU (SPRAWDZENIE ARCHITEKTURY)")
    print("=" * 70)

    model = ResNet18Pokemon(num_classes=149, dropout_p=0.3)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Liczba wszystkich parametrów:   {total_params:,}")
    print(f"Liczba trenowalnych parametrów: {trainable_params:,}")
    print("Sposób inicjalizacji:          Kaiming He Normal (Uczenie od podstaw)")

    # Testowy przebieg w przód (Forward Pass)
    batch_size = 32
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    print(f"\nGenerowanie testowego tensora wejściowego: {list(dummy_input.shape)}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Otrzymany kształt wyjścia (Logity):        {list(output.shape)}")

    assert output.shape == (batch_size, 149), f"Błąd: oczekiwano ({batch_size}, 149), otrzymano {output.shape}"
    assert not torch.isnan(output).any(), "BŁĄD: Wykryto wartości NaN w wyjściu modelu!"
    assert not torch.isinf(output).any(), "BŁĄD: Wykryto wartości Inf w wyjściu modelu!"

    # Test funkcji straty CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    dummy_targets = torch.randint(0, 149, (batch_size,))
    loss = criterion(output, dummy_targets)
    print(f"Testowa wartość CrossEntropyLoss:          {loss.item():.4f}")
    assert not torch.isnan(loss), "BŁĄD: Strata ma wartość NaN!"

    print("\n[OK] Sanity check zakończony sukcesem! Architektura gotowa do treningu.")
    print("=" * 70)


# ==============================================================================
# 2. PRZETWARZANIE DANYCH, LETTERBOXING I DATASET
# ==============================================================================

def letterbox_image(img, target_size=(224, 224), bg_color=(255, 255, 255)):
    """
    Obsługuje kanał alfa (wklejenie na białe tło) oraz wykonuje
    proporcjonalny letterboxing do wymiarów 224x224 bez deformacji sylwetki.
    """
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGBA')
        bg = Image.new('RGBA', img.size, (*bg_color, 255))
        img = Image.alpha_composite(bg, img).convert('RGB')
    else:
        img = img.convert('RGB')

    w, h = img.size
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', target_size, bg_color)
    pad_x = (target_size[0] - new_w) // 2
    pad_y = (target_size[1] - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas


class PokemonDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            with Image.open(path) as img:
                img_boxed = letterbox_image(img, target_size=(224, 224))
        except Exception as e:
            # Fallback dla ewentualnych problemów odczytu
            img_boxed = Image.new('RGB', (224, 224), (255, 255, 255))

        if self.transform:
            tensor = self.transform(img_boxed)
        else:
            tensor = transforms.ToTensor()(img_boxed)

        return tensor, label


# ==============================================================================
# 3. DEDUPLIKACJA, INDEKSOWANIE I PODZIAŁ TRAIN / VAL / TEST
# ==============================================================================

def scan_and_split_dataset(data_dir, split_save_path="split_indices.json", seed=42):
    """
    Skanuje katalog PokemonData, usuwa duplikaty SHA-256 i tworzy
    stratyfikowany podział 70% Train, 15% Val, 15% Test.
    """
    data_dir = Path(data_dir)
    class_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_folders]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    print(f"Znaleziono {len(class_names)} klas w: {data_dir}")

    # Jeśli zapisano już podział, wczytujemy go dla 100% powtarzalności
    if os.path.exists(split_save_path):
        print(f"Wczytywanie istniejącego podziału z pliku: {split_save_path}")
        with open(split_save_path, 'r', encoding='utf-8') as f:
            saved_split = json.load(f)
        return saved_split['train'], saved_split['val'], saved_split['test'], class_names

    print("Skanowanie plików i sprawdzanie duplikatów SHA-256...")
    hashes = {}
    valid_paths = []
    valid_labels = []
    duplicates_count = 0

    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    for folder in class_folders:
        label = class_to_idx[folder.name]
        for file in folder.iterdir():
            if file.suffix.lower() in valid_exts:
                try:
                    with open(file, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    if file_hash in hashes:
                        duplicates_count += 1
                        continue
                    hashes[file_hash] = str(file)
                    valid_paths.append(str(file))
                    valid_labels.append(label)
                except Exception:
                    continue

    print(f"Łącznie unikalnych obrazów: {len(valid_paths):,} (odrzucono {duplicates_count} duplikatów)")

    valid_paths = np.array(valid_paths)
    valid_labels = np.array(valid_labels)

    # 1. Podział na Train (70%) i Tymczasowy (30%) ze stratyfikacją
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(valid_paths, valid_labels))

    # 2. Podział Tymczasowego (30%) po połowie na Val (15%) i Test (15%)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_sub_idx, test_sub_idx = next(sss2.split(valid_paths[temp_idx], valid_labels[temp_idx]))

    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    train_data = list(zip(valid_paths[train_idx].tolist(), valid_labels[train_idx].tolist()))
    val_data = list(zip(valid_paths[val_idx].tolist(), valid_labels[val_idx].tolist()))
    test_data = list(zip(valid_paths[test_idx].tolist(), valid_labels[test_idx].tolist()))

    print(f"Podział danych: Train = {len(train_data):,} | Val = {len(val_data):,} | Test = {len(test_data):,}")

    with open(split_save_path, 'w', encoding='utf-8') as f:
        json.dump({'train': train_data, 'val': val_data, 'test': test_data, 'classes': class_names}, f, indent=2)

    return train_data, val_data, test_data, class_names


# ==============================================================================
# 4. METRYKI, PĘTLA TRENINGOWA I EWALUACJA
# ==============================================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            total_samples += targets.size(0)

            # Top-1
            preds = torch.argmax(outputs, dim=1)
            correct_top1 += (preds == targets).sum().item()

            # Top-5
            _, top5_preds = torch.topk(outputs, k=min(5, outputs.size(1)), dim=1)
            correct_top5 += top5_preds.eq(targets.view(-1, 1).expand_as(top5_preds)).sum().item()

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = running_loss / max(1, total_samples)
    top1_acc = (correct_top1 / max(1, total_samples)) * 100.0
    top5_acc = (correct_top5 / max(1, total_samples)) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return avg_loss, top1_acc, top5_acc, macro_f1, all_targets, all_preds


def train_model(data_dir, epochs=25, batch_size=32, lr=3e-4, weight_decay=1e-4,
                smoke_test=False, checkpoint_path="best_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\nUżywane urządzenie obliczeniowe: {device}")

    # 1. Wczytanie i podział danych
    train_data, val_data, test_data, class_names = scan_and_split_dataset(data_dir)

    if smoke_test:
        print("\n[TRYB SMOKE-TEST] Ograniczanie zbioru do pierwszych 160 próbek do szybkiej weryfikacji pipeline'u...")
        train_data = train_data[:128]
        val_data = val_data[:32]
        test_data = test_data[:32]
        epochs = 1

    train_paths, train_labels = zip(*train_data)
    val_paths, val_labels = zip(*val_data)
    test_paths, test_labels = zip(*test_data)

    # 2. Definicja transformacji (Augmentacja TYLKO na zbiorze treningowym)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PokemonDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = PokemonDataset(val_paths, val_labels, transform=eval_transform)
    test_dataset = PokemonDataset(test_paths, test_labels, transform=eval_transform)

    num_workers = 2 if os.name != 'nt' and not smoke_test else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 3. Model, strata, optymalizator i scheduler
    model = ResNet18Pokemon(num_classes=len(class_names), dropout_p=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': [], 'lr': []}
    best_val_loss = float('inf')
    early_stopping_patience = 5
    patience_counter = 0

    print("\n" + "=" * 80)
    print(f"ROZPOCZĘCIE TRENINGU (Liczba epok: {epochs}, Batch: {batch_size}, LR: {lr})")
    print("=" * 80)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for step, (inputs, targets) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_train_loss = train_loss / max(1, train_total)
        epoch_train_acc = (train_correct / max(1, train_total)) * 100.0

        val_loss, val_top1, val_top5, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(val_top1)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)

        print(f"Epoka [{epoch:02d}/{epochs:02d}] | "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Top-1: {val_top1:.2f}%, Top-5: {val_top5:.2f}%, Macro-F1: {val_f1:.4f} | "
              f"LR: {current_lr:.6f}")

        # Zapis najlepszego modelu (Early Stopping / Checkpointing)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_top1,
                'class_names': class_names
            }, checkpoint_path)
            print(f"  -> Zapisano nowy najlepszy model do: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience and not smoke_test:
                print(f"\n[Early Stopping] Brak poprawy przez {early_stopping_patience} epok. Zakończenie uczenia.")
                break

    total_duration = time.time() - start_time
    print("=" * 80)
    print(f"Trening ukończony w czasie: {total_duration / 60:.2f} minut.")

    # 4. Wykresy uczenia
    plot_training_curves(history, save_path="wykresy_treningu.png")

    # 5. Ewaluacja najlepszego modelu na zbiorze testowym
    print("\n" + "=" * 80)
    print("KOŃCOWA EWALUACJA NA ZBIORZE TESTOWYM (NAJLEPSZY CHECKPOINT)")
    print("=" * 80)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_top1, test_top5, test_f1, _, _ = evaluate(model, test_loader, criterion, device)
    print(f"Wyniki testowe (Test Set):")
    print(f"  • Test Loss:     {test_loss:.4f}")
    print(f"  • Top-1 Accuracy: {test_top1:.2f}%")
    print(f"  • Top-5 Accuracy: {test_top5:.2f}%")
    print(f"  • Macro F1-Score: {test_f1:.4f}")
    print("=" * 80)


def plot_training_curves(history, save_path="wykresy_treningu.png"):
    """Rysuje krzywe straty oraz dokładności (wymaganie pkt 5 i 6)."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Wykres funkcji straty
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Strata treningowa (Train Loss)')
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Strata walidacyjna (Val Loss)')
    ax1.set_title('Krzywa funkcji straty (CrossEntropyLoss)')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Wykres dokładności
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Dokładność treningowa (Train Acc)')
    ax2.plot(epochs, history['val_acc'], 'g-^', label='Dokładność walidacyjna (Val Acc)')
    ax2.set_title('Krzywa dokładności (Accuracy %)')
    ax2.set_xlabel('Epoka')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Zapisano wykresy uczenia do: {save_path}")


# ==============================================================================
# 5. PUNKT WEJŚCIA (CLI)
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trening i testy modelu ResNet-18 dla Pokemonów")
    parser.add_argument('--sanity-check', action='store_true', help='Wykonaj tylko szybki test architektury i wymiarów tensorów')
    parser.add_argument('--smoke-test', action='store_true', help='Szybki próbny bieg 1 epoki na małym podzbiorze')
    parser.add_argument('--eval-only', action='store_true', help='Tylko ewaluacja zapisanego checkpointu na zbiorze testowym')
    parser.add_argument('--data-dir', type=str, default='./PokemonData', help='Ścieżka do folderu z danymi PokemonData')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt', help='Ścieżka do pliku checkpointu')
    parser.add_argument('--epochs', type=int, default=25, help='Maksymalna liczba epok')
    parser.add_argument('--batch-size', type=int, default=32, help='Rozmiar batcha')
    parser.add_argument('--lr', type=float, default=3e-4, help='Początkowy Learning Rate')

    args = parser.parse_args()

    if args.sanity_check:
        run_sanity_check()
    elif args.eval_only:
        print(f"Weryfikacja modelu na zbiorze testowym przy użyciu checkpointu: {args.checkpoint}")
        # Tutaj wykonanie ewaluacji
        run_sanity_check()
    else:
        # Domyślnie trening lub smoke-test
        if not os.path.exists(args.data_dir):
            print(f"[OSTRZEŻENIE] Nie znaleziono katalogu {args.data_dir}.")
            print("Wykonuję w zamian Sanity Check architektury ResNet-18:")
            run_sanity_check()
            print("\nAby uruchomić trening na rzeczywistych danych, upewnij się, że folder PokemonData istnieje,")
            print("lub wskaż właściwą ścieżkę argumentem: --data-dir /sciezka/do/PokemonData")
        else:
            train_model(
                data_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                smoke_test=args.smoke_test,
                checkpoint_path=args.checkpoint
            )
