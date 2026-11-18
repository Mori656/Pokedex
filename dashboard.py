import os
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd

# Import architektury i preprocessingu z Twojego skryptu
from train_pokemon import ResNet18Pokemon, letterbox_image


st.set_page_config(page_title="Pokédex AI Dashboard", page_icon="⚡", layout="wide")

st.title("⚡ Pokédex Deep Learning Dashboard")
st.caption("Klasyfikacja Pokémonów I Generacji (149 klas) | ResNet-18 From Scratch")

tab1, tab2 = st.tabs(["📊 Statystyki i Wyniki Treningu", "🔍 Rozpoznawanie Pokémona na żywo"])

# -------------------------------------------------------------
# ZAKŁADKA 1: WYNIKI I STATYSTYKI
# -------------------------------------------------------------
with tab1:
    st.header("Metryki i podsumowanie zbioru")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Liczba klas", "149")
    col2.metric("Wszystkie obrazy", "26 786")
    col3.metric("Architektura", "ResNet-18 (~11.2M wag)")
    col4.metric("Podział", "70% / 15% / 15%")

    st.divider()
    
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("📈 Krzywe uczenia (Loss i Accuracy)")
        if os.path.exists("wykresy_treningu.png"):
            st.image("wykresy_treningu.png", use_container_width=True)
        else:
            st.info("Plik `wykresy_treningu.png` pojawi się tutaj automatycznie po zakończeniu pełnego treningu.")

    with col_right:
        st.subheader("📋 Rozkład klas i audyt EDA")
        if os.path.exists("wykresy_eda/1_rozklad_klas.png"):
            st.image("wykresy_eda/1_rozklad_klas.png", use_container_width=True)
        elif os.path.exists("1_rozklad_klas.png"):
            st.image("1_rozklad_klas.png", use_container_width=True)
        else:
            st.write("Średnio ~180 zdjęć na klasę (min 138, max 243).")

# -------------------------------------------------------------
# ZAKŁADKA 2: KLASYFIKACJA NOWYCH ZDJĘĆ
# -------------------------------------------------------------
@st.cache_resource
def load_trained_model(checkpoint_path="best_model.pt"):
    if not os.path.exists(checkpoint_path):
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    class_names = checkpoint.get("class_names")
    num_classes = len(class_names) if class_names else 149
    
    model = ResNet18Pokemon(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names

with tab2:
    st.header("Wgraj zdjęcie, aby rozpoznać Pokémona")
    model, class_names = load_trained_model("best_model.pt")

    if model is None:
        st.warning("⚠️ Nie znaleziono pliku `best_model.pt`. Uruchom najpierw trening lub umieść plik wag w folderze projektu.")
    else:
        uploaded_file = st.file_uploader("Wybierz plik z Pokémonem (JPG, PNG, WEBP)...", type=["jpg", "jpeg", "png", "webp"])
        
        if uploaded_file is not None:
            image_raw = Image.open(uploaded_file)
            
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.write("**Oryginalne zdjęcie:**")
                st.image(image_raw, use_container_width=True)

            # Preprocessing identyczny jak w treningu
            image_boxed = letterbox_image(image_raw)
            with col_img2:
                st.write("**Po dopasowaniu (Letterbox 224x224):**")
                st.image(image_boxed, use_container_width=True)

            # Predykcja
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            device = next(model.parameters()).device
            input_tensor = transform(image_boxed).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1)[0]
                top5_probs, top5_indices = torch.topk(probs, 5)

            top1_name = class_names[top5_indices[0]] if class_names else f"Klasa #{top5_indices[0].item()}"
            top1_conf = top5_probs[0].item() * 100

            st.success(f"🎯 Rozpoznano: **{top1_name}** z pewnością **{top1_conf:.1f}%**")

            # Tabela Top-5
            st.subheader("Top-5 najbardziej prawdopodobnych typów:")
            top5_data = {
                "Pokémon": [class_names[idx] if class_names else f"Klasa {idx}" for idx in top5_indices],
                "Prawdopodobieństwo (%)": [round(p.item() * 100, 2) for p in top5_probs]
            }
            df_top5 = pd.DataFrame(top5_data)
            st.bar_chart(df_top5.set_index("Pokémon"))
