# src/icons.py
import os
import requests
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QByteArray

# Optionnel : Cache local pour éviter de re-télécharger à chaque lancement
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lucide_icons")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_lucide_icon(icon_name: str, color: str = "#94a3b8") -> QIcon:
    """
    Récupère une icône Lucide par son nom, applique une couleur hexadécimale,
    et retourne un QIcon PyQt6.
    """
    file_path = os.path.join(CACHE_DIR, f"{icon_name}.svg")
    
    # 1. Téléchargement si non présent dans le cache local
    if not os.path.exists(file_path):
        url = f"https://unpkg.com/lucide-static/icons/{icon_name}.svg"
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
            else:
                return QIcon() # Retourne une icône vide si non trouvée
        except Exception:
            return QIcon()

    # 2. Lecture du SVG et injection de la couleur personnalisée
    with open(file_path, "r", encoding="utf-8") as f:
        svg_content = f.read()
    
    # Remplacement des attributs par défaut de Lucide (stroke="currentColor")
    # pour appliquer la couleur choisie par ton thème HSL
    svg_content = svg_content.replace('stroke="currentColor"', f'stroke="{color}"')
    
    # Convertir en QIcon via un tableau de bytes en mémoire
    byte_array = QByteArray(svg_content.encode("utf-8"))
    
    # Utilisation temporaire d'un fichier ou d'un pixmap pour générer le QIcon
    # (Le plus stable en PyQt6 avec SVG text direct est de passer par QPixmap)
    from PyQt6.QtGui import QPixmap
    pixmap = QPixmap()
    pixmap.loadFromData(byte_array, "SVG")
    
    return QIcon(pixmap)

def configure_button(button, text: str = None, icon_name: str = None, icon_color: str = "#f8fafc", bg_color: str = None):
    """
    Utility function to configure a QPushButton's text, icon, and optional background color style.
    """
    if text is not None:
        button.setText(text)
    if icon_name is not None:
        button.setIcon(get_lucide_icon(icon_name, color=icon_color))
    if bg_color is not None:
        button.setStyleSheet(f"background-color: {bg_color}; color: white; font-weight: bold;")