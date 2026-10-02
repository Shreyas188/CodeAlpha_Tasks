"""
CodeAlpha AI Internship - Master Setup Script
Installs all dependencies for all 4 tasks
Run this once before running any project.
"""
import subprocess
import sys
import os

def install(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

def run_command(cmd):
    """Run a shell command."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0

print("=" * 60)
print("  CodeAlpha AI Internship - Dependency Installer")
print("=" * 60)
print()

# Common packages
packages = {
    "Task 1 - Language Translation Tool": [
        "customtkinter==5.2.2",
        "deep-translator==1.11.4",
        "langdetect==1.0.9",
        "pyttsx3==2.90",
        "pillow==10.4.0",
        "pyperclip",
    ],
    "Task 2 - FAQ Chatbot": [
        "scikit-learn==1.5.2",
        "nltk==3.9.1",
        "numpy==1.26.4",
    ],
    "Task 3 - Music Generation": [
        "tensorflow",
        "music21==9.3.0",
        "midiutil==1.2.1",
        "matplotlib==3.9.2",
    ],
    "Task 4 - Object Detection": [
        "ultralytics==8.3.27",
        "opencv-python==4.10.0.84",
        "scipy==1.14.1",
    ],
}

for task, pkgs in packages.items():
    print(f"\n📦 Installing: {task}")
    print("-" * 40)
    for pkg in pkgs:
        print(f"  Installing {pkg}...", end=" ", flush=True)
        try:
            install(pkg)
            print("✅")
        except Exception as e:
            print(f"⚠️  Warning: {e}")

# Download NLTK data
print("\n📦 Downloading NLTK data...")
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print("  ✅ NLTK data downloaded")
except Exception as e:
    print(f"  ⚠️  Warning: {e}")

print()
print("=" * 60)
print("  ✅ All dependencies installed!")
print()
print("  To run each project:")
print("  Task 1: cd CodeAlpha_LanguageTranslation && python app.py")
print("  Task 2: cd CodeAlpha_FAQChatbot       && python chatbot.py")
print("  Task 3: cd CodeAlpha_MusicGeneration  && python app.py")
print("  Task 4: cd CodeAlpha_ObjectDetection  && python app.py")
print("=" * 60)
