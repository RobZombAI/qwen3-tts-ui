# Qwen3 TTS — Android Compatibility Guide

## Verdetto Onesto

**L'app Android nativa per Qwen3-TTS-12Hz-1.7B non è praticabile al momento.**

| Feature | Realistico? | Perché |
|---------|------------|--------|
| Modello 1.7B on-device | ❌ | 3.5 GB RAM solo pesi — nessun telefono ha tanta memoria libera |
| Modello 0.6B on-device | ⚠️ | Solo top di gamma (12GB+ RAM), molto lento su CPU |
| Voice Clone on-device | ❌ | Ancora più heavy (2 modelli caricati) |
| Streaming da server | ✅ | Funziona su qualsiasi telefono |

## Il problema

Il modello Qwen3-TTS è un **transformer autoregressivo da 1.7 miliardi di parametri**:

1. **RAM**: Solo i pesi in float16 occupano ~3.4 GB. Su Android, un'app tipica ha ~1-2 GB disponibili
2. **CPU/GPU**: Android non ha CUDA né MPS. Solo CPU (lento) o GPU via OpenCL (non supportato bene da PyTorch)
3. **Framework**: PyTorch Mobile esiste ma ha limitazioni grosse su modelli transformer custom
4. **Conversione**: Il modello usa architetture custom (`Qwen3TTSForConditionalGeneration`) non esportabili facilmente in TorchScript

## Soluzione praticabile: App Android "Companion"

L'unica via realistica: **un'app Android che si connette al server macOS/Windows sulla stessa rete**.

```
┌─────────────────────┐       WiFi       ┌─────────────────────┐
│  Android App        │ ────────────────→ │  macOS / Windows    │
│  • UI nativa        │   HTTP / JSON     │  • Flask Server     │
│  • Registra audio   │   ←────────────── │  • Qwen3 TTS model  │
│  • Riproduce output │   WAV stream      │  • Salva profili    │
│  • Salva profili    │                   │                     │
└─────────────────────┘                   └─────────────────────┘
```

### Cosa farebbe l'app Android

| Schermata | Funzione |
|-----------|----------|
| **Home** | Elenco server trovati sulla LAN (Bonjour/mDNS) |
| **TTS** | Testo → tasto Generate → ascolta risultato |
| **Voice Clone** | Registra audio → upload → clona |
| **Profili** | Salva/carica voci clonate |
| **Settings** | Indirizzo server, cartella output |

### Come implementarla

L'app Android sarebbe un'app nativa (Kotlin/Java o Flutter) che:
1. Cerca il server Qwen3 TTS sulla rete locale via mDNS
2. Si connette via HTTP alle stesse API del server
3. Mostra la UI nativa Android (Material Design 3)
4. Supporta registrazione audio per voice clone

## Compatibilità Telefoni (se volessi provare Termux)

Se vuoi assolutamente provare su Android via **Termux** (Python environment):

| Telefono | RAM | Storage | Esito |
|----------|-----|---------|-------|
| Samsung S25 Ultra | 12 GB | 256 GB+ | ⚠️ 0.6B forse, molto lento |
| Pixel 9 Pro | 16 GB | 128 GB+ | ⚠️ 0.6B, ~50× RTF |
| OnePlus 13 | 12 GB | 256 GB+ | ⚠️ 0.6B, ~50× RTF |
| iPhone 16 Pro (MPS) | 8 GB | 128 GB+ | ✅ 0.6B via CoreML, ~10× RTF |
| Telefoni < 8 GB | — | — | ❌ Non provare |

### Setup Termux (solo per test)

```bash
pkg install python python-numpy
pip install qwen-tts torch --no-deps
# Molto lento, non raccomandato
```

## Raccomandazione

**Invece di un APK standalone**, ti suggerisco:

1. ✅ **Usa l'app macOS/Windows** — già completa e testata
2. 📱 **Crea un'app Android Companion** (Flutter) che si connette al server via WiFi
3. 🔄 **Stesso server, interfaccia mobile** — parli alla stessa API, stessi profili

Se vuoi procedere con l'app Android Companion, dimmi e ti preparo il progetto Flutter completo. L'APK risultante sarebbe ~10 MB e funzionerebbe su **qualunque Android 8+**.
