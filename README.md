# Tri-Stream Spectral–Spatial Fusion Transformer for Melanoma Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q86wactcHUrKk_dH1PIQvEO7wzMlLqM3?usp=sharing)


## Overview
This repository contains the implementation of a novel deep learning framework for automated skin lesion classification. Standard Convolutional Neural Networks (CNNs) often miss complex boundaries and texture frequencies in dermatological images. To overcome this, we designed a **Tri-Stream Architecture** that extracts spatial, spectral, and geometric features, dynamically fusing them via a Transformer-based self-attention engine.

This model is designed to accurately classify 7 distinct types of skin lesions (including melanoma) to assist in early and reliable automated clinical diagnosis.

## System Architecture

The model utilizes three parallel "expert" streams:
1. **Spatial Stream (Visual Expert):** Uses a pre-trained **ConvNeXt-Tiny** backbone to extract high-level semantic features, global shape, and color distribution.
2. **Spectral Stream (Frequency Expert):** Utilizes Fast Fourier Transform (**FFT**) and Discrete Wavelet Transform (**DWT**) to capture sub-visual texture and multi-scale frequency patterns.
3. **Edge Stream (Boundary Expert):** Models clinical **ABCD** rules (Asymmetry, Border, Color, Diameter) using Sobel operators for gradient extraction and color histograms.

```mermaid
graph TD
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef stream fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef fusion fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:black;
    classDef output fill:#ffebee,stroke:#c62828,stroke-width:2px;

    subgraph Data_Pipeline [Data Pipeline & Preprocessing]
        direction TB
        RawData[("Raw Datasets<br/>(HAM10000)")]:::input
        subgraph Preprocessing_Steps [Preprocessing Module]
            Hair["Hair Removal<br/>(Morphological Ops)"]:::process
            Norm["Macenko Color<br/>Normalization"]:::process
            Seg["Lesion Segmentation<br/>(Otsu Thresholding)"]:::process
        end
        RawData --> Hair --> Norm --> Seg
    end

    subgraph Feature_Extraction [Feature Extraction Module]
        direction TB
        ImgRes["Resize Image<br/>(224x224 RGB)"]:::process
        subgraph Spectral_Feats [Spectral Features]
            FFT["FFT Extraction<br/>(Frequency Domain)"]:::process
            DWT["DWT Extraction<br/>(Wavelet/Texture)"]:::process
        end
        subgraph Edge_Feats [Hand-Crafted Features]
            ABCD["ABCD Rule Metrics<br/>(Asymmetry, Border, Color)"]:::process
            Sobel["Sobel Edges &<br/>Fractal Dimension"]:::process
        end
        Seg --> ImgRes
        Seg --> FFT
        Seg --> DWT
        Seg --> ABCD
        Seg --> Sobel
    end

    subgraph Model_Architecture [Tri-Stream Transformer Architecture]
        direction TB
        subgraph Stream_Spatial [Stream 1: Spatial Expert]
            CN["<b>ConvNeXt-Tiny</b><br/>(Pretrained Backbone)"]:::stream
            Proj1["Linear Projection<br/>(768 → 512 dim)"]:::stream
        end
        subgraph Stream_Spectral [Stream 2: Spectral Expert]
            Concat1["Concatenate<br/>FFT + DWT"]:::stream
            MLP1["<b>MLP Block</b><br/>(Linear+LayerNorm+GELU)"]:::stream
        end
        subgraph Stream_Edge [Stream 3: Edge Expert]
            Concat2["Concatenate<br/>ABCD + Edge Vectors"]:::stream
            MLP2["<b>MLP Block</b><br/>(Linear+LayerNorm+GELU)"]:::stream
        end
        ImgRes --> CN --> Proj1
        FFT & DWT --> Concat1 --> MLP1
        ABCD & Sobel --> Concat2 --> MLP2
        
        subgraph Fusion_Engine [Transformer Fusion Engine]
            TokenStack["Token Stacking<br/>[CLS, Spatial, Spectral, Edge]"]:::fusion
            PosEmbed["Add Positional<br/>Embeddings"]:::fusion
            TransEnc["<b>Transformer Encoder</b><br/>(4 Layers, 8 Heads)"]:::fusion
            AttnPool["<b>Attention Pooling</b><br/>(Weighted Sum)"]:::fusion
        end
        Proj1 --> TokenStack
        MLP1 --> TokenStack
        MLP2 --> TokenStack
        TokenStack --> PosEmbed --> TransEnc --> AttnPool
    end

    subgraph Outputs [Multi-Task Prediction Heads]
        direction TB
        MainHead("<b>Melanoma Classification</b><br/>7 Classes"):::output
        AuxHead1("Aux: Asymmetry Score"):::output
        AuxHead2("Aux: Border Score"):::output
        AuxHead3("Aux: Color Score"):::output
    end

    AttnPool --> MainHead
    AttnPool --> AuxHead1
    AttnPool --> AuxHead2
    AttnPool --> AuxHead3
