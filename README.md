# Tri-Stream Spectral–Spatial Fusion Transformer for Melanoma Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q86wactcHUrKk_dH1PIQvEO7wzMlLqM3?usp=sharing)

## Overview
This repository contains the implementation of a novel deep learning framework for automated skin lesion classification. Standard Convolutional Neural Networks (CNNs) often miss complex boundaries and texture frequencies in dermatological images. To overcome this, we designed a **Tri-Stream Architecture** that extracts spatial, spectral, and geometric features, dynamically fusing them via a Transformer-based self-attention engine.

This model accurately classifies 7 distinct types of skin lesions (including melanoma) to assist in early and reliable automated clinical diagnosis, robustly handling severe dataset class imbalances.

## System Architecture

The model utilizes three parallel "expert" feature extraction streams, which are then fused using a Transformer:
1. **Spatial Stream (Visual Expert):** Uses a pre-trained **ConvNeXt-Tiny** backbone to extract high-level semantic features, global shape, and color distribution.
2. **Spectral Stream (Frequency Expert):** Utilizes Fast Fourier Transform (**FFT**) and Discrete Wavelet Transform (**DWT**) to capture sub-visual texture and multi-scale frequency patterns.
3. **Edge Stream (Geometric Expert):** Mathematically models the clinical **ABCD rule** via manual feature extraction. It generates a 20-dimensional vector capturing Asymmetry (color histograms), Border irregularity (Sobel gradients), Compactness/Shape, and Edge orientation.
4. **Transformer Fusion:** The tokens from these three streams are dynamically weighted using a 4-Layer Transformer Encoder, passing the `[CLS]` token to a single, highly optimized 7-class prediction head.


```mermaid
graph TB
    subgraph Input["INPUT STAGE"]
        A[Raw Dermoscopic Image<br/>HAM10000 Dataset<br/>Variable Size × 3 RGB]
    end

    subgraph Preprocessing["PREPROCESSING PIPELINE"]
        B1[Hair Removal<br/>Morphological Ops<br/>Blackhat + Inpainting]
        B2[Macenko Color Normalization<br/>Per-channel Z-score<br/>Target: μ=0.485,0.456,0.406]
        B3[Otsu Lesion Segmentation<br/>Gaussian Blur → Threshold<br/>Morphological Close+Open]
        B4[Resize with Padding<br/>Center crop to 224×224<br/>Maintain aspect ratio]
        
        A --> B1
        B1 --> B2
        B2 --> B3
        B3 --> B4
    end

    subgraph FeatureExtraction["FEATURE EXTRACTION - 3 PARALLEL STREAMS"]
        
        subgraph Stream1["STREAM 1: SPATIAL FEATURES"]
            C1[ConvNeXt-Tiny Backbone<br/>Pretrained ImageNet-1K<br/>28.6M params]
            C2[Global Average Pooling<br/>7×7×768 → 768-dim]
            C3[Spatial Projection Head<br/>Linear: 768→512<br/>LayerNorm + GELU<br/>393K params]
            
            B4 --> C1
            C1 -->|768 dims| C2
            C2 --> C3
            C3 -->|512-dim<br/>Spatial Token| FUSION
        end
        
        subgraph Stream2["STREAM 2: SPECTRAL FEATURES"]
            D1[FFT Extraction<br/>Grayscale 2D FFT<br/>Center crop 112×112<br/>+ RGB freq stats<br/>12,562 dims]
            D2[DWT Extraction<br/>Haar Wavelet<br/>2 Levels × 3 Channels<br/>24 dims]
            D3[Concatenate<br/>FFT + DWT<br/>12,586 dims]
            D4[Spectral Encoder Layer 1<br/>Linear: 12,586→2,048<br/>LayerNorm + GELU<br/>Dropout 0.15<br/>25.7M params]
            D5[Spectral Encoder Layer 2<br/>Linear: 2,048→512<br/>LayerNorm<br/>1.0M params]
            
            B4 --> D1
            B4 --> D2
            D1 --> D3
            D2 --> D3
            D3 --> D4
            D4 --> D5
            D5 -->|512-dim<br/>Spectral Token| FUSION
        end
        
        subgraph Stream3[" STREAM 3: EDGE & GEOMETRIC FEATURES"]
            E1[Gradient Features<br/>Sobel X/Y + Laplacian<br/>Border statistics<br/>4 dims]
            E2[Shape Features<br/>Compactness, Circularity<br/>Solidity, Convexity<br/>Fractal Dimension<br/>5 dims]
            E3[Asymmetry Features<br/>PCA-based splitting<br/>Color histogram χ²<br/>3 dims]
            E4[Orientation Histogram<br/>Edge direction<br/>8-bin histogram<br/>8 dims]
            E5[Concatenate All<br/>Total: 20 dims]
            E6[Edge Encoder Layer 1<br/>Linear: 20→128<br/>LayerNorm + GELU<br/>2.6K params]
            E7[Edge Encoder Layer 2<br/>Linear: 128→256<br/>LayerNorm + GELU<br/>33K params]
            E8[Edge Encoder Layer 3<br/>Linear: 256→512<br/>LayerNorm<br/>131K params]
            
            B4 --> E1
            B3 --> E1
            B4 --> E2
            B3 --> E2
            B4 --> E3
            B4 --> E4
            
            E1 --> E5
            E2 --> E5
            E3 --> E5
            E4 --> E5
            E5 --> E6
            E6 --> E7
            E7 --> E8
            E8 -->|512-dim<br/>Edge Token| FUSION
        end
    end

    subgraph TransformerFusion[" TRANSFORMER FUSION MODULE"]
        F1[Token Preparation<br/>CLS Token: learnable 512-dim<br/>Spatial Token: 512-dim<br/>Spectral Token: 512-dim<br/>Edge Token: 512-dim]
        F2[Add Positional Embeddings<br/>Learned embeddings<br/>Shape: 1×4×512]
        F3[Token Sequence<br/>Shape: Batch×4×512<br/>CLS, Spatial, Spectral, Edge]
        
        F4[Transformer Encoder Layer 1<br/>━━━━━━━━━━━━━━━━━━━━━<br/>Pre-Norm Architecture<br/>━━━━━━━━━━━━━━━━━━━━━<br/>LayerNorm<br/>Multi-Head Self-Attention<br/>8 heads, head_dim=64<br/>Dropout 0.1<br/>Residual Connection<br/>━━━━━━━━━━━━━━━━━━━━━<br/>LayerNorm<br/>Feed-Forward Network<br/>Linear: 512→1024→512<br/>GELU + Dropout 0.1<br/>Residual Connection]
        
        F5[Transformer Encoder Layer 2<br/>━━━━━━━━━━━━━━━━━━━━━<br/>Same structure as Layer 1<br/>━━━━━━━━━━━━━━━━━━━━━<br/>2.1M total params<br/>for both layers]
        
        F6[CLS Token Extraction<br/>Extract index 0<br/>Shape: Batch×512<br/>NOT weighted pooling]
        
        FUSION --> F1
        F1 --> F2
        F2 --> F3
        F3 --> F4
        F4 --> F5
        F5 --> F6
    end

    subgraph Classification["CLASSIFICATION HEAD"]
        G1[LayerNorm<br/>512 dims]
        G2[Linear Layer 1<br/>512→256<br/>GELU activation]
        G3[Dropout<br/>p = 0.2]
        G4[Linear Layer 2<br/>256→7<br/>Output logits]
        G5[Softmax<br/>inference only]
        
        F6 --> G1
        G1 --> G2
        G2 --> G3
        G3 --> G4
        G4 --> G5
    end

    subgraph Output["OUTPUT"]
        H1["7-Class Probabilities<br/>━━━━━━━━━━━━━━━━<br/>Class 0: BKL Benign Keratosis<br/>Class 1: NV Melanocytic Nevi<br/>Class 2: DF Dermatofibroma<br/>Class 3: MEL Melanoma ⚠️<br/>Class 4: VASC Vascular Lesions<br/>Class 5: BCC Basal Cell Carcinoma"]
    end

    G5 --> H1
