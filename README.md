# TB-Guard: Multimodal Agentic AI for Tuberculosis Decision Support

TB-Guard is an educational multimodal decision-support prototype for tuberculosis (TB). It combines a chest X-ray classifier, a clinical tabular risk model, a WHO-catalogue-style genomic resistance expert, FAISS retrieval-augmented generation (RAG), LangGraph orchestration, LLM reviewers, and a Gradio interface.

> **Important:** This project is not a medical diagnostic system. It is a class/research prototype. Any TB or resistance interpretation must be confirmed by qualified clinicians, radiology review, microbiological testing, molecular testing, culture, and drug-susceptibility testing (DST).



## 1. What the system does

The final Notebook 4 app accepts or uses:

1. A chest X-ray image
2. Structured clinical information
3. Genomic/DNA marker flags

It runs these through:

- **CXR Agent:** EfficientNet-B0 chest X-ray classifier
- **Clinical Agent:** balanced ExtraTrees clinical risk model
- **Genomic/DNA Agent:** WHO-catalogue-style rule expert
- **RAG Layer:** FAISS vector database over TB, MDR-TB, pre-XDR, human-in-loop, and WHO mutation-catalogue text
- **LLM Reviewer Council:** GPT reviewer, Gemini reviewer, and Claude safety reviewer, with fallback mode
- **Debate + Judge Agent:** combines evidence, identifies disagreement/resistance concern, and produces a final decision-support report
- **Gradio App:** shows all intermediate outputs, RAG evidence, final report, and downloadable logs



## 2. Architecture summary

```text
Patient Inputs
│
├── Chest X-ray image
│   └── CXR Agent: EfficientNet-B0 → TB probability/class/confidence
│
├── Clinical tabular data
│   └── Clinical Agent: ExtraTrees pipeline → clinical TB risk
│
└── Genomic marker flags
    └── WHO Genomic Expert → resistance category + marker evidence

All expert outputs
    ↓
FAISS RAG retrieval
    ↓
LLM reviewer council
    ↓
Debate / disagreement summary
    ↓
Judge agent
    ↓
Final report + logs + downloadable outputs
```

The three modality agents are logically independent. In the Colab notebook they run sequentially for clean logs and debugging, but they can be parallelized later.



## 3. Repository structure



```text
TBGuard/
│
├── README.md
│
├── notebooks/
│   ├── TBGuard_1_CXR_Model_Training.ipynb
│   ├── TBGuard_2_Clinical_Model.ipynb
│   ├── TBGuard_3_WHO_Genomic_Rule_Expert.ipynb
│   └── TBGuard_4_App_RAG_LangGraph_Gradio_Last.ipynb
│
├── artifacts_to_upload/
│   ├── clinical_model_artifacts.zip
│   ├── cxr_model_artifacts.zip
│   └── model3_genomic_artifacts.zip
│
├── demo_outputs/
│   ├── sample_final_report.md
│   ├── sample_logs.json
│   └── screenshots/
│
└── documents/
    ├── TBGuard_IEEE_Project_Paper.tex
    └── TBGuard_Video_Demo_Instructions.tex
```

Do not commit API keys, Kaggle tokens, or Colab secrets.



## 4. Required artifact ZIP files for Notebook 4

Notebook 4 expects three uploaded ZIP files or direct files.

### 4.1 CXR artifact ZIP

Upload:

```text
cxr_model_artifacts.zip
```

It should contain files like:

```text
demo_images/
    clear_normal_0.png
    clear_tb_1.png
    patient1_low_suspicion_normal.png
    patient2_likely_tb.png
    patient3_mdr_context_tb.png
cxr_demo_image_scores.csv
cxr_efficientnet_b0_tb.pt
cxr_fine_tune_history.csv
cxr_metadata.csv
cxr_model_metadata.json
cxr_test_probability_scores.csv
cxr_threshold_search.csv
cxr_training_history.csv
test.csv
train.csv
val.csv
```

The required file is:

```text
cxr_efficientnet_b0_tb.pt
```

The `demo_images/` folder is strongly recommended because it gives the app clear demo X-rays.

### 4.2 Clinical artifact ZIP

Upload:

```text
clinical_model_artifacts.zip
```

It should contain:

```text
clinical_model_comparison_results.csv
clinical_model_metadata.json
clinical_tb_model.joblib
clinical_threshold_search.csv
```

The required file is:

```text
clinical_tb_model.joblib
```

### 4.3 Genomic artifact ZIP

Upload:

```text
model3_genomic_artifacts.zip
```

It should contain:

```text
genomic_resistance_model.joblib
genomic_resistance_model_metadata.json
who_mutation_catalogue_rag_text.txt
```

The required file is:

```text
genomic_resistance_model.joblib
```

The recommended RAG file is:

```text
who_mutation_catalogue_rag_text.txt
```



## 5. How to run Notebook 4 in Colab

### Step 1: Open Notebook 4

Open:

```text
notebooks/TBGuard_4_App_RAG_LangGraph_Gradio_Last.ipynb
```

Use a regular Colab runtime. A T4 GPU is enough. CPU can also run the demo, but CXR inference may be slower.

### Step 2: Install packages

Run the first cell. It installs Gradio, LangChain, LangGraph, FAISS, Hugging Face embeddings, and LLM integration packages.

### Step 3: Upload artifact ZIPs

When prompted, upload:

```text
clinical_model_artifacts.zip
cxr_model_artifacts.zip
model3_genomic_artifacts.zip
```

The notebook recursively extracts the ZIPs and searches for:

```text
cxr_efficientnet_b0_tb.pt
clinical_tb_model.joblib
genomic_resistance_model.joblib
who_mutation_catalogue_rag_text.txt
```

It then copies them into the expected project folders.

### Step 4: Configure API keys

The notebook can run in fallback mode if API keys are absent, but for full LLM reviewer output, configure keys in Colab Secrets.

In Colab:

```text
Left sidebar → Secrets/key icon → Add new secret
```

Use these exact names:

```text
OPENAI_API_KEY
GOOGLE_API_KEY
ANTHROPIC_API_KEY
```

Then toggle notebook access ON for each key you want to use.

The notebook reads them with:

```python
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY") or ""
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY") or ""
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY") or ""
```

Never hardcode keys into the notebook.

### Step 5: Upload or use demo X-ray images

Notebook 4 may ask for three demo X-ray images. Recommended:

```text
clear_normal_0.png
clear_tb_1.png
patient1_low_suspicion_normal.png
```

For the cleanest demonstration, use:

```text
Patient 1 → clear_normal_0.png
Patient 2 → clear_tb_1.png
Patient 3 → clear_tb_1.png plus rpoB/katG/gyrA genomic markers
```

The current recorded run also demonstrates a useful discordance case where Patient 3 has low CXR suspicion but high genomic resistance concern.

### Step 6: Build RAG

The notebook builds local RAG from text files such as:

```text
tb_overview.txt
drug_sensitive_tb.txt
mdr_tb.txt
pre_xdr_tb.txt
human_in_loop.txt
who_mutation_catalogue_rag_text.txt
```

It uses:

```text
RecursiveCharacterTextSplitter
sentence-transformers/all-MiniLM-L6-v2
FAISS
```

### Step 7: Load models and run sanity checks

The notebook loads:

```text
cxr_efficientnet_b0_tb.pt
clinical_tb_model.joblib
genomic_resistance_model.joblib
```

It then checks clinical and genomic artifact compatibility and can run a sample CXR, clinical, and genomic prediction.

### Step 8: Run LLM reviewer sanity check

The notebook checks GPT, Gemini, and Claude reviewer functions. If an API fails, fallback mode is used so the graph can still run.

### Step 9: Compile and run LangGraph

The graph contains these nodes:

```text
xray_agent
clinical_agent
genomic_agent
rag_retrieval
gpt_reviewer
gemini_reviewer
claude_safety_reviewer
debate_agent
judge_agent
report_generator
```

### Step 10: Launch Gradio

Run the final Gradio cell. It creates tabs for:

```text
Single Patient Workflow
Upload 1–3 X-ray Images
Run All 3 Demo Patients
```

Use the public Gradio link for demo if needed.



## 6. Demo interpretation

### Case 1: Low image suspicion / possible TB due to clinical risk

Example recorded output:

```text
CXR probability: 0.1748
CXR class: Normal/Low TB
Clinical risk: 0.5755
Genomic: no major resistance marker
Final: Possible TB
```

Interpretation: The X-ray is low risk, but the clinical model raises a screening flag. The system stays cautious and requires confirmatory testing.

### Case 2: Likely drug-sensitive TB

Example recorded output:

```text
CXR probability: 0.8033
CXR class: TB
Clinical risk: 0.6756
Genomic: no major resistance marker
Final: Likely TB
```

Interpretation: Image and clinical signals agree. Genomic markers do not suggest major resistance, so the report says no major genomic resistance marker detected.

### Case 3: MDR/pre-XDR genomic concern

Example marker pattern:

```text
rpoB = 1
katG = 1
gyrA = 1
```

Interpretation:

```text
rpoB → rifampicin resistance concern
katG → isoniazid resistance concern
gyrA → fluoroquinolone resistance concern
```

Together these raise possible MDR/pre-XDR concern, but the result must be confirmed with laboratory drug-susceptibility testing.



## 7. Deployment notes

### Colab/Gradio live demo

This is the easiest option. Run Notebook 4 and use:

```python
demo.launch(share=True)
```

The Gradio link is temporary.

### Hugging Face Spaces or cloud deployment

For a more permanent deployment, convert the final notebook into an `app.py` and upload the artifact files to the Space or cloud storage.

Required secrets/environment variables:

```text
OPENAI_API_KEY
GOOGLE_API_KEY
ANTHROPIC_API_KEY
```

Required model/artifact files:

```text
cxr_efficientnet_b0_tb.pt
clinical_tb_model.joblib
genomic_resistance_model.joblib
who_mutation_catalogue_rag_text.txt
```

Recommended folders/files:

```text
demo_images/
rag_sources/
```




## 8. Limitations

- The system is not a medical diagnostic system.
- The CXR model outputs probability/class, not localized radiology findings.
- The clinical model has high sensitivity but low precision, so it is a screening signal only.
- The genomic module is rule-based and needs laboratory DST confirmation.
- The RAG knowledge base is compact and demo-focused.
- LLM reviewers can hallucinate, so outputs are grounded with RAG and human-in-loop warnings.
- API rate limits or missing keys may trigger fallback mode.
- Current LangGraph execution is sequential for readability; future work can add true conditional edges and parallel execution.




## 9. License / use statement

This repository is for educational use only. It is not validated for clinical diagnosis, treatment planning, or real patient care.
