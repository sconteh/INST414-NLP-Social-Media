# 🧠 INST414 - Hate Speech Classification with HateBERT & GPT-4.1-Nano
INST414 - Data Science Techniques | Sprint 4 Final Project
Author: Samuel Conteh
Instructor: Professor Nguyen-Le

## 📌 Project Overview
This project explores the application of Natural Language Processing (NLP) methods to detect and categorize hate speech in social media. Using the Hate Speech and Offensive Language dataset from Davidson et al., we leverage both a fine-tuned transformer model (HateBERT) and OpenAI's GPT-4.1-nano API to classify tweets and uncover nuanced hate speech subcategories. 

The aim is to highlight the challenges of hate speech moderation, the limitations of static annotation, and the potential use of large language models in augmenting content moderation strategies.

## 🧪 Methods Summary

1. Dataset Used - Hate Speech and Offensive Language Dataset 
Source: Davidson et al. (2017)
Size: 24,783 tweets
Labels: hate speech (0), offensive (1), neither (2)

2. Preprocessing

Tweets were cleaned by removing mentions (@user), hyperlinks and any excess whitespace and non-alphanumerics

3. Classification

First round of classification was done using the HateBERT (GroNLP / CNERG model). It classified the tweets into three of the following:
- Hate Speech
- Offensive Speech
- Neither

OpenAI GPT-4.1-nano was then used to create subcategorize for hate speech category:
- Racism
- Sexism / Misogyny
- Homophobia / Transphobia
- Religious Hate
- Xenophobia
- Ageism
- Ableism

4. Evaluation Metrics
- Confusion Matrix (raw & normalized)
- Classification Report (Precision, Recall, F1-Score)
- Cohen’s Kappa Score
- Label distribution visualizations

## 📈 Key Results

- HateBERT accurately identifies offensive content but tends to overpredict the "offensive" label.
- OpenAI GPT provided interpretable subcategories for hate speech, showing promise for practical moderation tools.
- Low agreement (Cohen’s Kappa = ~0.16) between annotators and HateBERT reveals potential for model bias or label ambiguity.

## 📊 Visual Outputs
All plots are saved in the /plots/ directory. Highlights include:
- Annotator vs. HateBERT prediction comparison
- Confusion matrices (raw & percentage)
- TF-IDF keyword plots
- Subcategory frequency bar chart

## 💬 Key Takeaways
- Hate speech is contextual and not always textually explicit.
- Subcategorizing hate speech improves interpretability for moderation.
- Annotator disagreement suggests that human-labeled datasets alone are insufficient.
- LLMs like GPT can be used to augment human labeling with contextual granularity.


## 🔐 Reproducibility
This project uses the following tools and libraries:

- Python 3.11
- Transformers (HuggingFace)
- Google Colab GPU backend
- OpenAI API (GPT-4.1-nano via openai package)
- torch, pandas, scikit-learn, seaborn, matplotlib

⚠️ Remember: The OpenAI API is used, which required payment by the token to be used 

## 🧠 Future Work

To build upon this project, some of the things that I or anyone can do is:
- Expand subcategory classification to all ~8,370 hate speech tweets
- Fine-tune GPT or smaller LLMs for cost-effective labeling
- Analyze annotator-model disagreement to refine labeling guidelines
- Apply active learning or weak supervision to optimize label correction over time

## 🧾 Citation

To find all of the works used in this project, you can go to the Sprint 4 document that I have created. It has the Works Cited as well as some of the Annotated Bibliographies. 
