# 🧬DNA Sequence Similarity Checker

This project provides a visual and interactive way to understand the **Longest Common Subsequence (LCS)** algorithm using **Dynamic Programming**.  
It is designed mainly for DNA/RNA sequences (A, T, G, C), but can work on any short strings.

---

## 🔍 What This App Does

- Shows the **DP matrix** used to compute the LCS  
- Highlights **matches**, **mismatches**, and the **final LCS path**  
- Lets you choose **sample sequences** or enter your own  
- Displays:
  - LCS length  
  - LCS sequence  
  - Percentage match  
- Generates a **step-by-step animation** of how the DP table is filled  
  *(for short strings)*  

---

## 🧑‍💻 What Input You Provide

You can give:

- String A → e.g., `ATGCGTAG`  
- String B → e.g., `GTACGTA`  

(Only A, T, G, C are allowed if using DNA mode)

Or simply select from the built-in sample dataset.

---

## 📦 How to Run

1. Make sure you have Python installed on your system. You can download it from https://www.python.org/downloads/.
2. Open a terminal or command prompt.
3. Navigate to the directory where your `requirements.txt` file is located using the `cd` command. For example:
   ```
   cd path/to/your/directory
   ```
4. Install the required packages using pip by running the following command:
   ```
   pip install -r requirements.txt
   ```
5. Once the installation is complete, you can run your Streamlit app. If your app file is named `app.py`, use the following command:
   ```
   streamlit run app.py
   ```
6. Your default web browser should automatically open a new tab displaying your Streamlit app. If it doesn't, you can manually open your browser and go to the URL provided in the terminal (usually `http://localhost:8501`).

---

## 📁 Code Included

The code provided contains:

- Full LCS DP implementation  
- Path reconstruction  
- Styled DP table rendering  
- Gif animation generator  
- Streamlit UI with interactive inputs  

Just run it and start exploring how LCS works visually!

---
