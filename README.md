# Skincare Sage

**Skincare Sage** is an intelligent skincare advisor that leverages **Deep Learning** to predict skin parameters such as acne severity and skin type based on a user's image input. Additionally, users can input other skin concerns, and Skincare Sage will provide personalized recommendations for essential skincare products tailored to their needs.

---

## Table of Contents
1. [Features](#-features)
2. [Prerequisites](#-prerequisites)
3. [Installation](#-installation)
4. [Usage](#-usage)
5. [Future Enhancements](#-future-enhancements)

---

## Features
- Image Analysis: Predicts acne severity and skin type from user-uploaded images.
- Personalized Recommendations: Suggests skincare products based on analysis results and user concerns.
- User Interaction: Allows users to specify additional skin concerns for more tailored recommendations.
- Local Setup: Can be easily installed and run on a local system for convenient usage.

---

## Prerequisites
- Python 3.x installed on your machine.
- Pip package manager installed.

---

## Installation
1. Clone the **Skincare Sage** repository:
   ```bash
   git clone https://github.com/KaviyaaPriyadharshini/SKINCARE
   ```
2. Navigate to the project directory:
   ```bash
   cd skin
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Optional: Create and run in a virtual environment)*
   ```bash
   python3 -m venv myenv
   ```
   - **Windows**:
     ```bash
     myenv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source myenv/bin/activate
     ```

---

## Usage
1. Ensure you have a suitable image of your face for analysis or take one directly in the app.
2. Run the Skincare Sage application:
   ```bash
   streamlit run app.py
   ```
3. Follow the on-screen prompts to:
   - Upload your image
   - Input any additional skin concerns
4. Receive **personalized skincare recommendations** based on the analysis results.

---

## Future Enhancements
- Integrate **NLP-based skincare queries** for better user interaction.
- Expand the model for **more diverse skin types and conditions**.
- Optimize performance with **faster inference techniques**.
