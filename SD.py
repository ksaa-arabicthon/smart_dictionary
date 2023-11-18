import streamlit as st
import joblib
import csv
import difflib
import gensim
from sentence_transformers import SentenceTransformer
import torch
from torch.nn.functional import cosine_similarity
import pickle
from streamlit_text_rating.st_text_rater import st_text_rater
from spellchecker import SpellChecker
import os
import base64
import requests

# Initialize session_state
if 'response_generated' not in st.session_state:
    st.session_state.response_generated = False

# Function to load Arabic word list from CSV
def remove_diacritics(word):
    diacritics = ['\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650', '\u0651', '\u0652', '\u0653', '\u0654', '\u0655']
    for diacritic in diacritics:
        word = word.replace(diacritic, '')
    return word

def load_arabic_wordlist(csv_file_path):
    words = []
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if row:
                word = remove_diacritics(row[0])
                words.append(word)
    return words


# Function to create a custom spellchecker
def create_custom_spellchecker(word_list):
    spellchecker = SpellChecker()
    spellchecker.word_frequency.load_words(word_list)
    return spellchecker

# Function to check and correct spelling
def check_spelling(spellchecker, word):
    return spellchecker.correction(word)

import streamlit as st
import requests

# Function to search API
def search_api(query):
    api_key = st.secrets["siwar_api"]["key"]  # Accessing the API key from secrets

    url = f"https://siwar.ksaa.gov.sa/api/alriyadh/search?query={query}"
    headers = {
        "accept": "application/json",
        "apikey":  api_key  # Using the API key from secrets
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        results = response.json()
        definitions = []
        for result in results:
            first_sense = result.get("senses", [{}])[0]
            definition = first_sense.get("definition", {}).get("textRepresentations", [{}])[0].get("form", "N/A")
            definitions.append(definition)
            if len(definitions) == 5:
                break
        return definitions
    else:
        st.error(f"Failed to fetch data: {response.status_code}")
        return None

# Function to convert image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as file:
        encoded_string = base64.b64encode(file.read()).decode()
    return f"data:image/png;base64,{encoded_string}"

# Load your images
background_image_base64 = get_base64_of_bin_file("web_design/wallpaper.png")
logo_image_base64 = get_base64_of_bin_file("web_design/APPlogo.png")
footer_image_base64 = get_base64_of_bin_file("web_design/footer.png")

# Define badges and their criteria
badges = {
    "لقد حصلت على شارة المتفاعل": {"score_required": 50},
    "لقد حصلت على شارة المتفاعل المتقدم" : {"score_required": 150},
    "لقد حصلت على شارة المتفاعل الخبير": {"score_required": 300},
}

# CSV file path for user data
user_data_file = 'user_data.csv'

def custom_st_write(text):
    custom_write_style = f"""
    <style>
        .custom-st-write {{
            text-align: center; /* Center text */
            font-weight: bold; /* Make text bold */
            color: #08707a; /* Custom text color */
            font-size: 20px; /* Custom font size */
            margin: 20px auto; /* Center div horizontally */
            padding: 10px; /* Add padding */
            border-radius: 10px; /* Rounded corners */
            background-color: #f0f0f0; /* Light background for the text */
            display: block; /* Use block to enable margin auto */
            width: fit-content; /* Fit the width to the content */
        }}
    </style>
    <div class='custom-st-write'>{text}</div>
    """
    st.markdown(custom_write_style, unsafe_allow_html=True)


def read_user_data():
    if not os.path.exists(user_data_file):
        return {}
    with open(user_data_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return {row['user']: {'name': row['name'], 'email': row['email'], 'input': row['input'], 'response': row['response'], 'likes': int(row['likes']), 'dislikes': int(row['dislikes']), 'notes': row['notes'], 'score': int(row['score']), 'badges': row['badges'].split('|')} for row in reader}

def write_user_data(user_data):
    with open(user_data_file, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['user', 'name', 'email', 'input', 'response', 'likes', 'dislikes', 'notes', 'score', 'badges']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for user, data in user_data.items():
            writer.writerow({'user': user, 'name': data['name'], 'email': data['email'], 'input': data['input'], 'response': data['response'], 'likes': data['likes'], 'dislikes': data['dislikes'], 'notes': data['notes'], 'score': data['score'], 'badges': '|'.join(data['badges'])})


# Load initial user data
user_data = read_user_data()

# Load ML model and vectorizer for Aammi words
@st.cache_data(ttl=3600)
def load_aammi_model():
    model = joblib.load('ML/diaModel.pkl')
    vectorizer = joblib.load('ML/tfidf_vectorizer.pkl')
    return model, vectorizer

model_aammi, vectorizer_aammi = load_aammi_model()

# Load Word2Vec models for synonyms
@st.cache_resource(ttl=3600)
def load_gensim_models():
    gensim_model1 = gensim.models.Word2Vec.load('full_uni_cbow_100_twitter/full_uni_cbow_100_twitter.mdl')
    gensim_model2 = gensim.models.Word2Vec.load('full_uni_cbow_100_wiki/full_uni_cbow_100_wiki.mdl')
    return gensim_model1, gensim_model2

gensim_model1, gensim_model2 = load_gensim_models()

# Replace 'file_path' with the actual path to your 'sample_data.pickle' file
file_path = 'data/sample_data.pickle'

@st.cache_resource(ttl=3600)
def load_list_data(file_path):
    with open(file_path, 'rb') as handle:
        list_data = pickle.load(handle)
    return list_data

list_data = load_list_data(file_path)

# Function to read data from CSV for definitions
@st.cache_resource(ttl=3600)
def read_data_from_csv(file_path):
    data = {}
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row['word']
            definition = row['def']
            data[key] = definition
    return data

# Function for ML Prediction for Aammi words
@st.cache_data(ttl=3600)
def predict_faseh(aammi_word):
    vectorized_word = vectorizer_aammi.transform([aammi_word])
    prediction = model_aammi.predict(vectorized_word)
    return prediction[0]

# Function to read data from CSV for Mosstarbi words
@st.cache_resource(ttl=3600)
def read_mosstarbi_data():
    data = {}
    with open('data/Mostaarib.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row['Mostaarib']
            row_data = {col: val for col, val in row.items() if col != 'Mostaarib'}
            data[key] = row_data
    return data

# Function to read data from CSV for Aammi words
@st.cache_resource(ttl=3600)
def read_aammi_data():
    data = {}
    with open('data/Aammi_Faseh_Pairs.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row['Aammi']
            row_data = {col: val for col, val in row.items() if col != 'Aammi'}
            data[key] = row_data
    return data

# Function to find Mosstarbi word details
@st.cache_data(ttl=3600)
def find_word_details(input_word, data):
    closest_matches = difflib.get_close_matches(input_word, data.keys())
    if closest_matches:
        return data[closest_matches[0]]
    else:
        return None

# Function to find synonyms using Word2Vec model
@st.cache_data(ttl=3600)
def find_synonyms(_model, input_word):
    try:
        synonyms = _model.wv.most_similar(input_word, topn=20)
        return [syn[0] for syn in synonyms]
    except KeyError:
        return []

embedder = SentenceTransformer('all-MiniLM-L12-v2')
# Function to find reverse definition using Sentence Transformers
def find_reverse_definition(_list_data, input_text):
    emb = embedder.encode(input_text)
    similarities = {word['word']: cosine_similarity(torch.tensor(emb), word["emb"], dim=0) for word in _list_data}
    best_match_word = max(similarities, key=similarities.get)
    return best_match_word

# Function to update user score and badges
def update_user(user, score_increase):
    if user in user_data:
        user_data[user]["score"] += score_increase
        for badge, criteria in badges.items():
            if user_data[user]["score"] >= criteria["score_required"] and badge not in user_data[user]["badges"]:
                user_data[user]["badges"].append(badge)
    else:
        user_data[user] = {"score": score_increase, "badges": []}
    write_user_data(user_data)

# Function to handle rating
def rate_response(user, response, user_input, app_response):
    if user in user_data:
        user_data[user]["input"] = user_input
        user_data[user]["response"] = app_response
        if response == "liked":
            user_data[user]["likes"] += 1
            user_data[user]["score"] += 10
        elif response == "disliked":
            user_data[user]["dislikes"] += 1
            user_data[user]["score"] += 5
        # Update badges
        for badge, criteria in badges.items():
            if user_data[user]["score"] >= criteria["score_required"] and badge not in user_data[user]["badges"]:
                user_data[user]["badges"].append(badge)
        write_user_data(user_data)
        return "شكرا لك .. سيتم إرسال رسالة إلى المعجم لإضافة هذا المدخل" if response == "liked" else "سيتم مراجعة الإجابة."
    else:
        return "المستخدم غير موجود."

csv_file_path = 'data/correction.csv'  # Replace with your CSV file path
arabic_words = load_arabic_wordlist(csv_file_path)
custom_spellchecker = create_custom_spellchecker(arabic_words)

# Streamlit app layout
def handle_rating():
    # CSS to customize text input
    text_input_style = """
    <style>
        /* Target the text input widget */
        .stTextInput input {
            color: #08707a; /* Text color */
            background-color: #f0f0f0; /* Background color */
            border-radius: 10px; /* Rounded corners */
            border: 2px solid #08707a; /* Border color and width */
            padding: 10px; /* Inner space */
            font-size: 30px; /* Font size for input field */
        }

        /* Custom style for markdown label */
        .markdown-label {
            text-align: center;
            color: #08707a; /* Label color */
            font-size: 30px; /* Font size for markdown label */
            font-family: 'Almarai', sans-serif; /* Custom Google Font */
        }
    </style>
    """
    st.markdown(text_input_style, unsafe_allow_html=True)

    # Custom label using markdown
    st.markdown('<div class="markdown-label">أدخل الكلمة أو الجملة</div>', unsafe_allow_html=True)

    # Initialize session state for input text and potential correction
    if 'corrected_input' not in st.session_state:
        st.session_state['corrected_input'] = None


    # Text input without the default label
    input_text = st.text_input("", key="input_text")

    # CSS to center radio buttons
    radio_style = """
    <style>

        div.custom-label {
            color: #08707a; /* Text color */
            font-size: 40px; /* Font size */
            font-weight: bold; /* Font weight */
            text-align: center; /* Center alignment */
        }

        /* Style for the radio button container */
        div.row-widget.stRadio {
            display: flex;
            justify-content: center;
            color: #08707a; /* Text color */
            font-size: 40px; /* Font size */
            font-weight: bold; /* Font weight */
            text-align: center; /* Center alignment */
        }

        /* Style for individual radio options */
        .stRadio > div {
            display: flex;
            flex-direction: column;
        }
    </style>
    """
    st.markdown(radio_style, unsafe_allow_html=True)

    # Custom label using Markdown
    st.markdown('<div class="custom-label">اختر الخدمة</div>', unsafe_allow_html=True)

    # Radio buttons
    feature_option = st.radio("", ["***المدلول العكسي***", "***المدلول المعجمي***", "***النظير الكلمي***", "***النظير العامي***", "***الأصيل الكلمي***", "***الدّارج الكلمي***"])

    # Rest of your Streamlit code

    response_generated = False

    app_response = ""
    
    # CSS to style the button
    button_style = """
    <style>
        /* Style for the button container */
        div.stButton > button {
            display: block; /* Make the button a block element */
            margin: auto; /* Auto margin for horizontal centering */
            font-size: 20px; /* Increase font size */
            color: white; /* Text color */
            background-color: #08707a; /* Background color */
            padding: 10px 24px; /* Top/bottom and left/right padding */
            border: none; /* Remove border */
            border-radius: 10px; /* Rounded corners */
            cursor: pointer; /* Change cursor on hover */
        }

        /* Hover effect for button */
        div.stButton > button:hover {
            background-color: black; /* Change background color on hover to black */
        }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
        
    if st.button("ابحث:mag:"):

        if feature_option == "***المدلول المعجمي***":
            if input_text:
                corrected_word = check_spelling(custom_spellchecker, input_text)
                show_correction_button = corrected_word != input_text and corrected_word is not None
                
                if show_correction_button:
                    custom_st_write(f"هل تقصد {corrected_word}")
                    final_word = corrected_word
                    definitions = search_api(final_word)
                    if definitions:
                        for definition in definitions:
                            custom_st_write(f"تعريف: {definition}")
                    else:
                        custom_st_write("2لم يتم العثور على تعريف لهذه الكلمة أو النص")
                else:
                    final_word = input_text
            else:
                final_word = input_text

            # Execute the search with the final word
            if final_word:
                definitions = search_api(final_word)

                if definitions:
                    for definition in definitions:
                        custom_st_write(f"تعريف: {definition}")
                else:
                    custom_st_write("3لم يتم العثور على تعريف لهذه الكلمة أو النص")

        elif feature_option == "***النظير الكلمي***":
            synonyms = find_synonyms(gensim_model2, input_text)
            custom_st_write(f"مرادفات فصيحة للكلمة أو النص '{input_text}':")
            for synonym in synonyms:
                custom_st_write(synonym)

        elif feature_option == "***النظير العامي***":
            synonyms = find_synonyms(gensim_model1, input_text)
            custom_st_write(f"مرادفات عامة للكلمة أو النص '{input_text}':")
            for synonym in synonyms:
                custom_st_write(synonym)

        elif feature_option == "***الأصيل الكلمي***":
            # Load data for Mosstarbi words
            data = read_mosstarbi_data()
            word_details = find_word_details(input_text, data)

            if word_details:
                custom_st_write(f"المصطلح العامي للكلمة أو النص '{input_text}' هو '{word_details.get('Other', 'غير متاح')}'.")
            else:
                custom_st_write("لم يتم العثور على مطابقة")

        elif feature_option == "***الدّارج الكلمي***":
            # Load data for Aammi words
            data = read_aammi_data()
            closest_matches = difflib.get_close_matches(input_text, data.keys())
            if closest_matches:
                word_details = data[closest_matches[0]]
                custom_st_write(f"المصطلح الفصيح للكلمة أو النص '{input_text}' هو '{word_details.get('Faseh', 'غير متاح')}'.")
            else:
                # If no match found in Aammi and the dataset is Aammi, use ML model
                predicted_faseh = predict_faseh(input_text)
                custom_st_write(f"المصطلح الفصيح للكلمة أو النص '{input_text}' هو '{predicted_faseh}'.")

        elif feature_option == "***المدلول العكسي***":
            best_match_word = find_reverse_definition(list_data, input_text)
            custom_st_write(f"أقرب كلمة بالتعريف العكس للنص المدخل: '{best_match_word}'")

        st.session_state['response_generated'] = True
        st.session_state['user_input'] = input_text
        st.session_state['app_response'] = app_response

    if st.session_state['response_generated']:
        # Display the rating component
        st.write('<div class="markdown-label">يرجى تقييم الإجابة</div>', unsafe_allow_html=True)
        response = st_text_rater(text="هل تعجبك هذه الإجابة؟")        
        # Prompt for user registration to earn points
        st.markdown('<div class="markdown-label">للحصول على نقاط في برنامج إثراء المجمع يرجى التسجيل هنا</div>', unsafe_allow_html=True)
        # Custom label using markdown for each input
        st.markdown('<div class="markdown-label">أدخل اسمك:</div>', unsafe_allow_html=True)
        user_name = st.text_input("", key="user_name")

        st.markdown('<div class="markdown-label">أدخل بريدك الإلكتروني:</div>', unsafe_allow_html=True)
        user_email = st.text_input("", key="user_email")

        st.markdown('<div class="markdown-label">ملاحظات أخرى:</div>', unsafe_allow_html=True)
        user_notes = st.text_input("", key="user_notes")

        # Button to submit review
        if st.button("إرسال المراجعة"):
            # Initialize user data if new user
            if user_email and user_email not in user_data:
                user_data[user_email] = {
                    "name": user_name, 
                    "email": user_email, 
                    "input": st.session_state['user_input'], 
                    "response": st.session_state['app_response'], 
                    "likes": 0, 
                    "dislikes": 0, 
                    "notes": user_notes, 
                    "score": 0, 
                    "badges": []
                }
                write_user_data(user_data)

            # Handle the rating response
            user_id = user_email or "unknown_user"
            st.session_state['rating_message'] = rate_response(user_email, response, st.session_state['user_input'], st.session_state['app_response'])

            # Display any message after rating
            if st.session_state['rating_message']:
                custom_st_write(st.session_state['rating_message'])
                display_user_data(user_email)

def display_user_data(user):
    if user in user_data:
        custom_st_write(f"النقاط الحالية: {user_data[user]['score']}")
        if user_data[user]['badges']:
            custom_st_write("الشارات المكتسبة:")
            for badge in user_data[user]['badges']:
                custom_st_write(f"- {badge}")
        else:
            custom_st_write("لم يتم كسب أي شارة بعد.")

def main():
    # CSS to customize sidebar text
    sidebar_text_style = """
    <style>
        .sidebar-markdown-label{
            text-align: center;
            color: #08707a; /* Label color */
            font-size: 30px; /* Font size for markdown label */
            font-family: 'Almarai', sans-serif; /* Custom Google Font */
        }

        /* Additional styling can be added here */
    </style>
    """
    st.markdown(sidebar_text_style, unsafe_allow_html=True)
    st.sidebar.markdown('''<div class="sidebar-markdown-label"> <p> مرحبا بكم في </p> </div>''', unsafe_allow_html=True)
    st.sidebar.markdown(f'<img src="{logo_image_base64}" width="300">', unsafe_allow_html=True)
    st.sidebar.markdown('''
        <div class="sidebar-markdown-label">
            <p>يقدم المعجم الذكي المزايا التالية والتي تجيب على أسئلتكم باستخدام تقنيات و خوارزميات الذكاء الاصطناعي</p>
            <p>المفهوم العكسي: قم بكتابة تعريف وسيقوم التطبيق بإيجاد الكلمة الفصيحة المناسبة</p>
            <p>المدلول المعجمي: قم بكتابة كلمة فصيحة لتحصل على التعريف المناسب</p>
            <p>النظير الكلمي: قم بكتابة كلمة فصيحة وستحصل على نظائرها في اللغة العربية الفحصى</p>
            <p>النظير العامي: قم بكتابة كلمة عامية وستحصل على نظائرها في اللهجات العامية</p>
            <p>الأصيل الكلمي: قم بكتابة كلمة معربة وستحصل على الكلمة الفصحية لها</p>
            <p>الدّارج الكلمي: قم بكتابة كلمة من اللهجات العاميه وستحصل على الكملة الفصيحة لها</p>
            <p>مع تحيات فريق عمل المعجم الذكي</p>
        </div>
    ''', unsafe_allow_html=True)

    title_style = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Almarai&display=swap');

        .title {
            text-align: center;
            font-size: 70px; /* Adjust font size as needed */
            font-family: 'Almarai', sans-serif; /* Custom Google Font */
            color: #08707a; /* Specific shade of green */
            text-shadow: 2px 2px 4px #000000; /* Text shadow for depth */
        }
    </style>
    """
    st.markdown(title_style, unsafe_allow_html=True)

    # Custom title with HTML
    st.markdown(f'''
        <div style="text-align: center;">
            <img src="{logo_image_base64}" width="400">
        </div>
        ''', unsafe_allow_html=True)    #st.markdown('<div class="title">المعجم الذكي</div>', unsafe_allow_html=True)


    # CSS to inject contained in a Python multiline string
    css_style = f"""
    <style>
        /* Add CSS styling for the main page wallpaper */
        .stApp {{
            background-image: url('{background_image_base64}');
            background-size: cover;
        }}
    </style>
    """
    st.markdown(css_style, unsafe_allow_html=True)

    handle_rating()

if __name__ == "__main__":
    main()
