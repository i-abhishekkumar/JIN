from flask import Flask, render_template, request
import PyPDF2
import os

app = Flask(__name__)


from PyPDF2 import PdfReader
import tempfile
# UPLOAD_FOLDER = 'uploads'
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------

# my custome code to extract file OCR
import fitz 
from PIL import Image
import pytesseract
import io
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import spacy
import re
import json
from textblob import TextBlob
import random
import spacy
from nltk.corpus import wordnet
from collections import Counter
from collections import OrderedDict
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')




para = ""
def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    
    extracted_text = ""

    for page_number in range(pdf_document.page_count):

        page = pdf_document[page_number]

        page_text = page.get_text()

        extracted_text += page_text

    pdf_document.close()

    return extracted_text

def ocr_with_tesseract(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    return text

def extract_text_code(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)


    with open("extracted_text.txt", "w", encoding="utf-8") as text_file:
        text_file.write(pdf_text)

    image_dir = "images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Perform OCR on each page using Tesseract
    pdf_document = fitz.open(pdf_path)  
    for page_number in range(pdf_document.page_count):
        image_path = os.path.join(image_dir, f"page_{page_number + 1}.png")
        pixmap = pdf_document[page_number].get_pixmap()


        img_data = pixmap.tobytes("png")

        # Create a PIL Image from the PNG data:
        img = Image.open(io.BytesIO(img_data))  

        img.save(image_path)

        # Perform OCR on the image using Tesseract
        ocr_result = ocr_with_tesseract(image_path)
        # print(f"Text from page {page_number + 1}:\n{ocr_result}")
        global para
        para +=ocr_result

    pdf_document.close()

def OCR_code(case_info):
    pdf_path = case_info["file_path"]
    extract_text_code(pdf_path)



def summarize_document(full_text, num_sentences=3):
    document_text = full_text


    sentences = sent_tokenize(document_text)


    words = word_tokenize(document_text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

    # Calculate word frequencies
    word_freq = FreqDist(filtered_words)

    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence):
            if word.lower() in word_freq:
                if i not in sentence_scores:
                    sentence_scores[i] = word_freq[word.lower()]
                else:
                    sentence_scores[i] += word_freq[word.lower()]

    # Select the top sentences based on scores
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

    # Sort selected sentences by their original order
    top_sentences.sort()

    # Generate the summarized text
    summarized_text = ' '.join([sentences[i] for i in top_sentences])

    return summarized_text




def process_document(full_text):
    nlp = spacy.load("en_core_web_lg")

    document_text = full_text

    # Process the document using spaCy
    doc = nlp(document_text)


    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop]


    processed_text = ' '.join(lemmatized_tokens)

    return processed_text




































# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------







# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------




# Define the JSON structure
data = {
    "a": {
        "aa": {},
        "ab": {},
        "ac": {},
        "ad": {},
        "ae": {},
        "af": {},
        "ag": {},
        "ah": {},
        "ai": {},
        "aj": {},
        "ak": {},
        "al": {},
        "am": {},
        "an": {},
        "ao": {},
        "ap": {},
        "aq": {},
        "ar": {},
        "as": {},
        "at": {},
        "au": {},
        "av": {},
        "aw": {},
        "ax": {},
        "ay": {},
        "az": {}
    },
    "b": {
        "ba": {},
        "bb": {},
        "bc": {},
        "bd": {},
        "be": {},
        "bf": {},
        "bg": {},
        "bh": {},
        "bi": {},
        "bj": {},
        "bk": {},
        "bl": {},
        "bm": {},
        "bn": {},
        "bo": {},
        "bp": {},
        "bq": {},
        "br": {},
        "bs": {},
        "bt": {},
        "bu": {},
        "bv": {},
        "bw": {},
        "bx": {},
        "by": {},
        "bz": {}
    },
    "c": {
        "ca": {},
        "cb": {},
        "cc": {},
        "cd": {},
        "ce": {},
        "cf": {},
        "cg": {},
        "ch": {},
        "ci": {},
        "cj": {},
        "ck": {},
        "cl": {},
        "cm": {},
        "cn": {},
        "co": {},
        "cp": {},
        "cq": {},
        "cr": {},
        "cs": {},
        "ct": {},
        "cu": {},
        "cv": {},
        "cw": {},
        "cx": {},
        "cy": {},
        "cz": {}
    },
    "d": {
        "da": {},
        "db": {},
        "dc": {},
        "dd": {},
        "de": {},
        "df": {},
        "dg": {},
        "dh": {},
        "di": {},
        "dj": {},
        "dk": {},
        "dl": {},
        "dm": {},
        "dn": {},
        "do": {},
        "dp": {},
        "dq": {},
        "dr": {},
        "ds": {},
        "dt": {},
        "du": {},
        "dv": {},
        "dw": {},
        "dx": {},
        "dy": {},
        "dz": {}
    },
    "e": {
        "ea": {},
        "eb": {},
        "ec": {},
        "ed": {},
        "ee": {},
        "ef": {},
        "eg": {},
        "eh": {},
        "ei": {},
        "ej": {},
        "ek": {},
        "el": {},
        "em": {},
        "en": {},
        "eo": {},
        "ep": {},
        "eq": {},
        "er": {},
        "es": {},
        "et": {},
        "eu": {},
        "ev": {},
        "ew": {},
        "ex": {},
        "ey": {},
        "ez": {}
    },
    "f": {
        "fa": {},
        "fb": {},
        "fc": {},
        "fd": {},
        "fe": {},
        "ff": {},
        "fg": {},
        "fh": {},
        "fi": {},
        "fj": {},
        "fk": {},
        "fl": {},
        "fm": {},
        "fn": {},
        "fo": {},
        "fp": {},
        "fq": {},
        "fr": {},
        "fs": {},
        "ft": {},
        "fu": {},
        "fv": {},
        "fw": {},
        "fx": {},
        "fy": {},
        "fz": {}
    },
    "g": {
        "ga": {},
        "gb": {},
        "gc": {},
        "gd": {},
        "ge": {},
        "gf": {},
        "gg": {},
        "gh": {},
        "gi": {},
        "gj": {},
        "gk": {},
        "gl": {},
        "gm": {},
        "gn": {},
        "go": {},
        "gp": {},
        "gq": {},
        "gr": {},
        "gs": {},
        "gt": {},
        "gu": {},
        "gv": {},
        "gw": {},
        "gx": {},
        "gy": {},
        "gz": {}
    },
    "h": {
        "ha": {},
        "hb": {},
        "hc": {},
        "hd": {},
        "he": {},
        "hf": {},
        "hg": {},
        "hh": {},
        "hi": {},
        "hj": {},
        "hk": {},
        "hl": {},
        "hm": {},
        "hn": {},
        "ho": {},
        "hp": {},
        "hq": {},
        "hr": {},
        "hs": {},
        "ht": {},
        "hu": {},
        "hv": {},
        "hw": {},
        "hx": {},
        "hy": {},
        "hz": {}
    },
    "i": {
        "ia": {},
        "ib": {},
        "ic": {},
        "id": {},
        "ie": {},
        "if": {},
        "ig": {},
        "ih": {},
        "ii": {},
        "ij": {},
        "ik": {},
        "il": {},
        "im": {},
        "in": {},
        "io": {},
        "ip": {},
        "iq": {},
        "ir": {},
        "is": {},
        "it": {},
        "iu": {},
        "iv": {},
        "iw": {},
        "ix": {},
        "iy": {},
        "iz": {}
    },
    "j": {
        "ja": {},
        "jb": {},
        "jc": {},
        "jd": {},
        "je": {},
        "jf": {},
        "jg": {},
        "jh": {},
        "ji": {},
        "jj": {},
        "jk": {},
        "jl": {},
        "jm": {},
        "jn": {},
        "jo": {},
        "jp": {},
        "jq": {},
        "jr": {},
        "js": {},
        "jt": {},
        "ju": {},
        "jv": {},
        "jw": {},
        "jx": {},
        "jy": {},
        "jz": {}
    },
    "k": {
        "ka": {},
        "kb": {},
        "kc": {},
        "kd": {},
        "ke": {},
        "kf": {},
        "kg": {},
        "kh": {},
        "ki": {},
        "kj": {},
        "kk": {},
        "kl": {},
        "km": {},
        "kn": {},
        "ko": {},
        "kp": {},
        "kq": {},
        "kr": {},
        "ks": {},
        "kt": {},
        "ku": {},
        "kv": {},
        "kw": {},
        "kx": {},
        "ky": {},
        "kz": {}
    },
    "l": {
        "la": {},
        "lb": {},
        "lc": {},
        "ld": {},
        "le": {},
        "lf": {},
        "lg": {},
        "lh": {},
        "li": {},
        "lj": {},
        "lk": {},
        "ll": {},
        "lm": {},
        "ln": {},
        "lo": {},
        "lp": {},
        "lq": {},
        "lr": {},
        "ls": {},
        "lt": {},
        "lu": {},
        "lv": {},
        "lw": {},
        "lx": {},
        "ly": {},
        "lz": {}
    },
    "m": {
        "ma": {},
        "mb": {},
        "mc": {},
        "md": {},
        "me": {},
        "mf": {},
        "mg": {},
        "mh": {},
        "mi": {},
        "mj": {},
        "mk": {},
        "ml": {},
        "mm": {},
        "mn": {},
        "mo": {},
        "mp": {},
        "mq": {},
        "mr": {},
        "ms": {},
        "mt": {},
        "mu": {},
        "mv": {},
        "mw": {},
        "mx": {},
        "my": {},
        "mz": {}
    },
    "n": {
        "na": {},
        "nb": {},
        "nc": {},
        "nd": {},
        "ne": {},
        "nf": {},
        "ng": {},
        "nh": {},
        "ni": {},
        "nj": {},
        "nk": {},
        "nl": {},
        "nm": {},
        "nn": {},
        "no": {},
        "np": {},
        "nq": {},
        "nr": {},
        "ns": {},
        "nt": {},
        "nu": {},
        "nv": {},
        "nw": {},
        "nx": {},
        "ny": {},
        "nz": {}
    },
    "o": {
        "oa": {},
        "ob": {},
        "oc": {},
        "od": {},
        "oe": {},
        "of": {},
        "og": {},
        "oh": {},
        "oi": {},
        "oj": {},
        "ok": {},
        "ol": {},
        "om": {},
        "on": {},
        "oo": {},
        "op": {},
        "oq": {},
        "or": {},
        "os": {},
        "ot": {},
        "ou": {},
        "ov": {},
        "ow": {},
        "ox": {},
        "oy": {},
        "oz": {}
    },
    "p": {
        "pa": {},
        "pb": {},
        "pc": {},
        "pd": {},
        "pe": {},
        "pf": {},
        "pg": {},
        "ph": {},
        "pi": {},
        "pj": {},
        "pk": {},
        "pl": {},
        "pm": {},
        "pn": {},
        "po": {},
        "pp": {},
        "pq": {},
        "pr": {},
        "ps": {},
        "pt": {},
        "pu": {},
        "pv": {},
        "pw": {},
        "px": {},
        "py": {},
        "pz": {}
    },
    "q": {
        "qa": {},
        "qb": {},
        "qc": {},
        "qd": {},
        "qe": {},
        "qf": {},
        "qg": {},
        "qh": {},
        "qi": {},
        "qj": {},
        "qk": {},
        "ql": {},
        "qm": {},
        "qn": {},
        "qo": {},
        "qp": {},
        "qq": {},
        "qr": {},
        "qs": {},
        "qt": {},
        "qu": {},
        "qv": {},
        "qw": {},
        "qx": {},
        "qy": {},
        "qz": {}
    },
    "r": {
        "ra": {},
        "rb": {},
        "rc": {},
        "rd": {},
        "re": {},
        "rf": {},
        "rg": {},
        "rh": {},
        "ri": {},
        "rj": {},
        "rk": {},
        "rl": {},
        "rm": {},
        "rn": {},
        "ro": {},
        "rp": {},
        "rq": {},
        "rr": {},
        "rs": {},
        "rt": {},
        "ru": {},
        "rv": {},
        "rw": {},
        "rx": {},
        "ry": {},
        "rz": {}
    },
    "s": {
        "sa": {},
        "sb": {},
        "sc": {},
        "sd": {},
        "se": {},
        "sf": {},
        "sg": {},
        "sh": {},
        "si": {},
        "sj": {},
        "sk": {},
        "sl": {},
        "sm": {},
        "sn": {},
        "so": {},
        "sp": {},
        "sq": {},
        "sr": {},
        "ss": {},
        "st": {},
        "su": {},
        "sv": {},
        "sw": {},
        "sx": {},
        "sy": {},
        "sz": {}
    },
    "t": {
        "ta": {},
        "tb": {},
        "tc": {},
        "td": {},
        "te": {},
        "tf": {},
        "tg": {},
        "th": {},
        "ti": {},
        "tj": {},
        "tk": {},
        "tl": {},
        "tm": {},
        "tn": {},
        "to": {},
        "tp": {},
        "tq": {},
        "tr": {},
        "ts": {},
        "tt": {},
        "tu": {},
        "tv": {},
        "tw": {},
        "tx": {},
        "ty": {},
        "tz": {}
    },
    "u": {
        "ua": {},
        "ub": {},
        "uc": {},
        "ud": {},
        "ue": {},
        "uf": {},
        "ug": {},
        "uh": {},
        "ui": {},
        "uj": {},
        "uk": {},
        "ul": {},
        "um": {},
        "un": {},
        "uo": {},
        "up": {},
        "uq": {},
        "ur": {},
        "us": {},
        "ut": {},
        "uu": {},
        "uv": {},
        "uw": {},
        "ux": {},
        "uy": {},
        "uz": {}
    },
    "v": {
        "va": {},
        "vb": {},
        "vc": {},
        "vd": {},
        "ve": {},
        "vf": {},
        "vg": {},
        "vh": {},
        "vi": {},
        "vj": {},
        "vk": {},
        "vl": {},
        "vm": {},
        "vn": {},
        "vo": {},
        "vp": {},
        "vq": {},
        "vr": {},
        "vs": {},
        "vt": {},
        "vu": {},
        "vv": {},
        "vw": {},
        "vx": {},
        "vy": {},
        "vz": {}
    },
    "w": {
        "wa": {},
        "wb": {},
        "wc": {},
        "wd": {},
        "we": {},
        "wf": {},
        "wg": {},
        "wh": {},
        "wi": {},
        "wj": {},
        "wk": {},
        "wl": {},
        "wm": {},
        "wn": {},
        "wo": {},
        "wp": {},
        "wq": {},
        "wr": {},
        "ws": {},
        "wt": {},
        "wu": {},
        "wv": {},
        "ww": {},
        "wx": {},
        "wy": {},
        "wz": {}
    },
    "x": {
        "xa": {},
        "xb": {},
        "xc": {},
        "xd": {},
        "xe": {},
        "xf": {},
        "xg": {},
        "xh": {},
        "xi": {},
        "xj": {},
        "xk": {},
        "xl": {},
        "xm": {},
        "xn": {},
        "xo": {},
        "xp": {},
        "xq": {},
        "xr": {},
        "xs": {},
        "xt": {},
        "xu": {},
        "xv": {},
        "xw": {},
        "xx": {},
        "xy": {},
        "xz": {}
    },
    "y": {
        "ya": {},
        "yb": {},
        "yc": {},
        "yd": {},
        "ye": {},
        "yf": {},
        "yg": {},
        "yh": {},
        "yi": {},
        "yj": {},
        "yk": {},
        "yl": {},
        "ym": {},
        "yn": {},
        "yo": {},
        "yp": {},
        "yq": {},
        "yr": {},
        "ys": {},
        "yt": {},
        "yu": {},
        "yv": {},
        "yw": {},
        "yx": {},
        "yy": {},
        "yz": {}
    },
    "z": {
        "za": {},
        "zb": {},
        "zc": {},
        "zd": {},
        "ze": {},
        "zf": {},
        "zg": {},
        "zh": {},
        "zi": {},
        "zj": {},
        "zk": {},
        "zl": {},
        "zm": {},
        "zn": {},
        "zo": {},
        "zp": {},
        "zq": {},
        "zr": {},
        "zs": {},
        "zt": {},
        "zu": {},
        "zv": {},
        "zw": {},
        "zx": {},
        "zy": {},
        "zz": {}
    }
}







def save_to_json_local(case_info,prior,priortext):
  file_path_json = 'data.json'
  with open(file_path_json, 'r') as file:
    dataImport = json.load(file)

  words = case_info["compressed_text"].split()
  for word in words:
    if len(word) >= 2:
      if word[0].lower().isalnum() and word[1].lower().isalnum(): 
        first_two_letters = word[0].lower() + word[1].lower()
        if word[0].lower() in dataImport:
          if (first_two_letters in dataImport[word[0].lower()]) and (word.lower() in dataImport[word[0].lower()][first_two_letters]):
            # data[word[0].lower()][first_two_letters][word]["file_path"]=list(set(data[word[0].lower()][first_two_letters][word]["file_path"].append(case_info["file_path"])))
            # Append case_info["file_path"] to the list
            dataImport[word[0].lower()][first_two_letters][word.lower()]["file_path"].append(case_info["file_path"])

# Convert the list to a set to remove duplicates and then back to a list
            dataImport[word[0].lower()][first_two_letters][word.lower()]["file_path"] = list(set(dataImport[word[0].lower()][first_two_letters][word.lower()]["file_path"]))


            dataImport[word[0].lower()][first_two_letters][word.lower()]["sections"]=list(set(dataImport[word[0].lower()][first_two_letters][word.lower()]["sections"]).union(set(case_info["sections"])))

            dataImport[word[0].lower()][first_two_letters][word.lower()]["acts"]=list(set(dataImport[word[0].lower()][first_two_letters][word.lower()]["acts"]).union(set(case_info["acts"])))
            
          elif first_two_letters in dataImport[word[0].lower()]:
              dataImport[word[0].lower()][first_two_letters][word.lower()] = {"priority":prior,"file_path":[case_info["file_path"]],"sections":case_info["sections"],"acts":case_info["acts"]}
          else:
            pass
        else:
            print(f"First two letters '{first_two_letters}' not found in the JSON structure.")
  with open(file_path_json, 'w') as file:
    json.dump(dataImport, file, indent=4)         
  # json_dataImport = json.dumps(data, indent=4)
  # file_path_json = 'data.json'
  # with open(file_path_json, 'w') as file:
  #   json.dump(data, file)

  













def save_to_json_local2(case_info,prior,priortext):
  file_path_json = 'data.json'
  with open(file_path_json, 'r') as file:
    dataImport = json.load(file)

  words = priortext.split(',')
  for word in words:
    if len(word) >= 2:
      if word[0].lower().isalnum() and word[1].lower().isalnum(): 
        first_two_letters = word[0].lower() + word[1].lower()
        if word[0].lower() in dataImport:
          if (first_two_letters in dataImport[word[0].lower()]) and word.lower() in dataImport[word[0].lower()][first_two_letters]:
            # data[word[0].lower()][first_two_letters][word]["file_path"]=list(set(data[word[0].lower()][first_two_letters][word]["file_path"].append(case_info["file_path"])))
            # Append case_info["file_path"] to the list
            dataImport[word[0].lower()][first_two_letters][word.lower()]["file_path"].append(case_info["file_path"])

# Convert the list to a set to remove duplicates and then back to a list
            dataImport[word[0].lower()][first_two_letters][word.lower()]["file_path"] = list(set(dataImport[word[0].lower()][first_two_letters][word.lower()]["file_path"]))


            dataImport[word[0].lower()][first_two_letters][word.lower()]["sections"]=list(set(dataImport[word[0].lower()][first_two_letters][word.lower()]["sections"]).union(set(case_info["sections"])))

            dataImport[word[0].lower()][first_two_letters][word.lower()]["acts"]=list(set(dataImport[word[0].lower()][first_two_letters][word.lower()]["acts"]).union(set(case_info["acts"])))
            
          elif first_two_letters in dataImport[word[0].lower()]:
              dataImport[word[0].lower()][first_two_letters][word.lower()] = {"priority":prior,"file_path":[case_info["file_path"]],"sections":case_info["sections"],"acts":case_info["acts"]}
          else:
            pass
        else:
            print(f"First two letters '{first_two_letters}' not found in the JSON structure.")
  with open(file_path_json, 'w') as file:
    json.dump(dataImport, file, indent=4)         
  # json_dataImport = json.dumps(data, indent=4)
  # file_path_json = 'data.json'
  # with open(file_path_json, 'w') as file:
  #   json.dump(data, file)











# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------

# def extract_text_from_pdf(file):
#     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        
#         file.save(temp_file)
#         with open(temp_file.name, 'rb') as pdf_file:
#             pdf_reader = PdfReader(pdf_file)
#             text = ''
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#     return text



@app.route('/', methods=['GET', 'POST'])
def upload_pdf():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        # first_name = request.form['first_name']
        # last_name = request.form['last_name']
        # case_id = 
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        if file:
          file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
          file.save(file_path)
          

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------

          case_info = {
             "file_path": file_path,
             "court_name": None,
    
             "case_id": 0,

             "case_year": 0,
    
             "judges_names": [],
    
             "parties_involved": {
                 "appellants": [],
                 "respondents": []
             },
    
              "sections": [],
    
              "acts": [],
    
             "summary":None,
    
              "compressed_text": None,
    
             "similar_cases": [],
             "wordCloudpng": "static/logo.png"

              }

          case_info['court_name'] = request.form['court_name']
          case_info['case_id'] = request.form['case_id']
          case_info['case_year'] = request.form['case_year']
          case_info['judges_names'] = request.form['judges_names'].split()
          case_info['parties_involved']['appellants'] = request.form['appellants'].split()
          case_info['parties_involved']['respondents'] = request.form['respondents'].split()

          case_info_str = f"Name of Court = '{case_info['court_name']}'<br>" \
              f"Case ID = {case_info['case_id']}<br>" \
              f"Case Year = {case_info['case_year']}<br>" \
              f"Jydges Names= {request.form['judges_names']}<br>" \
              f"Appellants Name = {request.form['appellants']}<br>" \
              f"Respondents Name = {request.form['respondents']}<br>"
           


            


          OCR_code(case_info)
          summary = summarize_document(para)
          # case_info["summary"] = case_info_str + " \n " + summary
          case_info["summary"] = summary
          processed_text = process_document(para)
          case_info["compressed_text"] = processed_text
          wordCloud_longText = processed_text
          stopwords = set(STOPWORDS)
          stopwords.update(['party','arbitration','Court', 'court', 'Section', 'section', 'Act', 'act', 'Case', 'Law', 'Issue', 'Evidence', 'Fact', 'Decision', 'Appeal', 'Judge', 'Trial', 'Argument', 'Ruling', 'Verdict', 'Hearing', 'Jurisdiction', 'Counsel', 'Plaintiff', 'Defendant', 'Petitioner', 'Respondent', 'Motion', 'Order', 'Grounds', 'Objection', 'Precedent', 'Statute', 'Statutory', 'Provision', 'Interpretation', 'Standard', 'Principle', 'Application', 'Conclusion', 'Reasoning', 'Authority', 'Jurisprudence', 'Merit', 'Injunction', 'Remedy', 'Damages', 'Allegation', 'Testimony', 'Witness', 'Cross-examination', 'Admissible', 'Hearsay', 'Expert', 'Subsequent', 'Procedure', 'Procedural', 'Juror', 'Jury', 'Appellate', 'Circuit', 'Magistrate', 'Adjudication', 'Adjudicate', 'Appealable', 'Brief', 'Certiorari', 'Collateral', 'Concurrent', 'Concur', 'Dissent', 'Discretion', 'Discretionary', 'Ex parte', 'Garnish', 'Impeachment', 'Indict', 'Interrogatory', 'Joinder', 'Litigant', 'Litigation', 'Pleadings', 'Prima facie', 'Rebut', 'Sentence', 'Subpoena', 'Summons', 'Tort', 'Venue', 'Waive', 'Warrant', 'Acquit', 'Affidavit', 'Affirm', 'Amicus', 'Bail', 'Contempt', 'Deposition', 'Jurisdictional', 'Remand', 'Stipulation', 'Testify', 'Verdict','High','FIR','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','aa', 'Aa', 'aA', 'AA', 'ab', 'Ab', 'aB', 'AB', 'ac', 'Ac', 'aC', 'AC', 'ad', 'Ad', 'aD', 'AD', 'ae', 'Ae', 'aE', 'AE', 'af', 'Af', 'aF', 'AF', 'ag', 'Ag', 'aG', 'AG', 'ah', 'Ah', 'aH', 'AH', 'ai', 'Ai', 'aI', 'AI', 'aj', 'Aj', 'aJ', 'AJ', 'ak', 'Ak', 'aK', 'AK', 'al', 'Al', 'aL', 'AL', 'am', 'Am', 'aM', 'AM', 'an', 'An', 'aN', 'AN', 'ao', 'Ao', 'aO', 'AO', 'ap', 'Ap', 'aP', 'AP', 'aq', 'Aq', 'aQ', 'AQ', 'ar', 'Ar', 'aR', 'AR', 'as', 'As', 'aS', 'AS', 'at', 'At', 'aT', 'AT', 'au', 'Au', 'aU', 'AU', 'av', 'Av', 'aV', 'AV', 'aw', 'Aw', 'aW', 'AW', 'ax', 'Ax', 'aX', 'AX', 'ay', 'Ay', 'aY', 'AY', 'az', 'Az', 'aZ', 'AZ', 'ba', 'Ba', 'bA', 'BA', 'bb', 'Bb', 'bB', 'BB', 'bc', 'Bc', 'bC', 'BC', 'bd', 'Bd', 'bD', 'BD', 'be', 'Be', 'bE', 'BE', 'bf', 'Bf', 'bF', 'BF', 'bg', 'Bg', 'bG', 'BG', 'bh', 'Bh', 'bH', 'BH', 'bi', 'Bi', 'bI', 'BI', 'bj', 'Bj', 'bJ', 'BJ', 'bk', 'Bk', 'bK', 'BK', 'bl', 'Bl', 'bL', 'BL', 'bm', 'Bm', 'bM', 'BM', 'bn', 'Bn', 'bN', 'BN', 'bo', 'Bo', 'bO', 'BO', 'bp', 'Bp', 'bP', 'BP', 'bq', 'Bq', 'bQ', 'BQ', 'br', 'Br', 'bR', 'BR', 'bs', 'Bs', 'bS', 'BS', 'bt', 'Bt', 'bT', 'BT', 'bu', 'Bu', 'bU', 'BU', 'bv', 'Bv', 'bV', 'BV', 'bw', 'Bw', 'bW', 'BW', 'bx', 'Bx', 'bX', 'BX', 'by', 'By', 'bY', 'BY', 'bz', 'Bz', 'bZ', 'BZ', 'ca', 'Ca', 'cA', 'CA', 'cb', 'Cb', 'cB', 'CB', 'cc', 'Cc', 'cC', 'CC', 'cd', 'Cd', 'cD', 'CD', 'ce', 'Ce', 'cE', 'CE', 'cf', 'Cf', 'cF', 'CF', 'cg', 'Cg', 'cG', 'CG', 'ch', 'Ch', 'cH', 'CH', 'ci', 'Ci', 'cI', 'CI', 'cj', 'Cj', 'cJ', 'CJ', 'ck', 'Ck', 'cK', 'CK', 'cl', 'Cl', 'cL', 'CL', 'cm', 'Cm', 'cM', 'CM', 'cn', 'Cn', 'cN', 'CN', 'co', 'Co', 'cO', 'CO', 'cp', 'Cp', 'cP', 'CP', 'cq', 'Cq', 'cQ', 'CQ', 'cr', 'Cr', 'cR', 'CR', 'cs', 'Cs', 'cS', 'CS', 'ct', 'Ct', 'cT', 'CT', 'cu', 'Cu', 'cU', 'CU', 'cv', 'Cv', 'cV', 'CV', 'cw', 'Cw', 'cW', 'CW', 'cx', 'Cx', 'cX', 'CX', 'cy', 'Cy', 'cY', 'CY', 'cz', 'Cz', 'cZ', 'CZ', 'da', 'Da', 'dA', 'DA', 'db', 'Db', 'dB', 'DB', 'dc', 'Dc', 'dC', 'DC', 'dd', 'Dd', 'dD', 'DD', 'de', 'De', 'dE', 'DE', 'df', 'Df', 'dF', 'DF', 'dg', 'Dg', 'dG', 'DG', 'dh', 'Dh', 'dH', 'DH', 'di', 'Di', 'dI', 'DI', 'dj', 'Dj', 'dJ', 'DJ', 'dk', 'Dk', 'dK', 'DK', 'dl', 'Dl', 'dL', 'DL', 'dm', 'Dm', 'dM', 'DM', 'dn', 'Dn', 'dN', 'DN', 'do', 'Do', 'dO', 'DO', 'dp', 'Dp', 'dP', 'DP', 'dq', 'Dq', 'dQ', 'DQ', 'dr', 'Dr', 'dR', 'DR', 'ds', 'Ds', 'dS', 'DS', 'dt', 'Dt', 'dT', 'DT', 'du', 'Du', 'dU', 'DU', 'dv', 'Dv', 'dV', 'DV', 'dw', 'Dw', 'dW', 'DW', 'dx', 'Dx', 'dX', 'DX', 'dy', 'Dy', 'dY', 'DY', 'dz', 'Dz', 'dZ', 'DZ', 'ea', 'Ea', 'eA', 'EA', 'eb', 'Eb', 'eB', 'EB', 'ec', 'Ec', 'eC', 'EC', 'ed', 'Ed', 'eD', 'ED', 'ee', 'Ee', 'eE', 'EE', 'ef', 'Ef', 'eF', 'EF', 'eg', 'Eg', 'eG', 'EG', 'eh', 'Eh', 'eH', 'EH', 'ei', 'Ei', 'eI', 'EI', 'ej', 'Ej', 'eJ', 'EJ', 'ek', 'Ek', 'eK', 'EK', 'el', 'El', 'eL', 'EL', 'em', 'Em', 'eM', 'EM', 'en', 'En', 'eN', 'EN', 'eo', 'Eo', 'eO', 'EO', 'ep', 'Ep', 'eP', 'EP', 'eq', 'Eq', 'eQ', 'EQ', 'er', 'Er', 'eR', 'ER', 'es', 'Es', 'eS', 'ES', 'et', 'Et', 'eT', 'ET', 'eu', 'Eu', 'eU', 'EU', 'ev', 'Ev', 'eV', 'EV', 'ew', 'Ew', 'eW', 'EW', 'ex', 'Ex', 'eX', 'EX', 'ey', 'Ey', 'eY', 'EY', 'ez', 'Ez', 'eZ', 'EZ', 'fa', 'Fa', 'fA', 'FA', 'fb', 'Fb', 'fB', 'FB', 'fc', 'Fc', 'fC', 'FC', 'fd', 'Fd', 'fD', 'FD', 'fe', 'Fe', 'fE', 'FE', 'ff', 'Ff', 'fF', 'FF', 'fg', 'Fg', 'fG', 'FG', 'fh', 'Fh', 'fH', 'FH', 'fi', 'Fi', 'fI', 'FI', 'fj', 'Fj', 'fJ', 'FJ', 'fk', 'Fk', 'fK', 'FK', 'fl', 'Fl', 'fL', 'FL', 'fm', 'Fm', 'fM', 'FM', 'fn', 'Fn', 'fN', 'FN', 'fo', 'Fo', 'fO', 'FO', 'fp', 'Fp', 'fP', 'FP', 'fq', 'Fq', 'fQ', 'FQ', 'fr', 'Fr', 'fR', 'FR', 'fs', 'Fs', 'fS', 'FS', 'ft', 'Ft', 'fT', 'FT', 'fu', 'Fu', 'fU', 'FU', 'fv', 'Fv', 'fV', 'FV', 'fw', 'Fw', 'fW', 'FW', 'fx', 'Fx', 'fX', 'FX', 'fy', 'Fy', 'fY', 'FY', 'fz', 'Fz', 'fZ', 'FZ', 'ga', 'Ga', 'gA', 'GA', 'gb', 'Gb', 'gB', 'GB', 'gc', 'Gc', 'gC', 'GC', 'gd', 'Gd', 'gD', 'GD', 'ge', 'Ge', 'gE', 'GE', 'gf', 'Gf', 'gF', 'GF', 'gg', 'Gg', 'gG', 'GG', 'gh', 'Gh', 'gH', 'GH', 'gi', 'Gi', 'gI', 'GI', 'gj', 'Gj', 'gJ', 'GJ', 'gk', 'Gk', 'gK', 'GK', 'gl', 'Gl', 'gL', 'GL', 'gm', 'Gm', 'gM', 'GM', 'gn', 'Gn', 'gN', 'GN', 'go', 'Go', 'gO', 'GO', 'gp', 'Gp', 'gP', 'GP', 'gq', 'Gq', 'gQ', 'GQ', 'gr', 'Gr', 'gR', 'GR', 'gs', 'Gs', 'gS', 'GS', 'gt', 'Gt', 'gT', 'GT', 'gu', 'Gu', 'gU', 'GU', 'gv', 'Gv', 'gV', 'GV', 'gw', 'Gw', 'gW', 'GW', 'gx', 'Gx', 'gX', 'GX', 'gy', 'Gy', 'gY', 'GY', 'gz', 'Gz', 'gZ', 'GZ', 'ha', 'Ha', 'hA', 'HA', 'hb', 'Hb', 'hB', 'HB', 'hc', 'Hc', 'hC', 'HC', 'hd', 'Hd', 'hD', 'HD', 'he', 'He', 'hE', 'HE', 'hf', 'Hf', 'hF', 'HF', 'hg', 'Hg', 'hG', 'HG', 'hh', 'Hh', 'hH', 'HH', 'hi', 'Hi', 'hI', 'HI', 'hj', 'Hj', 'hJ', 'HJ', 'hk', 'Hk', 'hK', 'HK', 'hl', 'Hl', 'hL', 'HL', 'hm', 'Hm', 'hM', 'HM', 'hn', 'Hn', 'hN', 'HN', 'ho', 'Ho', 'hO', 'HO', 'hp', 'Hp', 'hP', 'HP', 'hq', 'Hq', 'hQ', 'HQ', 'hr', 'Hr', 'hR', 'HR', 'hs', 'Hs', 'hS', 'HS', 'ht', 'Ht', 'hT', 'HT', 'hu', 'Hu', 'hU', 'HU', 'hv', 'Hv', 'hV', 'HV', 'hw', 'Hw', 'hW', 'HW', 'hx', 'Hx', 'hX', 'HX', 'hy', 'Hy', 'hY', 'HY', 'hz', 'Hz', 'hZ', 'HZ', 'ia', 'Ia', 'iA', 'IA', 'ib', 'Ib', 'iB', 'IB', 'ic', 'Ic', 'iC', 'IC', 'id', 'Id', 'iD', 'ID', 'ie', 'Ie', 'iE', 'IE', 'if', 'If', 'iF', 'IF', 'ig', 'Ig', 'iG', 'IG', 'ih', 'Ih', 'iH', 'IH', 'ii', 'Ii', 'iI', 'II', 'ij', 'Ij', 'iJ', 'IJ', 'ik', 'Ik', 'iK', 'IK', 'il', 'Il', 'iL', 'IL', 'im', 'Im', 'iM', 'IM', 'in', 'In', 'iN', 'IN', 'io', 'Io', 'iO', 'IO', 'ip', 'Ip', 'iP', 'IP', 'iq', 'Iq', 'iQ', 'IQ', 'ir', 'Ir', 'iR', 'IR', 'is', 'Is', 'iS', 'IS', 'it', 'It', 'iT', 'IT', 'iu', 'Iu', 'iU', 'IU', 'iv', 'Iv', 'iV', 'IV', 'iw', 'Iw', 'iW', 'IW', 'ix', 'Ix', 'iX', 'IX', 'iy', 'Iy', 'iY', 'IY', 'iz', 'Iz', 'iZ', 'IZ', 'ja', 'Ja', 'jA', 'JA', 'jb', 'Jb', 'jB', 'JB', 'jc', 'Jc', 'jC', 'JC', 'jd', 'Jd', 'jD', 'JD', 'je', 'Je', 'jE', 'JE', 'jf', 'Jf', 'jF', 'JF', 'jg', 'Jg', 'jG', 'JG', 'jh', 'Jh', 'jH', 'JH', 'ji', 'Ji', 'jI', 'JI', 'jj', 'Jj', 'jJ', 'JJ', 'jk', 'Jk', 'jK', 'JK', 'jl', 'Jl', 'jL', 'JL', 'jm', 'Jm', 'jM', 'JM', 'jn', 'Jn', 'jN', 'JN', 'jo', 'Jo', 'jO', 'JO', 'jp', 'Jp', 'jP', 'JP', 'jq', 'Jq', 'jQ', 'JQ', 'jr', 'Jr', 'jR', 'JR', 'js', 'Js', 'jS', 'JS', 'jt', 'Jt', 'jT', 'JT', 'ju', 'Ju', 'jU', 'JU', 'jv', 'Jv', 'jV', 'JV', 'jw', 'Jw', 'jW', 'JW', 'jx', 'Jx', 'jX', 'JX', 'jy', 'Jy', 'jY', 'JY', 'jz', 'Jz', 'jZ', 'JZ', 'ka', 'Ka', 'kA', 'KA', 'kb', 'Kb', 'kB', 'KB', 'kc', 'Kc', 'kC', 'KC', 'kd', 'Kd', 'kD', 'KD', 'ke', 'Ke', 'kE', 'KE', 'kf', 'Kf', 'kF', 'KF', 'kg', 'Kg', 'kG', 'KG', 'kh', 'Kh', 'kH', 'KH', 'ki', 'Ki', 'kI', 'KI', 'kj', 'Kj', 'kJ', 'KJ', 'kk', 'Kk', 'kK', 'KK', 'kl', 'Kl', 'kL', 'KL', 'km', 'Km', 'kM', 'KM', 'kn', 'Kn', 'kN', 'KN', 'ko', 'Ko', 'kO', 'KO', 'kp', 'Kp', 'kP', 'KP', 'kq', 'Kq', 'kQ', 'KQ', 'kr', 'Kr', 'kR', 'KR', 'ks', 'Ks', 'kS', 'KS', 'kt', 'Kt', 'kT', 'KT', 'ku', 'Ku', 'kU', 'KU', 'kv', 'Kv', 'kV', 'KV', 'kw', 'Kw', 'kW', 'KW', 'kx', 'Kx', 'kX', 'KX', 'ky', 'Ky', 'kY', 'KY', 'kz', 'Kz', 'kZ', 'KZ', 'la', 'La', 'lA', 'LA', 'lb', 'Lb', 'lB', 'LB', 'lc', 'Lc', 'lC', 'LC', 'ld', 'Ld', 'lD', 'LD', 'le', 'Le', 'lE', 'LE', 'lf', 'Lf', 'lF', 'LF', 'lg', 'Lg', 'lG', 'LG', 'lh', 'Lh', 'lH', 'LH', 'li', 'Li', 'lI', 'LI', 'lj', 'Lj', 'lJ', 'LJ', 'lk', 'Lk', 'lK', 'LK', 'll', 'Ll', 'lL', 'LL', 'lm', 'Lm', 'lM', 'LM', 'ln', 'Ln', 'lN', 'LN', 'lo', 'Lo', 'lO', 'LO', 'lp', 'Lp', 'lP', 'LP', 'lq', 'Lq', 'lQ', 'LQ', 'lr', 'Lr', 'lR', 'LR', 'ls', 'Ls', 'lS', 'LS', 'lt', 'Lt', 'lT', 'LT', 'lu', 'Lu', 'lU', 'LU', 'lv', 'Lv', 'lV', 'LV', 'lw', 'Lw', 'lW', 'LW', 'lx', 'Lx', 'lX', 'LX', 'ly', 'Ly', 'lY', 'LY', 'lz', 'Lz', 'lZ', 'LZ', 'ma', 'Ma', 'mA', 'MA', 'mb', 'Mb', 'mB', 'MB', 'mc', 'Mc', 'mC', 'MC', 'md', 'Md', 'mD', 'MD', 'me', 'Me', 'mE', 'ME', 'mf', 'Mf', 'mF', 'MF', 'mg', 'Mg', 'mG', 'MG', 'mh', 'Mh', 'mH', 'MH', 'mi', 'Mi', 'mI', 'MI', 'mj', 'Mj', 'mJ', 'MJ', 'mk', 'Mk', 'mK', 'MK', 'ml', 'Ml', 'mL', 'ML', 'mm', 'Mm', 'mM', 'MM', 'mn', 'Mn', 'mN', 'MN', 'mo', 'Mo', 'mO', 'MO', 'mp', 'Mp', 'mP', 'MP', 'mq', 'Mq', 'mQ', 'MQ', 'mr', 'Mr', 'mR', 'MR', 'ms', 'Ms', 'mS', 'MS', 'mt', 'Mt', 'mT', 'MT', 'mu', 'Mu', 'mU', 'MU', 'mv', 'Mv', 'mV', 'MV', 'mw', 'Mw', 'mW', 'MW', 'mx', 'Mx', 'mX', 'MX', 'my', 'My', 'mY', 'MY', 'mz', 'Mz', 'mZ', 'MZ', 'na', 'Na', 'nA', 'NA', 'nb', 'Nb', 'nB', 'NB', 'nc', 'Nc', 'nC', 'NC', 'nd', 'Nd', 'nD', 'ND', 'ne', 'Ne', 'nE', 'NE', 'nf', 'Nf', 'nF', 'NF', 'ng', 'Ng', 'nG', 'NG', 'nh', 'Nh', 'nH', 'NH', 'ni', 'Ni', 'nI', 'NI', 'nj', 'Nj', 'nJ', 'NJ', 'nk', 'Nk', 'nK', 'NK', 'nl', 'Nl', 'nL', 'NL', 'nm', 'Nm', 'nM', 'NM', 'nn', 'Nn', 'nN', 'NN', 'no', 'No', 'nO', 'NO', 'np', 'Np', 'nP', 'NP', 'nq', 'Nq', 'nQ', 'NQ', 'nr', 'Nr', 'nR', 'NR', 'ns', 'Ns', 'nS', 'NS', 'nt', 'Nt', 'nT', 'NT', 'nu', 'Nu', 'nU', 'NU', 'nv', 'Nv', 'nV', 'NV', 'nw', 'Nw', 'nW', 'NW', 'nx', 'Nx', 'nX', 'NX', 'ny', 'Ny', 'nY', 'NY', 'nz', 'Nz', 'nZ', 'NZ', 'oa', 'Oa', 'oA', 'OA', 'ob', 'Ob', 'oB', 'OB', 'oc', 'Oc', 'oC', 'OC', 'od', 'Od', 'oD', 'OD', 'oe', 'Oe', 'oE', 'OE', 'of', 'Of', 'oF', 'OF', 'og', 'Og', 'oG', 'OG', 'oh', 'Oh', 'oH', 'OH', 'oi', 'Oi', 'oI', 'OI', 'oj', 'Oj', 'oJ', 'OJ', 'ok', 'Ok', 'oK', 'OK', 'ol', 'Ol', 'oL', 'OL', 'om', 'Om', 'oM', 'OM', 'on', 'On', 'oN', 'ON', 'oo', 'Oo', 'oO', 'OO', 'op', 'Op', 'oP', 'OP', 'oq', 'Oq', 'oQ', 'OQ', 'or', 'Or', 'oR', 'OR', 'os', 'Os', 'oS', 'OS', 'ot', 'Ot', 'oT', 'OT', 'ou', 'Ou', 'oU', 'OU', 'ov', 'Ov', 'oV', 'OV', 'ow', 'Ow', 'oW', 'OW', 'ox', 'Ox', 'oX', 'OX', 'oy', 'Oy', 'oY', 'OY', 'oz', 'Oz', 'oZ', 'OZ', 'pa', 'Pa', 'pA', 'PA', 'pb', 'Pb', 'pB', 'PB', 'pc', 'Pc', 'pC', 'PC', 'pd', 'Pd', 'pD', 'PD', 'pe', 'Pe', 'pE', 'PE', 'pf', 'Pf', 'pF', 'PF', 'pg', 'Pg', 'pG', 'PG', 'ph', 'Ph', 'pH', 'PH', 'pi', 'Pi', 'pI', 'PI', 'pj', 'Pj', 'pJ', 'PJ', 'pk', 'Pk', 'pK', 'PK', 'pl', 'Pl', 'pL', 'PL', 'pm', 'Pm', 'pM', 'PM', 'pn', 'Pn', 'pN', 'PN', 'po', 'Po', 'pO', 'PO', 'pp', 'Pp', 'pP', 'PP', 'pq', 'Pq', 'pQ', 'PQ', 'pr', 'Pr', 'pR', 'PR', 'ps', 'Ps', 'pS', 'PS', 'pt', 'Pt', 'pT', 'PT', 'pu', 'Pu', 'pU', 'PU', 'pv', 'Pv', 'pV', 'PV', 'pw', 'Pw', 'pW', 'PW', 'px', 'Px', 'pX', 'PX', 'py', 'Py', 'pY', 'PY', 'pz', 'Pz', 'pZ', 'PZ', 'qa', 'Qa', 'qA', 'QA', 'qb', 'Qb', 'qB', 'QB', 'qc', 'Qc', 'qC', 'QC', 'qd', 'Qd', 'qD', 'QD', 'qe', 'Qe', 'qE', 'QE', 'qf', 'Qf', 'qF', 'QF', 'qg', 'Qg', 'qG', 'QG', 'qh', 'Qh', 'qH', 'QH', 'qi', 'Qi', 'qI', 'QI', 'qj', 'Qj', 'qJ', 'QJ', 'qk', 'Qk', 'qK', 'QK', 'ql', 'Ql', 'qL', 'QL', 'qm', 'Qm', 'qM', 'QM', 'qn', 'Qn', 'qN', 'QN', 'qo', 'Qo', 'qO', 'QO', 'qp', 'Qp', 'qP', 'QP', 'qq', 'Qq', 'qQ', 'QQ', 'qr', 'Qr', 'qR', 'QR', 'qs', 'Qs', 'qS', 'QS', 'qt', 'Qt', 'qT', 'QT', 'qu', 'Qu', 'qU', 'QU', 'qv', 'Qv', 'qV', 'QV', 'qw', 'Qw', 'qW', 'QW', 'qx', 'Qx', 'qX', 'QX', 'qy', 'Qy', 'qY', 'QY', 'qz', 'Qz', 'qZ', 'QZ', 'ra', 'Ra', 'rA', 'RA', 'rb', 'Rb', 'rB', 'RB', 'rc', 'Rc', 'rC', 'RC', 'rd', 'Rd', 'rD', 'RD', 're', 'Re', 'rE', 'RE', 'rf', 'Rf', 'rF', 'RF', 'rg', 'Rg', 'rG', 'RG', 'rh', 'Rh', 'rH', 'RH', 'ri', 'Ri', 'rI', 'RI', 'rj', 'Rj', 'rJ', 'RJ', 'rk', 'Rk', 'rK', 'RK', 'rl', 'Rl', 'rL', 'RL', 'rm', 'Rm', 'rM', 'RM', 'rn', 'Rn', 'rN', 'RN', 'ro', 'Ro', 'rO', 'RO', 'rp', 'Rp', 'rP', 'RP', 'rq', 'Rq', 'rQ', 'RQ', 'rr', 'Rr', 'rR', 'RR', 'rs', 'Rs', 'rS', 'RS', 'rt', 'Rt', 'rT', 'RT', 'ru', 'Ru', 'rU', 'RU', 'rv', 'Rv', 'rV', 'RV', 'rw', 'Rw', 'rW', 'RW', 'rx', 'Rx', 'rX', 'RX', 'ry', 'Ry', 'rY', 'RY', 'rz', 'Rz', 'rZ', 'RZ', 'sa', 'Sa', 'sA', 'SA', 'sb', 'Sb', 'sB', 'SB', 'sc', 'Sc', 'sC', 'SC', 'sd', 'Sd', 'sD', 'SD', 'se', 'Se', 'sE', 'SE', 'sf', 'Sf', 'sF', 'SF', 'sg', 'Sg', 'sG', 'SG', 'sh', 'Sh', 'sH', 'SH', 'si', 'Si', 'sI', 'SI', 'sj', 'Sj', 'sJ', 'SJ', 'sk', 'Sk', 'sK', 'SK', 'sl', 'Sl', 'sL', 'SL', 'sm', 'Sm', 'sM', 'SM', 'sn', 'Sn', 'sN', 'SN', 'so', 'So', 'sO', 'SO', 'sp', 'Sp', 'sP', 'SP', 'sq', 'Sq', 'sQ', 'SQ', 'sr', 'Sr', 'sR', 'SR', 'ss', 'Ss', 'sS', 'SS', 'st', 'St', 'sT', 'ST', 'su', 'Su', 'sU', 'SU', 'sv', 'Sv', 'sV', 'SV', 'sw', 'Sw', 'sW', 'SW', 'sx', 'Sx', 'sX', 'SX', 'sy', 'Sy', 'sY', 'SY', 'sz', 'Sz', 'sZ', 'SZ', 'ta', 'Ta', 'tA', 'TA', 'tb', 'Tb', 'tB', 'TB', 'tc', 'Tc', 'tC', 'TC', 'td', 'Td', 'tD', 'TD', 'te', 'Te', 'tE', 'TE', 'tf', 'Tf', 'tF', 'TF', 'tg', 'Tg', 'tG', 'TG', 'th', 'Th', 'tH', 'TH', 'ti', 'Ti', 'tI', 'TI', 'tj', 'Tj', 'tJ', 'TJ', 'tk', 'Tk', 'tK', 'TK', 'tl', 'Tl', 'tL', 'TL', 'tm', 'Tm', 'tM', 'TM', 'tn', 'Tn', 'tN', 'TN', 'to', 'To', 'tO', 'TO', 'tp', 'Tp', 'tP', 'TP', 'tq', 'Tq', 'tQ', 'TQ', 'tr', 'Tr', 'tR', 'TR', 'ts', 'Ts', 'tS', 'TS', 'tt', 'Tt', 'tT', 'TT', 'tu', 'Tu', 'tU', 'TU', 'tv', 'Tv', 'tV', 'TV', 'tw', 'Tw', 'tW', 'TW', 'tx', 'Tx', 'tX', 'TX', 'ty', 'Ty', 'tY', 'TY', 'tz', 'Tz', 'tZ', 'TZ', 'ua', 'Ua', 'uA', 'UA', 'ub', 'Ub', 'uB', 'UB', 'uc', 'Uc', 'uC', 'UC', 'ud', 'Ud', 'uD', 'UD', 'ue', 'Ue', 'uE', 'UE', 'uf', 'Uf', 'uF', 'UF', 'ug', 'Ug', 'uG', 'UG', 'uh', 'Uh', 'uH', 'UH', 'ui', 'Ui', 'uI', 'UI', 'uj', 'Uj', 'uJ', 'UJ', 'uk', 'Uk', 'uK', 'UK', 'ul', 'Ul', 'uL', 'UL', 'um', 'Um', 'uM', 'UM', 'un', 'Un', 'uN', 'UN', 'uo', 'Uo', 'uO', 'UO', 'up', 'Up', 'uP', 'UP', 'uq', 'Uq', 'uQ', 'UQ', 'ur', 'Ur', 'uR', 'UR', 'us', 'Us', 'uS', 'US', 'ut', 'Ut', 'uT', 'UT', 'uu', 'Uu', 'uU', 'UU', 'uv', 'Uv', 'uV', 'UV', 'uw', 'Uw', 'uW', 'UW', 'ux', 'Ux', 'uX', 'UX', 'uy', 'Uy', 'uY', 'UY', 'uz', 'Uz', 'uZ', 'UZ', 'va', 'Va', 'vA', 'VA', 'vb', 'Vb', 'vB', 'VB', 'vc', 'Vc', 'vC', 'VC', 'vd', 'Vd', 'vD', 'VD', 've', 'Ve', 'vE', 'VE', 'vf', 'Vf', 'vF', 'VF', 'vg', 'Vg', 'vG', 'VG', 'vh', 'Vh', 'vH', 'VH', 'vi', 'Vi', 'vI', 'VI', 'vj', 'Vj', 'vJ', 'VJ', 'vk', 'Vk', 'vK', 'VK', 'vl', 'Vl', 'vL', 'VL', 'vm', 'Vm', 'vM', 'VM', 'vn', 'Vn', 'vN', 'VN', 'vo', 'Vo', 'vO', 'VO', 'vp', 'Vp', 'vP', 'VP', 'vq', 'Vq', 'vQ', 'VQ', 'vr', 'Vr', 'vR', 'VR', 'vs', 'Vs', 'vS', 'VS', 'vt', 'Vt', 'vT', 'VT', 'vu', 'Vu', 'vU', 'VU', 'vv', 'Vv', 'vV', 'VV', 'vw', 'Vw', 'vW', 'VW', 'vx', 'Vx', 'vX', 'VX', 'vy', 'Vy', 'vY', 'VY', 'vz', 'Vz', 'vZ', 'VZ', 'wa', 'Wa', 'wA', 'WA', 'wb', 'Wb', 'wB', 'WB', 'wc', 'Wc', 'wC', 'WC', 'wd', 'Wd', 'wD', 'WD', 'we', 'We', 'wE', 'WE', 'wf', 'Wf', 'wF', 'WF', 'wg', 'Wg', 'wG', 'WG', 'wh', 'Wh', 'wH', 'WH', 'wi', 'Wi', 'wI', 'WI', 'wj', 'Wj', 'wJ', 'WJ', 'wk', 'Wk', 'wK', 'WK', 'wl', 'Wl', 'wL', 'WL', 'wm', 'Wm', 'wM', 'WM', 'wn', 'Wn', 'wN', 'WN', 'wo', 'Wo', 'wO', 'WO', 'wp', 'Wp', 'wP', 'WP', 'wq', 'Wq', 'wQ', 'WQ', 'wr', 'Wr', 'wR', 'WR', 'ws', 'Ws', 'wS', 'WS', 'wt', 'Wt', 'wT', 'WT', 'wu', 'Wu', 'wU', 'WU', 'wv', 'Wv', 'wV', 'WV', 'ww', 'Ww', 'wW', 'WW', 'wx', 'Wx', 'wX', 'WX', 'wy', 'Wy', 'wY', 'WY', 'wz', 'Wz', 'wZ', 'WZ', 'xa', 'Xa', 'xA', 'XA', 'xb', 'Xb', 'xB', 'XB', 'xc', 'Xc', 'xC', 'XC', 'xd', 'Xd', 'xD', 'XD', 'xe', 'Xe', 'xE', 'XE', 'xf', 'Xf', 'xF', 'XF', 'xg', 'Xg', 'xG', 'XG', 'xh', 'Xh', 'xH', 'XH', 'xi', 'Xi', 'xI', 'XI', 'xj', 'Xj', 'xJ', 'XJ', 'xk', 'Xk', 'xK', 'XK', 'xl', 'Xl', 'xL', 'XL', 'xm', 'Xm', 'xM', 'XM', 'xn', 'Xn', 'xN', 'XN', 'xo', 'Xo', 'xO', 'XO', 'xp', 'Xp', 'xP', 'XP', 'xq', 'Xq', 'xQ', 'XQ', 'xr', 'Xr', 'xR', 'XR', 'xs', 'Xs', 'xS', 'XS', 'xt', 'Xt', 'xT', 'XT', 'xu', 'Xu', 'xU', 'XU', 'xv', 'Xv', 'xV', 'XV', 'xw', 'Xw', 'xW', 'XW', 'xx', 'Xx', 'xX', 'XX', 'xy', 'Xy', 'xY', 'XY', 'xz', 'Xz', 'xZ', 'XZ', 'ya', 'Ya', 'yA', 'YA', 'yb', 'Yb', 'yB', 'YB', 'yc', 'Yc', 'yC', 'YC', 'yd', 'Yd', 'yD', 'YD', 'ye', 'Ye', 'yE', 'YE', 'yf', 'Yf', 'yF', 'YF', 'yg', 'Yg', 'yG', 'YG', 'yh', 'Yh', 'yH', 'YH', 'yi', 'Yi', 'yI', 'YI', 'yj', 'Yj', 'yJ', 'YJ', 'yk', 'Yk', 'yK', 'YK', 'yl', 'Yl', 'yL', 'YL', 'ym', 'Ym', 'yM', 'YM', 'yn', 'Yn', 'yN', 'YN', 'yo', 'Yo', 'yO', 'YO', 'yp', 'Yp', 'yP', 'YP', 'yq', 'Yq', 'yQ', 'YQ', 'yr', 'Yr', 'yR', 'YR', 'ys', 'Ys', 'yS', 'YS', 'yt', 'Yt', 'yT', 'YT', 'yu', 'Yu', 'yU', 'YU', 'yv', 'Yv', 'yV', 'YV', 'yw', 'Yw', 'yW', 'YW', 'yx', 'Yx', 'yX', 'YX', 'yy', 'Yy', 'yY', 'YY', 'yz', 'Yz', 'yZ', 'YZ', 'za', 'Za', 'zA', 'ZA', 'zb', 'Zb', 'zB', 'ZB', 'zc', 'Zc', 'zC', 'ZC', 'zd', 'Zd', 'zD', 'ZD', 'ze', 'Ze', 'zE', 'ZE', 'zf', 'Zf', 'zF', 'ZF', 'zg', 'Zg', 'zG', 'ZG', 'zh', 'Zh', 'zH', 'ZH', 'zi', 'Zi', 'zI', 'ZI', 'zj', 'Zj', 'zJ', 'ZJ', 'zk', 'Zk', 'zK', 'ZK', 'zl', 'Zl', 'zL', 'ZL', 'zm', 'Zm', 'zM', 'ZM', 'zn', 'Zn', 'zN', 'ZN', 'zo', 'Zo', 'zO', 'ZO', 'zp', 'Zp', 'zP', 'ZP', 'zq', 'Zq', 'zQ', 'ZQ', 'zr', 'Zr', 'zR', 'ZR', 'zs', 'Zs', 'zS', 'ZS', 'zt', 'Zt', 'zT', 'ZT', 'zu', 'Zu', 'zU', 'ZU', 'zv', 'Zv', 'zV', 'ZV', 'zw', 'Zw', 'zW', 'ZW', 'zx', 'Zx', 'zX', 'ZX', 'zy', 'Zy', 'zY', 'ZY', 'zz', 'Zz', 'zZ', 'ZZ'])
          wordcloud = WordCloud(width = 400, height = 400, background_color ='white',stopwords=stopwords).generate(wordCloud_longText)
          wordcloud.to_file(file_path[:-4]+'-wordCloud.png')
          case_info['wordCloudpng'] = file_path[:-4]+'-wordCloud.png'








          text = case_info["compressed_text"]


          section_pattern = re.compile(r'\bSection \d+\b', re.IGNORECASE)
          section_to_act_pattern = re.compile(r'\bSection\b\s+(\w+\s+){2,3}\bAct\b',re.DOTALL | re.IGNORECASE)

          # Find all matches in the text
          sections = section_pattern.findall(text)
          acts = section_to_act_pattern.findall(text)

          # Print the results and store 
          sections = [word.lower() for word in sections]
          sections = list(set(sections))
          acts = [word.lower() for word in acts]
          acts = list(set(acts))
          case_info["acts"] = acts


          case_info["sections"] = sections



          file_path_json = 'filesdata.json'
          with open(file_path_json, 'r') as file:
            dataFileImport = json.load(file)


          
          
          dataFileImport[case_info["file_path"]] = case_info 
          with open(file_path_json, 'w') as file:
            json.dump(dataFileImport, file, indent=4) 








          prior = 20
          priortext = f"{case_info['file_path']},{case_info['court_name']},{case_info['case_id']},{case_info['case_year']},{' ,'.join(case_info['judges_names'])},{' ,'.join(case_info['parties_involved']['appellants'])},{' ,'.join(case_info['parties_involved']['respondents'])} "
          save_to_json_local2(case_info,prior,priortext)
          prior = 2
          priortext = ''
          save_to_json_local(case_info,prior,priortext)
 # -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
            # pdf_text = extract_text_from_pdf(file)
            # pdf_text = "hd fj"
            # num_words = len(pdf_text.split())
          
          return render_template('summary.html', dataFile = case_info)
    
    return render_template('index.html')





def get_synonyms(word):
    synonyms = []

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())

    return list(set(synonyms))  # Convert to set to remove duplicates, then back to list

def extract_words(paragraph):
    words = word_tokenize(paragraph)
    return [word.lower() for word in words if word.isalnum()]  # Remove non-alphabetic characters

def calculate_match_percentage(json_data, sections_list, acts_list):
    filesList = {}


    # for key, value in json_data.items():
    for inner_key, inner_value in json_data.items():
        sections = inner_value["sections"]
        acts = inner_value["acts"]
        total_sections = max(len(sections_list), 1) 
        total_acts = max(len(acts_list), 1) 
        if total_sections == 0 or total_acts == 0:
          pass
        matched_sections = sum(1 for section in sections if section in sections_list) or 0.01
        matched_acts = sum(1 for act in acts if act in acts_list) or 0.01
        total_match_percentage = (matched_sections + matched_acts) / (total_acts+total_sections) * 100
        print(total_match_percentage)
        if total_match_percentage > 50:
          filesList[inner_key] = total_match_percentage
    print(filesList)

    sorted_listFile = [key for key, _ in sorted(filesList.items(), key=lambda x: -x[1])]

    return sorted_listFile

    
    
    
    
    

    
   
    
   
    
    
      


    

def search_synonyms_in_json(synonyms_dict):
  file_path_json = 'data.json'
  with open(file_path_json, 'r') as file:
    dataImport = json.load(file)
  file_path_json = 'filesdata.json'
  with open(file_path_json, 'r') as file:
    filedataImport = json.load(file)

  # one_d_list = []
  one_d_list = [value for values in synonyms_dict.values() for value in values]
  keys_list = list(synonyms_dict.keys())
  # for key, value in synonyms_dict.items():
  #   if isinstance(value, list):
  #     one_d_list.extend([key, *value])
  #   else:
  #     one_d_list.extend([key, value])




  temp_dest = { }
  print(f'one dlist of synonims: {one_d_list}')
  for word in keys_list:
    
    if len(word) >= 2:
      if word[0].lower().isalnum() and word[1].lower().isalnum(): 
        first_two_letters = word[0].lower() + word[1].lower()
        if word[0].lower() in dataImport:
          if first_two_letters in dataImport[word[0].lower()] and word.lower() in dataImport[word[0].lower()][first_two_letters]:
            temp_dest[word] =  dataImport[word[0].lower()][first_two_letters][word.lower()]
            temp_dest[word]['priority'] = 50
          else:
            pass

   




  for word in one_d_list:
    
    if len(word) >= 2:
      if word[0].lower().isalnum() and word[1].lower().isalnum(): 
        first_two_letters = word[0].lower() + word[1].lower()
        if word[0].lower() in dataImport:
          if first_two_letters in dataImport[word[0].lower()] and word.lower() in dataImport[word[0].lower()][first_two_letters]:
            temp_dest[word] =  dataImport[word[0].lower()][first_two_letters][word.lower()]
          else:
            pass
  
  file_path_counts = Counter()
  file_path_priorities = Counter()
  unique_sections = set()
  unique_acts = set()
# Iterate over the data and accumulate counts and priorities
  # for key, value in data.items():
  for key, value in temp_dest.items():
    unique_sections.update(value['sections'])
    unique_acts.update(value['acts'])
    for file_path in value['file_path']:
      file_path_counts[file_path] += 10
      file_path_priorities[file_path] += value['priority']


    # for file_path, priority in zip(value['file_path'], [value['priority']] * len(value['file_path'])):
    #   file_path_counts[file_path] += 1
    #   file_path_priorities[file_path] += priority
  temp_files_with_value = {}
  totalCP = 0
  countFiles = 0
# Print the sums of count and priority for each file path
  
  for file_path in file_path_counts.keys():
      total_count = file_path_counts[file_path]
      total_priority = file_path_priorities[file_path]
      totalCandP = total_count + total_priority*5
      temp_files_with_value[file_path] = totalCandP
      
      totalCP += totalCandP
      countFiles +=1
      print(f'File path: {file_path}, total CandP {temp_files_with_value[file_path]}')
  

  # sorted_List_wordMatch = [key for key, _ in sorted(temp_files_with_value.items(), key=lambda x: -x[1])]

  totalCP = totalCP or 1
  countFiles = countFiles or 1
  avgCandp = totalCP/countFiles

  sorted_List_wordMatch = {key: value for key, value in sorted(temp_files_with_value.items(), key=lambda x: x[1], reverse = True) if value >= avgCandp}
  sorted_List_wordMatch = list(sorted_List_wordMatch.keys())
  print(sorted_List_wordMatch)
  




  filesListSecAct = calculate_match_percentage(filedataImport, list(unique_sections),list(unique_acts))

  ordered_set = OrderedDict()

  for item in sorted_List_wordMatch:
    ordered_set[item] = None
  
  for item in filesListSecAct:
    ordered_set[item] = None




  # finalList = sorted_List_wordMatch + filesListSecAct
  # ordered_set = set()
  # for item in finalList:
  #   ordered_set.add(item)

  return list(ordered_set.keys())


# ------------------------------------------------------

    
    

@app.route('/process_query', methods = ['GET','POST'])
def process_query():
    query = request.form['query']


    # Process the query here (e.g., search for the query in the PDF file)
    blob = TextBlob(query)
    corrected_text = str(blob.correct())
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(corrected_text)
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop]
    query_processed_text = ' '.join(lemmatized_tokens)
    input_words = extract_words(query_processed_text)

    synonyms_dict = {}
    for word in input_words:
      synonyms_dict[word.lower()] = get_synonyms(word.lower())
    
    
    dic_with_val = search_synonyms_in_json(synonyms_dict)
    print("process text:")
    print(dic_with_val)
    file_path_json = 'filesdata.json'
    with open(file_path_json, 'r') as file:
      dataFileImport = json.load(file)



    # filtered_dict = {key: dataFileImport[key] for key in dic_with_val if key in dataFileImport}

    filtered_dict = OrderedDict((key, dataFileImport[key]) for key in dic_with_val if key in dataFileImport)
    print("Filtered")
    for key in filtered_dict:
      print(key)


    
 
    




    # return f"Query: {dic_with_val}"
    return render_template('file_list.html', files=filtered_dict)

















if __name__ == '__main__':
    app.run(debug=True)
