import spacy

from tqdm import tqdm
import codecs as cs
from os.path import join as pjoin
import os
import pickle
import numpy as np
nlp = spacy.load('en_core_web_sm')
# pip install -i https://mirrors.cloud.tencent.com/pypi/simple en_core_web_sm-2.3.0.tar.gz 
# en_core_web_sm-2.3.0.tar.gz Download link: https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz

def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list


def process_hhi(filename, output_file):
    out_f = open(output_file, 'w')
    start, end = '0.0', '0.0'
    with open(filename, 'r') as f:
        for line in f.readlines():
            caption = line.rstrip('\n')
            word_list, pose_list = process_text(caption)
            tokens = ' '.join(['%s/%s'%(word_list[i], pose_list[i]) for i in range(len(word_list))])
            out_f.write('%s#%s#%s#%s\n'%(caption, tokens, start, end))
    out_f.close()

def load_glove_vectors(glove_file):
    word_vectors = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = [float(x) for x in values[1:]]
            word_vectors[word] = vector
    return word_vectors

unseen_words = {'sshaped': ['s', 'shaped'], 'thumbsdown': ['thumbs', 'down'], 'backtoback': ['back', 'to', 'back'], 'shoulderwidth': ['shoulder', 'width'], 'doublehanded': ['double', 'handed'], 'vsign': ['v', 'sign'], 'selfie': ['self', 'portrait'], 'chestlevel': ['chest', 'level'], 'highfive': ['high', 'five'], 'semisquats': ['semi', 'squats'], 'facetoface': ['face', 'to', 'face'], 'tugofwar': ['tug', 'of', 'war'], 'upperleft': ['upper', 'left'], 'thumbup': ['thumb', 'up'], 'sideslams': ['side', 'slams'], 'scissorhand': ['scissor', 'hand'], 'shouldertoshoulder': ['shoulder', 'to', 'shoulder'], 'halfsquatting': ['half', 'squatting'], 'onequarter': ['one', 'quarter'], 'semicrouched': ['semi', 'crouched'], 'highfives': ['high', 'fives'], 'fronttoback': ['front', 'to', 'back'], 'frontandback': ['front', 'and', 'back'], 'leftfront': ['left', 'front'], 'leftupper': ['left', 'upper'], 'rockpaperscissors': ['rock', 'paper', 'scissors'], 'frontright': ['front', 'right'], 'fistclenching': ['fist', 'clenching'], 'thumbsup': ['thumbs', 'up'], 'handwrestling': ['hand', 'wrestling'], 'scissorlike': ['scissor', 'like'], 'uncrosses': ['un', 'crosses'], 'fingerguesses': ['finger', 'guesses'], 'prayerlike': ['prayer', 'like'], 'reselects': ['re', 'selects'], 'halfcrouching': ['half', 'crouching'], 'sidetoside': ['side', 'to', 'side'], 'halfcrouch': ['half', 'crouch'], 'fastwalks': ['fast', 'walks'], 'wristwrestle': ['wrist', 'wrestle'], 'halfturn': ['half', 'turn'], 'twohanded': ['two', 'handed'], 'backhandedly': ['back', 'handedly'], 'wristwrestling': ['wrist', 'wrestling'], 'armraising': ['arm', 'raising'], 'armwrestle': ['arm', 'wrestle'], 'semisquat': ['semi', 'squat'], 'nonmirrored': ['non', 'mirrored'], 'halfcrouches': ['half', 'crouches'], 'rightfront': ['right', 'front'], 'ambulates': ['walks'], 'handclapping': ['hand', 'clapping'], 'halfkneels': ['half', 'kneels'], 'halfsquats': ['half', 'squats'], 'armwrestling': ['arm', 'wrestling'], 'scissorshands': ['scissors', 'hands'], 'palmtopalm': ['palm', 'to', 'palm'], 'onefinger': ['one', 'finger'], 'midmatch': ['mid', 'match'], 'tsign': ['t', 'sign'], 'backpatting': ['back', 'patting'], 'backandforth': ['back', 'and', 'forth'], 'knuckletoknuckle': ['knuckle', 'to', 'knuckle'], 'semicrouches': ['semi', 'crouches'], 'halfstep': ['half', 'step'], 'reddens': ['redden'], 'halfcrouched': ['half', 'crouched'], 'gridlike': ['grid', 'like'], 'interlaces': ['inter', 'laces'], 'vsigns': ['v', 'signs'], 'halfcircle': ['half', 'circle'], 'gesticulates': ['gesticulate'], 'twostep': ['two', 'step'], 'heartshaped': ['heart', 'shaped'], 'outstretches': ['out', 'stretches'], 'upanddown': ['up', 'and', 'down'], 'kickhopping': ['kick', 'hopping'], 'highfiving': ['high', 'fiving'], 'threehandshake': ['three', 'handshake'], 'rockscissors': ['rock', 'scissors'], 'gesticulation': ['gesticulate'], 'fistbump': ['fist', 'bump'], 'handslapping': ['hand', 'slapping'], 'halfsquat': ['half', 'squat'], 'crosslegged': ['cross', 'legged'], 'unclasping': ['un', 'clasping'], 'hearttoheart': ['heart', 'to', 'heart'], 'forwardfacing': ['forward', 'facing'], 'heartlike': ['heart', 'like'], 'onehanded': ['one', 'handed'], 'handtohand': ['hand', 'to', 'hand'], 'uturn': ['u', 'turn'], 'semireclines': ['semi', 'reclines'], 'upsidedown': ['up', 'side', 'down'], 'fingerguessing': ['finger', 'guessing'], 'shuashua': ['shua', 'shua'], 'rightrear': ['right', 'rear'], 'sitted': ['sitting']}

def generate_glove_files(folder):
    glove_file = './glove.6B/glove.6B.300d.txt'
    word_vectors = load_glove_vectors(glove_file)
    print("Loaded.")

    extract_features = {}
    for filename in tqdm(sorted(os.listdir(folder))):
        output_file = pjoin(processed_folder, filename)
        with open(os.path.join(text_folder, filename), 'r') as f:
            for line in f.readlines():
                sentence = line.rstrip('\n')
                sentence = sentence.replace('-', '')
                doc = nlp(sentence)
                for token in doc:
                    word = token.text.lower()
                    if not word.isalpha():
                        continue
                    if word not in extract_features:
                        try:
                            extract_features[word] = word_vectors[word]
                        except:
                            tmp = [0] * 300
                            for split in unseen_words[word]:
                                tmp += word_vectors[split]
                            tmp = [x / len(unseen_words[word]) for x in tmp]
                            extract_features[word] = tmp

    vab_data = []
    vab_words = []
    vab_idx_dict = {}
    extra_words = ['sos', 'eos', 'unk']
    for index in range(len(extra_words)):
        vab_data.append(word_vectors[extra_words[index]])
        vab_words.append(extra_words[index])
        vab_idx_dict[extra_words[index]] = index
    for idx, key in enumerate(extract_features):
        print(idx, key)
        vab_data.append(np.array(extract_features[key]))
        vab_words.append(key)
        vab_idx_dict[key] = idx+3
    vab_data = np.concatenate(vab_data, axis=0).reshape((-1, 300))
    with open(os.path.join(glove_folder, 'hhi_vab_idx.pkl'), 'wb') as fo:
        pickle.dump(vab_idx_dict, fo)
        fo.close()
    with open(os.path.join(glove_folder, 'hhi_vab_words.pkl'), 'wb') as fo:
        pickle.dump(vab_words, fo)
        fo.close()
    np.save(os.path.join(glove_folder, 'hhi_vab_data.npy'), vab_data)


text_folder = '/data/xuliang/Inter-X/Inter-X_Dataset/texts'
processed_folder = '/data/xuliang/Inter-X/texts_processed'
glove_folder = '/data/xuliang/Inter-X/glove'

if __name__ == "__main__":
    os.makedirs(processed_folder, exist_ok=True)
    os.makedirs(glove_folder, exist_ok=True)
    ### 1. process raw text files ###
    for filename in tqdm(sorted(os.listdir(text_folder))):
        output_file = pjoin(processed_folder, filename)
        process_hhi(os.path.join(text_folder, filename), output_file)

    ### 2. generate glove files ###
    generate_glove_files(text_folder)
