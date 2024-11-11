from flask import Flask, render_template, request, jsonify
import os
from openai import OpenAI
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import datetime
from pinecone import Pinecone
from flask import Flask, render_template, request, jsonify
import pymysql

def open_file(filepath):   
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# 파일 저장하는 함수
def save_file(filepath, content):  
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

#제이슨 파일 불러오는 함수
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

# 제이슨파일 저장하는함수
def save_json(filepath, payload):                                                   #payload: JSON 형식으로 저장하고자 하는 데이터를 담고 있는 변수
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)  #json.dump() 함수는 Python에서 JSON 데이터를 파일로 저장하는 데 사용

#시간,날짜체계 만들기
def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")

#문장 벡터로 임베딩
def gpt3_embedding(content, model='text-embedding-ada-002'):
    client = OpenAI(api_key=openai_api_key)
    # content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    # response = client.embeddings.create(input=content, model=model)
    # vector = response['data'][0]['embedding']  # this is a normal list
    content = content.replace("\n", " ")
    vector = client.embeddings.create(input = [content], model=model).data[0].embedding
    return vector

def gpt3_completion(prompt, model='ft:gpt-3.5-turbo-0613:personal::8x76h0Py'):
    max_retry = 5
    retry = 0
    # prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = client.chat.completions.create(
                model = model,
                messages=[
                    {"role": "system", "content": "날짜 체계를 고려해서 두번째 문장에대한 대답해"},
                    {"role": "user", "content": prompt}
                ]
            )
            text = response.choices[0].message.content.strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)

# 유사도로 가까운거 뽑은것들 텍스트로 쭉
def load_conversation(results):    #results = vdb.query(vector=vector, top_k=convo_length)
    result = list()   #-> result라는 리스트 만듬
    for m in results['matches']:   #results에는 matches라는게있음 그만큼 반복 -> 아마 top_k 일듯?
        info = load_json('nexus/%s.json' % m['id'])   
        result.append(info)                           #result 리스트에 info 삽입
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip()

def save_list_file_for_finetuning(file_name, content):     #-> 파인튜닝할용도로 prompt와 output저장
    try:
        # 파일을 쓰기 모드로 엽니다.
        with open(file_name, 'a') as file:
            # content를 파일에 씁니다.
            file.write(content)
        print(f'파일 "{file_name}"에 성공적으로 저장되었습니다.')
    except Exception as e:
        print(f'파일 "{file_name}" 저장 중 오류가 발생했습니다: {e}')


# 유사도로 가까운거 뽑은것들 텍스트로 쭉
def load_conversation2(results):    #results = vdb.query(vector=vector, top_k=convo_length)
    result = list()   #-> result라는 리스트 만듬
    #parsed_data = json.loads(results)           #첫번째 문제  the JSON object must be str, bytes or bytearray, not QueryResponse
    highest_score = float("-inf")  # 가장 작은 값으로 초기화
    highest_score_id = None
    for m in results["matches"]:
        if m["score"] > highest_score:
            highest_score = m["score"]
            highest_score_id = m["id"]   #lowest_score_id에 제일 유사한 문장의 벡터의 id가 들어가있음  여기까진 잘뽑아짐
    print(highest_score_id)
    info = load_json('nexus/%s.json' % highest_score_id)  #두번째 에러: [Errno 2] No such file or directory: 'nexus/3eba54a1-95eb-4996-b0e9-f2e4f04d973c.json'
    result.append(info)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip() 

openai_api_key = open_file('key_openai.txt')
client = OpenAI(api_key=openai_api_key)

def chat_main(a):
    if __name__ == '__main__':
        convo_length = 3 #유사도 가장 높은 거 3개 뽑는 용도
        openai_api_key = open_file('key_openai.txt')
        client = OpenAI(api_key=openai_api_key)
        pc = Pinecone(api_key="f907c6e2-ca80-4c89-9614-be9befcad63e")
        vdb = pc.Index("nadongban")
        #while True:
        now = datetime.datetime.now()                      #현재시간 가져오기  #now에는 현재시간 들어감
        formatted_date_time = now.strftime("%Y%m%d-%H:%M")  #날짜체계가들어감 ex) -> 20240307-21:50
        hangletime = str(formatted_date_time)             #ormatted_date_time과 똑같은게 들어감
        print(type(formatted_date_time))
        #### get user input, save it, vectorize it, save to pinecone
        payload = list()                                            #JSON 형식으로 저장하고자 하는 데이터를 담고 있는 변수를 리스트로 형변환
        #a = input('\n\nUSER: ')                  # 사용자가 프로그램에게 말하고싶은것,질문하고싶은건 ex) -> 내가 4월27일날 뭐하기로 했더라
        timestamp = time()          
        timestring = timestamp_to_datetime(timestamp)
        #message = '%s: %s - %s' % ('USER', timestring, a)
        message = hangletime + " : " + a        # ex)-> 20240307-21:50 : 내가 4월 27일날 뭐하기로 했지?
        print(message)                          #사용자가 말한것 출력
        vector = gpt3_embedding(message)        #사용자가 말한것 벡터로 임베딩
        unique_id = str(uuid4())      #uid를 생성 
        metadata = {'speaker': 'USER', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id} #메타데이터 만들고
        save_json('nexus/%s.json' % unique_id, metadata) #제이슨파일만들어서 저장
        #payload.append((unique_id, vector))
        ################################여기까지가 사용자의 답변 #################



        #### search for relevant messages, and generate a response
        results = vdb.query(vector=vector, top_k=convo_length) # 유사문장 찾기(vector값, top_k= 유사문장 갯수)
        print(type(results))   #<class 'pinecone.core.client.model.query_response.QueryResponse'>
        print(results)
        '''
            {'matches': [{'id': '3eba54a1-95eb-4996-b0e9-f2e4f04d973c',
                'score': 0.786891937,
                'values': []}],
                'namespace': '',
                'usage': {'readUnits': 5}}
        '''
        #'3eba54a1-95eb-4996-b0e9-f2e4f04d973c',  내 4월5일날 누구하고 가평가기로 했지?
        #3eba54a1-95eb-4996-b0e9-f2e4f04d973c     내가 4월27일날 뭐하기로 했지?
        #3eba54a1-95eb-4996-b0e9-f2e4f04d973c  안녕
        
        #내가 예상하기론 그때 파인콘 벡터 한번 지워서 파인콘에 3eba54a1-95eb-4996-b0e9-f2e4f04d973c 이거 하나밖에 안들어있어서 그런듯
        conversation = load_conversation2(results)  # results should be a DICT with 'matches' which is a LIST of DICTS, with 'id' # 결과는 'id'가 포함된 'matches'가 포함된 DICT여야 합니다
        prompt = open_file('prompt_response.txt').replace('<<CONVERSATION>>', conversation).replace('<<MESSAGE>>', a)  #사용자가 현재말한 문장과 conversation을 합치기 ex) 20240226-13:56 : 오 이 자주색 코트 이쁘다 내일 인터넷으로 찾아봐야겠다.  20240227-16:17 : 어제 내가 이쁘다고한 코트 무슨색이었지?
        #### generate response, vectorize, save, etc
        output = gpt3_completion(prompt)    #gptapi에 prompt 넣고 파인튜닝된 gpt의 답변 구하기   ex) 어제 자주색 코트가 이쁘다며 말씀하셨어요.
        timestamp = time()
        timestring = timestamp_to_datetime(timestamp)
        #message = '%s: %s - %s' % ('RAVEN', timestring, output)
        message = output      #ex)message에는 어제 자주색 코트가 이쁘다며 말씀하셨어요. 가 들어가있음
            

        #vector = gpt3_embedding(message)      # ex)message에는 어제 자주색 코트가 이쁘다며 말씀하셨어요. 가 임베딩되어있음
        #답변은 굳이 임베딩할필요가 없을수도?  답변은 그냥 사용자에게 보여주기만해도 될듯 -> vdb에 넣을게 아니라서 임베딩 안해도될듯

        #############################################안해도되는것###############################
        '''unique_id = str(uuid4())    #uid 생성
        metadata = {'speaker': '나동반', 'time': timestamp, 'message': message, 'timestring': timestring, 'uuid': unique_id} #메타데이터 만들기
        save_json('nexus/%s.json' % unique_id, metadata)   #답변을 json파일 만들어서 저장  metadata가 payload
        payload.append((unique_id, vector))
        vdb.upsert(payload)  #payload는 리스트임'''
        #############################################안해도되는것###############################
            
        #vdb.upsert(vector)
        payload.append((unique_id, vector))
        vdb.upsert(payload)
        #vdb.upsert(vectors=[{"id":unique_id,"values":vector}])

        #print('\n\n나동반: %s' % output) # output 출력 ex)어제 자주색 코트가 이쁘다며 말씀하셨어요.  ->성공!!

        file_name = "listup_file_for_finetuning"   #->그냥 텍스트 파일
        content = prompt+'\n'+output+'\n\n\n'  #-> ex) 20240226-13:56 : 오 이 자주색 코트 이쁘다 내일 인터넷으로 찾아봐야겠다.  20240227-16:17 : 어제 내가 이쁘다고한 코트 무슨색이었지? \n 어제 자주색 코트가 이쁘다며 말씀하셨어요. \n\n\n
        save_list_file_for_finetuning(file_name, content)
    return output

app = Flask(__name__)

'''
@app.route('/',methods = ['GET','POST'])
def chat_page():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        data = request.get_json()
        transcript = data['text']
        print(transcript)              #음성인식 잘됨 확인
        #text = request.form['text']   #이부분을 바꿔야함
        a_of_chat_main = chat_main(transcript)
        print(a_of_chat_main)
        #output = str(text+'라고 말씀하셨군요!!!')
        #print(output)
        #return render_template('index2.html')
        return str(a_of_chat_main)
        #return jsonify(message=str(a_of_chat_main))
'''

@app.route("/")
def index():
    return render_template('main.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return chat_main(input)

if __name__ == '__main__':
    app.run(port=5000,debug=True)

    