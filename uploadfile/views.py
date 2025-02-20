from django.shortcuts import render
import os
import torch
import json
import whisper  # ✅ Whisper 추가
from pydub import AudioSegment
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views import View
from django.utils.decorators import method_decorator
from transformers import BertTokenizer

# KoBERT 모델 불러오기
from kobert.pytorch_kobert import get_pytorch_kobert_model

# FFmpeg 경로 직접 설정
FFMPEG_PATH = r"D:\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
AudioSegment.converter = FFMPEG_PATH
print(f"🔧 FFmpeg 설정 완료: {FFMPEG_PATH}")

# Whisper 모델 로드
whisper_model = whisper.load_model("base")

# 임시 파일 저장 경로 설정
TEMP_DIR = os.path.join(os.getcwd(), "temp_files")
os.makedirs(TEMP_DIR, exist_ok=True)  # ✅ 폴더 없으면 생성

class BERTClassifier(torch.nn.Module):
    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.bert, _ = get_pytorch_kobert_model()
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, token_ids, valid_length, segment_ids):
        _, pooled_output = self.bert(input_ids=token_ids, return_dict=False)
        return self.classifier(pooled_output)

# KoBERT 모델 초기화
MODEL_PATH = os.path.join(os.path.dirname(__file__), "kobert_state_dict.pth")
model = BERTClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")


@method_decorator(csrf_exempt, name="dispatch")
class AudioFileUploadView(View):
    def post(self, request):
        """ 음성 파일을 업로드하고 변환 후 분석 """
        print("[📌 요청] 파일 업로드 및 분석 요청")

        if "file" not in request.FILES:
            return JsonResponse({"error": "파일이 포함되지 않았습니다."}, status=400)

        uploaded_file = request.FILES["file"]
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        allowed_extensions = [".mp3", ".wav", ".ogg", ".m4a"]

        if ext not in allowed_extensions:
            return JsonResponse({"error": "지원되지 않는 파일 형식입니다."}, status=400)

        try:
            # 🔹 파일을 TEMP_DIR에 저장
            temp_audio_path = os.path.join(TEMP_DIR, uploaded_file.name)
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.read())

            # 🔹 변환: mp3, ogg → wav
            wav_file_path = os.path.join(TEMP_DIR, os.path.splitext(uploaded_file.name)[0] + ".wav")
            print(f"[🎙️ 변환] {ext} → WAV 변환 중...")

            try:
                audio = AudioSegment.from_file(temp_audio_path, format=ext[1:])
                audio.export(wav_file_path, format="wav")
            except Exception as e:
                print(f"🚨 [오류] FFmpeg 변환 실패: {str(e)}")
                return JsonResponse({"error": "FFmpeg 변환 실패"}, status=500)

            # 🔹 변환 후 파일 확인
            if not os.path.exists(wav_file_path):
                print(f"🚨 [오류] WAV 파일이 존재하지 않음: {wav_file_path}")
                return JsonResponse({"error": "WAV 변환 후 파일이 존재하지 않습니다."}, status=500)

            print(f"[🎙️ STT 시작]: {wav_file_path}")

            # 🔹 STT 실행
            text = AudioFileUploadView.audio_to_text(wav_file_path)

            if text.startswith("Whisper 변환 실패"):
                return JsonResponse({"error": text}, status=500)

            print(f"[🔍 분석할 텍스트]: {text}")

            probability = self.analyze_text(text) * 100

            # 🔹 임시 파일 삭제
            # os.remove(temp_audio_path)
            # os.remove(wav_file_path)

            return JsonResponse({"probability": probability, "text": text}, status=200)

        except Exception as e:
            print(f"🚨 [ERROR] 업로드 중 오류 발생: {str(e)}")
            return JsonResponse({"error": "파일 업로드 중 오류가 발생했습니다."}, status=500)

    @staticmethod
    def audio_to_text(wav_file_path):
        """ 음성 파일을 텍스트로 변환하는 함수 """
        if not os.path.exists(wav_file_path):
            print(f"🚨 [오류] 파일이 존재하지 않음: {wav_file_path}")
            return "Whisper 변환 실패: 파일 없음"

        print(f"[🎙️ STT 시작]: {wav_file_path}")
        try:
            result = whisper_model.transcribe(wav_file_path)
            print(result)
            return result["text"]
        except Exception as e:
            print(f"🚨 [오류] Whisper 변환 실패: {str(e)}")
            return f"Whisper 변환 실패: {str(e)}"

    def analyze_text(self, text):
        """ KoBERT 모델을 사용해 텍스트 분석 """
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            valid_length = torch.tensor([len(inputs["input_ids"][0])])  # 문장 길이
            segment_ids = torch.zeros_like(inputs["input_ids"])  # 모든 단어의 segment ID를 0으로 설정

            with torch.no_grad():
                output = model(inputs["input_ids"], valid_length, segment_ids)
            
            val = output.squeeze(1)
            chk = torch.sigmoid(val)
            chk = chk.item()

            return chk
        
        except Exception as e:
            print(f"🚨 [ERROR] 분석 중 오류 발생: {str(e)}")
            return "오류 발생"
